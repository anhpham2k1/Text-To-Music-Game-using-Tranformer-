"""
Trainer — Training loop for Music Transformer.

Features:
- AdamW optimizer with weight decay
- Cosine LR schedule with linear warm-up
- Gradient accumulation for large effective batch size
- Gradient clipping
- Label smoothing
- TensorBoard logging
- Early stopping
- Checkpointing
"""

import os
import math
import time
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class WarmupCosineScheduler:
    """
    Learning Rate scheduler: linear warm-up → cosine decay.

    Supports resuming via set_step(initial_step).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        initial_step: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = initial_step
        self._update_lr()

    def step(self):
        self.step_count += 1
        self._update_lr()

    def set_step(self, step: int):
        """Set current step (for resume) and immediately update LR."""
        self.step_count = step
        self._update_lr()

    def _update_lr(self):
        if self.total_steps <= 0 or self.step_count <= 0:
            lr = self.base_lr
        elif self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / max(1, self.warmup_steps))
        else:
            progress = min(1.0, (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            ))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """
    Trainer for Music Transformer.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        # Training params
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.98),
        eps: float = 1e-9,
        num_epochs: int = 50,
        warmup_ratio: float = 0.05,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        label_smoothing: float = 0.1,
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        save_every: int = 5,
        eval_every: int = 1,
        early_stopping_patience: int = 10,
        # Logging
        log_dir: str = "logs",
        # Device
        device: str = "auto",
        # Tokenizer info
        pad_token_id: int = 0,
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training params
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        self.pad_token_id = pad_token_id

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # Scheduler - robust for small datasets
        steps_per_epoch = max(1, len(train_loader) // max(1, gradient_accumulation_steps))
        self.total_steps = max(1, steps_per_epoch * num_epochs)
        warmup_steps = max(1, int(self.total_steps * warmup_ratio))

        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, self.total_steps, initial_step=0
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
        )

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Logging
        self.writer = None
        if SummaryWriter is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.start_epoch = 1

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        # Resume support
        self.start_epoch = 1

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Total steps: {self.total_steps}, Warmup: {warmup_steps}")
        print(f"[Trainer] Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")

    def load_checkpoint(self, path: str, continue_epochs: bool = True):
        """Resume training from checkpoint with proper scheduler sync.

        Args:
            path: Path to checkpoint
            continue_epochs: If True, will continue from saved epoch toward original num_epochs.
        """
        if not os.path.exists(path):
            print(f"[Trainer] No checkpoint found at {path}, starting fresh.")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))

        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Restore scheduler state (deep resume fix)
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.set_step(self.global_step)

        # Try to restore epoch
        epoch = checkpoint.get("epoch", 1)
        if continue_epochs:
            self.start_epoch = epoch + 1
        else:
            self.start_epoch = 1

        # Restore scaler if present
        if "scaler" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"[Trainer] Resumed from {path} (global_step={self.global_step}, start_epoch={self.start_epoch}, best_val={self.best_val_loss:.4f})")

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            dict with training history
        """
        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        print("\n" + "=" * 60)
        print("  TRAINING START")
        print("=" * 60)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # --- Train ---
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # --- Evaluate ---
            if epoch % self.eval_every == 0:
                val_loss = self._evaluate()
                history["val_loss"].append(val_loss)

                lr = self.scheduler.get_lr()
                history["learning_rate"].append(lr)

                print(
                    f"  Epoch {epoch:3d}/{self.num_epochs} │ "
                    f"Train Loss: {train_loss:.4f} │ "
                    f"Val Loss: {val_loss:.4f} │ "
                    f"LR: {lr:.2e} │ "
                    f"PPL: {math.exp(min(val_loss, 10)):.1f}"
                )

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    self.writer.add_scalar("LR", lr, epoch)
                    self.writer.add_scalar(
                        "Perplexity", math.exp(min(val_loss, 10)), epoch
                    )

                # --- Checkpoint ---
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pt", epoch=epoch)
                    print(f"  ★ New best model saved (val_loss={val_loss:.4f})")
                else:
                    self.patience_counter += 1

                # --- Early stopping ---
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n  ⚠ Early stopping at epoch {epoch} (patience={self.early_stopping_patience})")
                    break

            # --- Periodic save ---
            if epoch % self.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch=epoch)

        print("\n" + "=" * 60)
        print(f"  TRAINING COMPLETE — Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        if self.writer:
            self.writer.close()

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch with proper gradient accumulation handling."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated = 0

        self.optimizer.zero_grad()

        # Progress bar for better UX
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        try:
            for step, batch in enumerate(pbar):
                tokens = batch["tokens"].to(self.device)

                # Fast structured conditioning
                mood = batch["mood"].to(self.device)
                genre = batch["genre"].to(self.device)
                scene = batch["scene"].to(self.device)
                tempo = batch["tempo"].to(self.device)
                instrument = batch["instrument"].to(self.device)
                energy = batch["energy"].to(self.device)

                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]

                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    logits, _ = self.model(
                        input_tokens,
                        mood=mood, genre=genre, scene=scene,
                        tempo=tempo, instrument=instrument, energy=energy
                    )
                    loss = self.loss_fn(
                        logits.reshape(-1, logits.size(-1)),
                        target_tokens.reshape(-1),
                    )
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                accumulated += 1

                # Robustness: detect NaN
                if torch.isnan(loss):
                    print(f"[WARNING] NaN loss detected at step {step}. Skipping update.")

                # Update progress bar
                current_lr = self.scheduler.get_lr()
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}"
                })

                # Perform optimizer step when accumulation is complete
                if accumulated % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    accumulated = 0
        finally:
            pbar.close()

        # Handle remaining gradients if last accumulation was incomplete
        if accumulated > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            tokens = batch["tokens"].to(self.device)

            # === FAST STRUCTURED PROMPT (same as training) ===
            mood = batch["mood"].to(self.device)
            genre = batch["genre"].to(self.device)
            scene = batch["scene"].to(self.device)
            tempo = batch["tempo"].to(self.device)
            instrument = batch["instrument"].to(self.device)
            energy = batch["energy"].to(self.device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits, _ = self.model(
                    input_tokens,
                    mood=mood, genre=genre, scene=scene,
                    tempo=tempo, instrument=instrument, energy=energy
                )
                loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _save_checkpoint(self, filename: str, epoch: int = None):
        """Save model checkpoint with resume info."""
        path = os.path.join(self.checkpoint_dir, filename)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "epoch": epoch or 0,
            "config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "max_seq_len": self.model.max_seq_len,
            },
        }
        if self.scaler:
            save_dict["scaler"] = self.scaler.state_dict()

        torch.save(save_dict, path)
