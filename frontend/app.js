/**
 * Text-to-Music Generator — Frontend Logic
 *
 * Handles form submission, API communication, audio playback,
 * and UI state management.
 */

const API_BASE = window.location.origin;

// ============================================================
// DOM Elements
// ============================================================

const form = document.getElementById("music-form");
const btnGenerate = document.getElementById("btn-generate");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");

const audioPlayer = document.getElementById("audio-player");
const promptDisplay = document.getElementById("prompt-display");

const infoDuration = document.getElementById("info-duration");
const infoNotes = document.getElementById("info-notes");
const infoId = document.getElementById("info-id");

const btnMidi = document.getElementById("btn-midi");
const btnWav = document.getElementById("btn-wav");

const tempSlider = document.getElementById("temperature");
const tempValue = document.getElementById("temp-value");
const toppSlider = document.getElementById("top_p");
const toppValue = document.getElementById("topp-value");

// ============================================================
// Range Slider Updates
// ============================================================

tempSlider.addEventListener("input", () => {
    tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
});

toppSlider.addEventListener("input", () => {
    toppValue.textContent = parseFloat(toppSlider.value).toFixed(2);
});

// ============================================================
// Form Submission
// ============================================================

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await generateMusic();
});

async function generateMusic() {
    // Build request body
    const body = {
        mood: document.getElementById("mood").value,
        genre: document.getElementById("genre").value,
        scene: document.getElementById("scene").value,
        tempo: document.getElementById("tempo").value,
        instrument: document.getElementById("instrument").value,
        energy: document.getElementById("energy").value,
        temperature: parseFloat(tempSlider.value),
        top_p: parseFloat(toppSlider.value),
        max_length: parseInt(document.getElementById("max_length").value),
    };

    // UI: Loading state
    setLoading(true);
    hideSection(resultSection);
    hideSection(errorSection);

    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        showResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// ============================================================
// UI Helpers
// ============================================================

function showResult(data) {
    // Prompt text
    promptDisplay.textContent = `"${data.prompt_text}"`;

    // Info
    infoDuration.textContent = `${data.duration}s`;
    infoNotes.textContent = data.num_notes.toLocaleString();
    infoId.textContent = data.request_id;

    // Audio player
    const wavUrl = `${API_BASE}${data.wav_url}`;
    const midiUrl = `${API_BASE}${data.midi_url}`;

    audioPlayer.src = wavUrl;

    // Download links
    btnMidi.href = midiUrl;
    btnWav.href = wavUrl;

    // Show section with animation
    showSection(resultSection);

    // Scroll to result
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
    document.getElementById("error-message").textContent = message;
    showSection(errorSection);
}

function setLoading(loading) {
    btnGenerate.disabled = loading;
    if (loading) {
        btnGenerate.classList.add("loading");
    } else {
        btnGenerate.classList.remove("loading");
    }
}

function showSection(el) {
    el.style.display = "block";
    el.style.animation = "none";
    // Trigger reflow
    void el.offsetHeight;
    el.style.animation = "fadeInUp 0.5s ease-out";
}

function hideSection(el) {
    el.style.display = "none";
}

// ============================================================
// Health Check (optional)
// ============================================================

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        console.log("[Health]", data);
    } catch {
        console.log("[Health] API not reachable");
    }
}

// Run health check on load
checkHealth();
