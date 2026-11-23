// ---------- Grab page elements ----------
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewBox = document.getElementById("preview-box");
const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const fileNameText = document.getElementById("file-name");
const errorMessage = document.getElementById("error-message");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const loadingCard = document.getElementById("loading-card");
const resultCard = document.getElementById("result-card");

let selectedFile = null; // the file the user picked

// ---------- Choosing a file (click or drag & drop) ----------
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();                       // needed to allow dropping
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

// ---------- Show preview of the chosen file ----------
function handleFile(file) {
  if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) {
    showError("Please choose an image or video file.");
    return;
  }
  hideError();
  resultCard.classList.add("hidden");
  selectedFile = file;

  const url = URL.createObjectURL(file); // temporary local URL for the preview
  previewImage.classList.add("hidden");
  previewVideo.classList.add("hidden");
  if (file.type.startsWith("image/")) {
    previewImage.src = url;
    previewImage.classList.remove("hidden");
  } else {
    previewVideo.src = url;
    previewVideo.classList.remove("hidden");
  }
  fileNameText.textContent = file.name;
  previewBox.classList.remove("hidden");
  analyzeBtn.disabled = false;
  resetBtn.classList.remove("hidden");
}

// ---------- Send the file to Flask and show the result ----------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);   // key "file" must match app.py

  hideError();
  analyzeBtn.disabled = true;
  resultCard.classList.add("hidden");
  loadingCard.classList.remove("hidden");

  try {
    const response = await fetch("/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    showResult(data);
  } catch (err) {
    showError(err.message || "Could not reach the server.");
    analyzeBtn.disabled = false;
  } finally {
    loadingCard.classList.add("hidden");
  }
});

// ---------- Display the result ----------
function showResult(data) {
  const label = document.getElementById("result-label");
  label.textContent = data.label;
  label.className = data.label.toLowerCase();   // "real" or "fake" -> colors

  document.getElementById("result-confidence").textContent = data.confidence + "%";
  document.getElementById("bar-fill").style.width = data.confidence + "%";
  document.getElementById("result-explanation").textContent = data.explanation;

  const badge = document.getElementById("result-badge");
  badge.className = "mode-badge " + data.mode;
  badge.textContent = data.mode === "demo" ? "SIMULATED RESULT (DEMO)" : "MODEL PREDICTION";

  document.getElementById("result-frames").textContent =
    data.frames_analyzed > 1 ? "Frames analyzed: " + data.frames_analyzed : "";

  resultCard.classList.remove("hidden");
  analyzeBtn.disabled = false;
}

// ---------- Reset ----------
resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewVideo.src = "";
  previewBox.classList.add("hidden");
  resultCard.classList.add("hidden");
  loadingCard.classList.add("hidden");
  resetBtn.classList.add("hidden");
  analyzeBtn.disabled = true;
  hideError();
});

function showError(msg) { errorMessage.textContent = msg; errorMessage.classList.remove("hidden"); }
function hideError() { errorMessage.classList.add("hidden"); }
