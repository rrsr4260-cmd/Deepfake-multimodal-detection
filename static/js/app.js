
const form = document.getElementById("uploadForm");
const input = document.getElementById("mediaInput");
const dropzone = document.getElementById("dropzone");
const fileTitle = document.getElementById("fileTitle");
const fileHint = document.getElementById("fileHint");
const fileChip = document.getElementById("fileChip");
const analyzeBtn = document.getElementById("analyzeBtn");
const progressWrap = document.getElementById("progressWrap");
const progressText = document.getElementById("progressText");
const result = document.getElementById("result");

let selectedFile = null;

function chooseFile(file) {
  if (!file) return;

  selectedFile = file;
  fileTitle.textContent = file.name;
  fileHint.textContent =
    `${(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze`;

  fileChip.hidden = false;
  fileChip.textContent = file.type || "Media file";
  result.hidden = true;
}

input.addEventListener("change", () => chooseFile(input.files[0]));

["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];

  if (file) {
    chooseFile(file);

    // Keep the input and selected file in sync.
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
  }
});

function showError(message) {
  result.hidden = false;
  result.className = "result error";
  result.innerHTML =
    '<span class="result-tag">ANALYSIS FAILED</span>' +
    '<h3>We could not analyze this file</h3><p></p>';

  result.querySelector("p").textContent = message;
}

function showResult(data) {
  // The backend may return an error inside a JSON response.
  if (data.error) {
    showError(data.error);
    return;
  }

  result.hidden = false;
  result.className = "result";

  result.innerHTML =
    '<span class="result-tag"></span>' +
    '<h3></h3>' +
    '<div class="result-grid"></div>' +
    '<p class="explanation"></p>' +
    '<p class="disclaimer"></p>';

  const tag = result.querySelector(".result-tag");

  tag.textContent = data.prediction
    ? "ANALYSIS COMPLETE"
    : "ANALYSIS RESULT";

  result.querySelector("h3").textContent =
    data.prediction
      ? `Prediction: ${data.prediction}`
      : (data.label || "Analysis result");

  const grid = result.querySelector(".result-grid");

  // Backend returns fake_probability as a percentage from 0 to 100.
  // Do not multiply it by 100 again.
  const fakeScore = data.fake_probability;
  const realScore = data.real_probability;

  const stats = [
    ["Media type", data.media_type || "—"],
    [
      "Fake score",
      fakeScore == null ? "Not available" : `${Number(fakeScore).toFixed(1)}%`
    ],
    [
      "Real score",
      realScore == null ? "Not available" : `${Number(realScore).toFixed(1)}%`
    ],
    [
      "Confidence",
      data.confidence == null ? "Not available" : `${Number(data.confidence).toFixed(1)}%`
    ],
    [
      "Processing time",
      data.processing_seconds == null ? "—" : `${data.processing_seconds}s`
    ]
  ];

  stats.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "result-stat";

    const small = document.createElement("small");
    small.textContent = label;

    const bold = document.createElement("b");
    bold.textContent = value;

    item.append(small, bold);
    grid.append(item);
  });

  result.querySelector(".explanation").textContent =
    data.explanation || "";

  result.querySelector(".disclaimer").textContent =
    data.disclaimer ||
    "AI-generated scores are estimates, not proof that media is real or manipulated.";

  if (data.details && data.details.frames_sampled) {
    const p = document.createElement("p");
    p.textContent = `Video frames sampled: ${data.details.frames_sampled}`;
    result.append(p);
  }
}

form.addEventListener("submit", async e => {
  e.preventDefault();

  if (!selectedFile) {
    showError("Please choose an image, audio file, or video first.");
    return;
  }

  if (selectedFile.size > 100 * 1024 * 1024) {
    showError("The maximum file size is 100 MB.");
    return;
  }

  const body = new FormData();
  body.append("media", selectedFile);

  analyzeBtn.disabled = true;
  progressWrap.hidden = false;
  progressText.textContent = "Analyzing media";
  result.hidden = true;

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Analysis request failed.");
    }

    showResult(data);
  } catch (err) {
    showError(
      err.message || "Network error. Check whether the Flask server is running."
    );
  } finally {
    analyzeBtn.disabled = false;
    progressWrap.hidden = true;
  }
});
