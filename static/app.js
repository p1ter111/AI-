const fileInput = document.querySelector("#fileInput");
const folderInput = document.querySelector("#folderInput");
const dropzone = document.querySelector("#dropzone");
const jobGrid = document.querySelector("#jobGrid");
const emptyState = document.querySelector("#emptyState");
const queueHint = document.querySelector("#queueHint");
const queuedCount = document.querySelector("#queuedCount");
const doneCount = document.querySelector("#doneCount");
const rawState = document.querySelector("#rawState");
const downloadAll = document.querySelector("#downloadAll");
const clearAll = document.querySelector("#clearAll");
const pickFiles = document.querySelector("#pickFiles");
const pickFolder = document.querySelector("#pickFolder");
const referenceState = document.querySelector("#referenceState");
const referenceMessage = document.querySelector("#referenceMessage");
const referenceList = document.querySelector("#referenceList");
const calibrationPanel = document.querySelector("#calibrationPanel");
const calibrationCount = document.querySelector("#calibrationCount");
const calibrationGrid = document.querySelector("#calibrationGrid");
const eventCount = document.querySelector("#eventCount");
const eventList = document.querySelector("#eventList");

let polling = null;

const formatBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
};

const statusText = {
  queued: "等待",
  processing: "处理中",
  done: "完成",
  failed: "失败",
};

const sceneText = {
  portrait: "人像",
  landscape: "风景",
  product: "静物",
  general: "通用",
  night: "夜景",
};

function renderJobs(jobs) {
  const total = jobs.length;
  const done = jobs.filter((job) => job.status === "done").length;
  const active = jobs.filter((job) => job.status === "queued" || job.status === "processing").length;

  queuedCount.textContent = String(active);
  doneCount.textContent = String(done);
  downloadAll.disabled = done === 0;
  emptyState.hidden = total > 0;
  queueHint.textContent = total ? `${total} 张图片，${done} 张已完成` : "等待导入图片";

  jobGrid.innerHTML = jobs.map(jobTemplate).join("");
}

function renderReference(reference) {
  if (!reference) return;
  referenceState.textContent = `${reference.count || 0} 个参考`;
  referenceMessage.textContent = reference.status?.message || "自动参考库已启用。";
  referenceList.innerHTML = (reference.profiles || [])
    .map((profile) => `<span class="reference-pill">${escapeHtml(profile.name)} · ${escapeHtml(sceneText[profile.kind] || profile.kind)}</span>`)
    .join("");
}

function renderCalibration(calibration) {
  const samples = calibration?.samples || [];
  calibrationPanel.hidden = samples.length === 0;
  if (!samples.length) {
    calibrationCount.textContent = "0 组";
    calibrationGrid.innerHTML = "";
    return;
  }
  calibrationCount.textContent = `${samples.length} 组`;
  calibrationGrid.innerHTML = samples
    .map(
      (sample) => `
        <article class="calibration-card">
          <div class="thumb">
            <div class="compare">
              <figure>
                <img src="${sample.sourceUrl}?v=${sample.updatedAt || 0}" alt="${escapeHtml(sample.name)} 原图" />
                <figcaption>原图</figcaption>
              </figure>
              <figure>
                <img src="${sample.outputUrl}?v=${sample.updatedAt || 0}" alt="${escapeHtml(sample.name)} 校准后" />
                <figcaption>校准后</figcaption>
              </figure>
            </div>
          </div>
          <div class="calibration-meta">
            <strong>${escapeHtml(sample.name)}</strong>
            <span>${escapeHtml(sceneText[sample.scene] || sample.scene)} · 自动参考校准</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderEvents(events) {
  const list = events || [];
  eventCount.textContent = `${list.length} 条`;
  eventList.innerHTML = list
    .slice()
    .reverse()
    .slice(0, 8)
    .map((event) => `<div class="event-item ${escapeHtml(event.kind || "")}"><strong>${escapeHtml(event.kind || "event")}</strong><span>${escapeHtml(event.message || "")}</span></div>`)
    .join("");
}

function jobTemplate(job) {
  const badgeClass = `badge ${job.status}`;
  const preview = job.previewUrl && job.originalPreviewUrl
    ? `
      <div class="compare">
        <figure>
          <img src="${job.originalPreviewUrl}?v=${job.updatedAt}" alt="${escapeHtml(job.originalName)} 原图预览" />
          <figcaption>原图</figcaption>
        </figure>
        <figure>
          <img src="${job.previewUrl}?v=${job.updatedAt}" alt="${escapeHtml(job.originalName)} 成品预览" />
          <figcaption>成品</figcaption>
        </figure>
      </div>
    `
    : job.previewUrl
      ? `<img src="${job.previewUrl}?v=${job.updatedAt}" alt="${escapeHtml(job.originalName)} 修图预览" />`
      : `<span>${job.progress || 0}%</span>`;
  const size = job.width && job.height ? `${job.width} x ${job.height}` : formatBytes(job.inputSize);
  const details = job.error
    ? escapeHtml(job.error)
    : [
        job.adjustments?.input ? `${job.adjustments.input}` : "",
        job.adjustments?.scene ? `${sceneText[job.adjustments.scene] || job.adjustments.scene}` : "",
        job.adjustments?.reference || "",
        job.adjustments?.guardrails || "",
        job.adjustments?.rawFallback ? "RAW 预览兜底" : "",
        job.adjustments?.sizeGuard ? "超大图优化" : "",
        job.adjustments?.quality ? `质量 ${job.adjustments.quality}` : "",
        job.adjustments?.referenceScore ? `参考匹配 ${job.adjustments.referenceScore}` : "",
        job.adjustments?.referenceFit ? `参考适配 ${job.adjustments.referenceFit}` : "",
        Number.isFinite(job.adjustments?.referenceIntensity) ? `参考强度 ${Math.round(job.adjustments.referenceIntensity * 100)}%` : "",
        job.adjustments?.highQualityProtected ? "高质量保真" : "",
        job.adjustments?.weirdness ? `稳定度 ${stabilityText(job.adjustments.weirdness)}` : "",
        job.adjustments?.aestheticScore ? `审美 ${job.adjustments.aestheticScore}` : "",
      ]
        .filter(Boolean)
        .join(" / ");

  return `
    <article class="job-card">
      <div class="thumb">${preview}</div>
      <div class="job-body">
        <div class="job-title">
          <strong title="${escapeHtml(job.originalName)}">${escapeHtml(job.originalName)}</strong>
          <span class="${badgeClass}">${statusText[job.status] || job.status}</span>
        </div>
        <div class="job-meta">${escapeHtml(job.message)} · ${escapeHtml(size)}</div>
        <div class="progress" style="--progress: ${Math.max(0, Math.min(100, job.progress || 0))}%"><span></span></div>
        <div class="job-adjustments">${escapeHtml(details || "正在分析图片")}</div>
        ${job.downloadUrl ? `<a class="download" href="${job.downloadUrl}">下载成品</a>` : ""}
        ${job.status === "failed" || job.status === "done" ? `<button class="retry" data-job-id="${job.id}" type="button">重新处理</button>` : ""}
        ${job.status === "failed" ? `<button class="retry inspect" data-job-id="${job.id}" type="button">诊断</button>` : ""}
      </div>
    </article>
  `;
}

function stabilityText(metrics) {
  const penalty = Number(metrics?.penalty || 0);
  if (penalty <= 3) return "自然";
  if (penalty <= 10) return "已校正";
  return "已回退";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function refreshJobs() {
  const response = await fetch("/api/jobs");
  const data = await response.json();
  rawState.textContent = data.rawAvailable ? "已启用" : "需安装";
  renderReference(data.reference);
  renderCalibration(data.calibration);
  renderEvents(data.events);
  renderJobs(data.jobs);

  const active = data.jobs.some((job) => job.status === "queued" || job.status === "processing");
  if (!active && polling) {
    clearInterval(polling);
    polling = null;
  }
}

function startPolling() {
  if (!polling) {
    polling = setInterval(refreshJobs, 900);
  }
}

async function uploadFiles(files) {
  const list = Array.from(files || []);
  if (!list.length) return;

  const chunkSize = 2;
  for (let index = 0; index < list.length; index += chunkSize) {
    const chunk = list.slice(index, index + chunkSize);
    const form = new FormData();
    for (const file of chunk) {
      form.append("files", file, file.name);
    }
    const response = await fetch("/api/upload", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (data.rejected?.length) {
      console.warn("Rejected files", data.rejected);
    }
    await refreshJobs();
    startPolling();
  }
}

fileInput.addEventListener("change", (event) => {
  uploadFiles(event.target.files);
  fileInput.value = "";
});

folderInput.addEventListener("change", (event) => {
  uploadFiles(event.target.files);
  folderInput.value = "";
});

pickFiles.addEventListener("click", (event) => {
  event.preventDefault();
  fileInput.click();
});

pickFolder.addEventListener("click", (event) => {
  event.preventDefault();
  folderInput.click();
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  uploadFiles(event.dataTransfer.files);
});

downloadAll.addEventListener("click", () => {
  window.location.href = "/api/download-all";
});

clearAll.addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  await refreshJobs();
});

jobGrid.addEventListener("click", async (event) => {
  const inspectButton = event.target.closest(".inspect");
  if (inspectButton) {
    inspectButton.disabled = true;
    await fetch(`/api/inspect/${inspectButton.dataset.jobId}`, { method: "POST" });
    await refreshJobs();
    return;
  }
  const button = event.target.closest(".retry");
  if (!button) return;
  button.disabled = true;
  await fetch(`/api/retry/${button.dataset.jobId}`, { method: "POST" });
  await refreshJobs();
  startPolling();
});

refreshJobs();
