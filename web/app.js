"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const files = { img1: null, img2: null };
let groundTruth = null;   // {tx, ty} quando o par é sintético
let engine = null;
const objectUrls = [];

const form = $("#form");
const status = $("#status");
const runBtn = $("#btn-run");

function setStatus(msg, { error = false, busy = false } = {}) {
  status.className = "status" + (error ? " error" : "");
  status.innerHTML = (busy ? '<span class="spinner"></span>' : "") + msg;
}

function updateRunState() {
  runBtn.disabled = !(files.img1 && files.img2 && engine && engine.ready);
}

// ======================================================================
// Motores de execução: servidor FastAPI local ou Pyodide no navegador
// ======================================================================
async function serverEngine() {
  const r = await fetch("api/config", { headers: { Accept: "application/json" } });
  if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) throw new Error("sem API");
  const cfg = await r.json();
  return {
    name: "servidor Python local",
    ready: true,
    config: cfg,
    async stitch(f1, f2, options) {
      const body = new FormData();
      body.append("img1", f1);
      body.append("img2", f2);
      for (const [k, v] of Object.entries(options)) body.append(k, v);
      const resp = await fetch("api/stitch", { method: "POST", body });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : "falha no processamento");
      return data;
    },
  };
}

function pyodideEngine() {
  const worker = new Worker("pyodide-worker.js");
  let seq = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") { setStatus(data.message, { busy: true }); return; }
    const p = pending.get(data.id);
    if (!p) return;
    pending.delete(data.id);
    data.ok ? p.resolve(data.result) : p.reject(new Error(data.error));
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || "erro no worker"));
    pending.clear();
  };
  const call = (msg, transfer = []) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer);
  });
  const eng = {
    name: "navegador (Pyodide)",
    ready: false,
    config: null,
    async init() { eng.config = await call({ type: "init" }); eng.ready = true; return eng; },
    async stitch(f1, f2, options) {
      const [b1, b2] = await Promise.all([f1.arrayBuffer(), f2.arrayBuffer()]);
      const res = await call({ type: "stitch", img1: b1, img2: b2, options }, [b1, b2]);
      for (const [k, bytes] of Object.entries(res.images)) {
        const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
        objectUrls.push(url);
        res.images[k] = url;
      }
      return res;
    },
  };
  return eng;
}

async function startEngine() {
  const forced = new URLSearchParams(location.search).get("engine");
  // No GitHub Pages não há API: vai direto ao Pyodide, sem uma requisição 404.
  if (forced !== "pyodide" && !location.hostname.endsWith("github.io")) {
    try {
      engine = await serverEngine();
    } catch { /* sem servidor: GitHub Pages ou arquivo estático */ }
  }
  if (!engine) {
    engine = pyodideEngine();
    $("#engine").textContent = "carregando Python…";
    await engine.init();
  }
  $("#engine").textContent = "executando em: " + engine.name;
  applyConfig(engine.config);
  setStatus(files.img1 && files.img2 ? "Pronto para gerar o mosaico." : "Pronto. Escolha um exemplo ou envie duas imagens.");
  updateRunState();
}

// ======================================================================
// Entrada de imagens
// ======================================================================
function setFile(slot, file, { keepTruth = false } = {}) {
  files[slot] = file;
  if (!keepTruth) groundTruth = null;
  const drop = $(`.drop[data-slot="${slot}"]`);
  const img = $("img", drop);
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = URL.createObjectURL(file);
  img.src = img.dataset.url;
  img.hidden = false;
  $(".drop-hint", drop).hidden = true;
  drop.classList.add("filled");
  updateRunState();
}

$$(".drop").forEach((drop) => {
  const slot = drop.dataset.slot;
  const input = $("input", drop);
  input.addEventListener("change", () => {
    if (input.files[0]) { selectSample(null); setFile(slot, input.files[0]); }
  });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) { selectSample(null); setFile(slot, f); }
  });
});

async function fetchFile(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`falha ao baixar ${url}`);
  const blob = await r.blob();
  return new File([blob], url.split("/").pop(), { type: blob.type });
}

function selectSample(sample) {
  $$(".card", $("#gallery")).forEach((c) => c.setAttribute("aria-pressed", String(sample && c.dataset.id === sample.id)));
  const note = $("#sample-note");
  note.hidden = !sample;
  if (sample) note.innerHTML = `<b>${sample.title}.</b> ${sample.note}`;
}

async function loadGallery() {
  const samples = await (await fetch("samples/samples.json")).json();
  $("#gallery").innerHTML = samples.map((s) => `
    <button type="button" class="card" role="listitem" data-id="${s.id}" aria-pressed="false">
      <span class="thumbs"><img src="${s.img1}" alt="" loading="lazy"><img src="${s.img2}" alt="" loading="lazy"></span>
      <span class="card-title">${s.title}</span>
      <span class="tag ${s.kind}">${s.tag}</span>
    </button>`).join("");
  $$(".card", $("#gallery")).forEach((card) => {
    card.addEventListener("click", async () => {
      const s = samples.find((x) => x.id === card.dataset.id);
      try {
        const [f1, f2] = await Promise.all([fetchFile(s.img1), fetchFile(s.img2)]);
        setFile("img1", f1);
        setFile("img2", f2);
        selectSample(s);
      } catch (err) {
        setStatus(err.message, { error: true });
      }
    });
  });
}

// ----- Par sintético: recortes com deslocamento conhecido -----
function canvasToFile(canvas, name) {
  return new Promise((resolve) => canvas.toBlob((b) => resolve(new File([b], name, { type: "image/png" })), "image/png"));
}

function cropWithNoise(bitmap, x, y, w, h, sigma) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(bitmap, x, y, w, h, 0, 0, w, h);
  if (sigma > 0) {
    const im = ctx.getImageData(0, 0, w, h);
    const d = im.data;
    for (let i = 0; i < d.length; i += 4) {
      for (let k = 0; k < 3; k++) {
        // Ruído gaussiano (Box–Muller).
        const n = Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
        d[i + k] = Math.max(0, Math.min(255, d[i + k] + sigma * n));
      }
    }
    ctx.putImageData(im, 0, 0);
  }
  return c;
}

$("#btn-synth").addEventListener("click", async () => {
  if (!files.img1) { setStatus("Escolha primeiro uma Imagem 1 (um exemplo ou um arquivo seu).", { error: true }); return; }
  const tx = Math.round(Number($("#synth-tx").value) || 0);
  const ty = Math.round(Number($("#synth-ty").value) || 0);
  const sigma = Math.max(0, Math.min(40, Number($("#synth-noise").value) || 0));
  const bmp = await createImageBitmap(files.img1);
  const w = bmp.width - Math.abs(tx);
  const h = bmp.height - Math.abs(ty);
  if (w < 0.3 * bmp.width || h < 0.3 * bmp.height || w < 64 || h < 64) {
    setStatus(`Deslocamento grande demais para uma imagem de ${bmp.width}×${bmp.height}.`, { error: true });
    return;
  }
  // Um ponto em x2 no recorte 2 está em x1 = x2 + t no recorte 1  ⇒  t = origem2 − origem1.
  const ox1 = Math.max(0, -tx), oy1 = Math.max(0, -ty);
  const c1 = cropWithNoise(bmp, ox1, oy1, w, h, sigma);
  const c2 = cropWithNoise(bmp, ox1 + tx, oy1 + ty, w, h, sigma);
  const [f1, f2] = await Promise.all([canvasToFile(c1, "sintetica_1.png"), canvasToFile(c2, "sintetica_2.png")]);
  setFile("img1", f1);
  setFile("img2", f2);
  groundTruth = { tx, ty };
  selectSample(null);
  setStatus(`Par sintético gerado com t = (${tx}, ${ty}) px${sigma ? `, ruído σ = ${sigma}` : ""}.`);
});

// ======================================================================
// Parâmetros
// ======================================================================
function applyConfig(cfg) {
  const fill = (name, items) => {
    form[name].innerHTML = items.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    form[name].disabled = false;
  };
  fill("detector", cfg.detectors);
  fill("blend", cfg.blend_modes);
  const d = cfg.defaults;
  form.detector.value = d.detector;
  form.blend.value = d.blend;
  form.ratio.value = d.ratio;
  form.ransac_threshold.value = d.ransac_threshold;
  form.mutual.checked = d.mutual;
  syncOutputs();
}

function syncOutputs() {
  $$("output[data-for]").forEach((o) => { o.textContent = form[o.dataset.for].value; });
}
form.addEventListener("input", syncOutputs);
syncOutputs();

// ======================================================================
// Execução e resultados
// ======================================================================
const fmt = (x, n = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: n, maximumFractionDigits: n });

function metric(k, v, s = "", cls = "") {
  return `<div class="metric ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
}

function render(res) {
  const [tx, ty] = res.t;
  const scaled = res.scale < 1;
  const cards = [
    metric("Translação t = (tx, ty)", `(${fmt(tx)}, ${fmt(ty)})`,
      scaled ? `≈ (${fmt(res.t_original[0], 1)}, ${fmt(res.t_original[1], 1)}) px na escala original` : "pixels", "hl"),
    metric("Inliers", res.n_inliers, `${fmt(100 * res.inlier_ratio, 1)}% de ${res.n_matches} correspondências`, "hl"),
  ];
  if (groundTruth) {
    const [ox, oy] = res.t_original;
    const err = Math.hypot(ox - groundTruth.tx, oy - groundTruth.ty);
    cards.push(metric("Erro vs. gabarito", `${fmt(err, 2)} px`,
      `gabarito t = (${groundTruth.tx}, ${groundTruth.ty})`, err <= 1 ? "good" : "badc"));
  }
  cards.push(
    metric("Pontos detectados", `${res.n_keypoints[0]} / ${res.n_keypoints[1]}`, "imagem 1 / imagem 2"),
    metric("Erro RMS", `${fmt(res.rmse, 3)} px`, "resíduo dos inliers"),
    metric("Iterações RANSAC", res.ransac_iterations, `ε = ${res.params.ransac_threshold} px`),
    metric("Tempo", `${fmt(res.timings_ms.total / 1000, 2)} s`, `detecção ${fmt(res.timings_ms.deteccao / 1000, 2)} s`),
  );
  $("#metrics").innerHTML = cards.join("");

  const im = res.images;
  for (const k of ["mosaico", "justaposicao", "correspondencias", "contornos", "pontos1", "pontos2"]) {
    $(`#img-${k}`).src = im[k];
  }
  $("#btn-download").href = im.mosaico;
  $("#resultados").hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!files.img1 || !files.img2 || !engine?.ready) return;
  const options = {
    detector: form.detector.value,
    blend: form.blend.value,
    ratio: Number(form.ratio.value),
    ransac_threshold: Number(form.ransac_threshold.value),
    mutual: form.mutual.checked,
  };
  runBtn.disabled = true;
  setStatus("Processando: detecção, correspondência, RANSAC e composição…", { busy: true });
  try {
    while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop());
    const data = await engine.stitch(files.img1, files.img2, options);
    render(data);
    setStatus(`Concluído: t = (${fmt(data.t[0])}, ${fmt(data.t[1])}), ${data.n_inliers} inliers.`);
    $("#resultados").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus("Erro: " + err.message, { error: true });
  } finally {
    updateRunState();
  }
});

// ----- Abas -----
$$(".tabs [role=tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tabs [role=tab]").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $$("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== tab.dataset.tab; });
  });
});

loadGallery().catch(() => setStatus("Não foi possível carregar os exemplos.", { error: true }));
startEngine().catch((err) => {
  $("#engine").textContent = "motor indisponível";
  setStatus("Falha ao iniciar o Python: " + err.message, { error: true });
});
