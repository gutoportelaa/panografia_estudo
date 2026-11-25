"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const SVG = "http://www.w3.org/2000/svg";

const files = { img1: null, img2: null };
let groundTruth = null;      // {tx, ty} quando o par é sintético
let engine = null;
let samples = [];
let userTouched = false;     // o usuário já escolheu imagens? então não rodamos a demonstração sozinhos
let lastResult = null;
const objectUrls = [];

const form = $("#form");
const status = $("#status");
const runBtn = $("#btn-run");
const cmpBtn = $("#btn-cmp");

// ---------------------------------------------------------------------------
// Formatação: vírgula decimal e sinal de menos tipográfico
// ---------------------------------------------------------------------------
const fmt = (x, n = 2) =>
  Number(x).toLocaleString("pt-BR", { minimumFractionDigits: n, maximumFractionDigits: n }).replace("-", "−");
const fmtT = (t, n = 2) => `(${fmt(t[0], n)}; ${fmt(t[1], n)})`;

function setStatus(msg, { error = false, busy = false } = {}) {
  status.className = "status" + (error ? " error" : "");
  status.innerHTML = (busy ? '<span class="spinner" aria-hidden="true"></span>' : "") + msg;
}

function updateRunState() {
  const ready = Boolean(engine && engine.ready);
  runBtn.disabled = !(files.img1 && files.img2 && ready);
  cmpBtn.disabled = !(files.img1 && files.img2 && ready);
}

// ===========================================================================
// Motores: servidor FastAPI local ou Pyodide no navegador
// ===========================================================================
async function serverEngine() {
  const r = await fetch("api/config", { headers: { Accept: "application/json" } });
  if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) throw new Error("sem API");
  const cfg = await r.json();
  return {
    name: "Python local",
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
    name: "Python no navegador",
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
    try { engine = await serverEngine(); } catch { /* sem servidor */ }
  }
  if (!engine) {
    engine = pyodideEngine();
    $("#engine").textContent = "carregando Python";
    await engine.init();
  }
  $("#engine").textContent = engine.name;
  $("#engine").classList.add("ready");
  applyConfig(engine.config);
  updateRunState();
  await galleryReady;
  if (!userTouched) await runSample("atividade");
  else setStatus("Pronto");
}

// ===========================================================================
// Entrada de imagens
// ===========================================================================
function setFile(slot, file, { keepTruth = false } = {}) {
  files[slot] = file;
  if (!keepTruth) groundTruth = null;
  const drop = $(`.drop[data-slot="${slot}"]`);
  const img = $("img", drop);
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = URL.createObjectURL(file);
  img.src = img.dataset.url;
  img.hidden = false;
  img.onload = () => {
    $(`[data-fname="${slot}"]`).textContent = `${file.name} · ${img.naturalWidth} × ${img.naturalHeight} px`;
  };
  $(".drop-hint", drop).hidden = true;
  markComparisonStale();
  drop.classList.add("filled");
  updateRunState();
}

$$(".drop").forEach((drop) => {
  const slot = drop.dataset.slot;
  const input = $("input", drop);
  const take = (f) => { userTouched = true; selectSample(null); setFile(slot, f); };
  input.addEventListener("change", () => input.files[0] && take(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) take(f);
  });
});

async function fetchFile(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`falha ao baixar ${url}`);
  const blob = await r.blob();
  return new File([blob], url.split("/").pop(), { type: blob.type });
}

function selectSample(sample) {
  $$(".card", $("#gallery")).forEach((c) => c.setAttribute("aria-pressed", String(Boolean(sample) && c.dataset.id === sample.id)));
  const note = $("#sample-note");
  note.hidden = !sample;
  if (sample) note.innerHTML = `<b>${sample.title}.</b> ${sample.note}`;
}

async function loadSample(id) {
  const s = samples.find((x) => x.id === id);
  if (!s) return null;
  const [f1, f2] = await Promise.all([fetchFile(s.img1), fetchFile(s.img2)]);
  setFile("img1", f1);
  setFile("img2", f2);
  selectSample(s);
  return s;
}

async function runSample(id) {
  try {
    await loadSample(id);
    await run({ scroll: false });
  } catch (err) {
    setStatus(err.message, { error: true });
  }
}

async function loadGallery() {
  samples = await (await fetch("samples/samples.json")).json();
  $("#gallery").innerHTML = samples.map((s) => `
    <button type="button" class="card" role="listitem" data-id="${s.id}" aria-pressed="false">
      <span class="thumbs"><img src="${s.img1}" alt="" loading="lazy"><img src="${s.img2}" alt="" loading="lazy"></span>
      <span class="card-title">${s.title}</span>
      <span class="card-meta"><span class="dot ${s.kind}"></span>${s.tag}</span>
    </button>`).join("");
  $$(".card", $("#gallery")).forEach((card) => {
    card.addEventListener("click", async () => {
      userTouched = true;
      try { await loadSample(card.dataset.id); setStatus("Pronto para gerar o mosaico"); }
      catch (err) { setStatus(err.message, { error: true }); }
    });
  });
  const credits = [...new Set(samples.map((s) => s.credit).filter(Boolean))];
  if (credits.length) $("#credits").textContent = "Fontes das imagens de teste: " + credits.join("; ") + ".";
}

$$("[data-sample]").forEach((b) => b.addEventListener("click", () => {
  userTouched = true;
  runSample(b.dataset.sample);
}));


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
        const n = Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
        d[i + k] = Math.max(0, Math.min(255, d[i + k] + sigma * n));
      }
    }
    ctx.putImageData(im, 0, 0);
  }
  return c;
}

$("#btn-synth").addEventListener("click", async () => {
  userTouched = true;
  if (!files.img1) { setStatus("Escolha antes uma Imagem 1", { error: true }); return; }
  const tx = Math.round(Number($("#synth-tx").value) || 0);
  const ty = Math.round(Number($("#synth-ty").value) || 0);
  const sigma = Math.max(0, Math.min(40, Number($("#synth-noise").value) || 0));
  const bmp = await createImageBitmap(files.img1);
  const w = bmp.width - Math.abs(tx);
  const h = bmp.height - Math.abs(ty);
  if (w < 0.3 * bmp.width || h < 0.3 * bmp.height || w < 64 || h < 64) {
    setStatus(`Deslocamento grande demais para ${bmp.width}×${bmp.height} px`, { error: true });
    return;
  }
  // Um ponto em x₂ no recorte 2 está em x₁ = x₂ + t no recorte 1, logo t = origem₂ − origem₁.
  const ox1 = Math.max(0, -tx), oy1 = Math.max(0, -ty);
  const c1 = cropWithNoise(bmp, ox1, oy1, w, h, sigma);
  const c2 = cropWithNoise(bmp, ox1 + tx, oy1 + ty, w, h, sigma);
  const [f1, f2] = await Promise.all([canvasToFile(c1, "sintetica_1.png"), canvasToFile(c2, "sintetica_2.png")]);
  setFile("img1", f1);
  setFile("img2", f2);
  groundTruth = { tx, ty };
  selectSample(null);
  setStatus(`Par sintético com t = ${fmtT([tx, ty], 0)} px${sigma ? `, ruído σ = ${sigma}` : ""}`);
});

// ===========================================================================
// Parâmetros
// ===========================================================================
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
  $$("output[data-for]").forEach((o) => {
    o.textContent = fmt(form[o.dataset.for].value, 2) + (o.dataset.unit || "");
  });
}
form.addEventListener("input", syncOutputs);
syncOutputs();

// ===========================================================================
// Execução
// ===========================================================================
function scrollToResult() {
  $("#resultados").scrollIntoView({ behavior: "smooth", block: "start" });
}

function currentOptions() {
  return {
    detector: form.detector.value,
    blend: form.blend.value,
    ratio: Number(form.ratio.value),
    ransac_threshold: Number(form.ransac_threshold.value),
    mutual: form.mutual.checked,
  };
}

async function run({ scroll = true } = {}) {
  if (!files.img1 || !files.img2 || !engine?.ready) return;
  const options = currentOptions();
  runBtn.disabled = true;
  setStatus("Detectando, casando, votando e mesclando", { busy: true });
  try {
    while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop());
    const data = await engine.stitch(files.img1, files.img2, options);
    lastResult = data;
    render(data);
    markComparisonStale();
    maybeCompare();
    setStatus(`t = ${fmtT(data.t)} px, ${data.n_inliers} inliers, ${fmt(data.timings_ms.total / 1000, 1)} s`);
    if (scroll) scrollToResult();
  } catch (err) {
    setStatus("Erro: " + err.message, { error: true });
  } finally {
    updateRunState();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); userTouched = true; run(); });

// ===========================================================================
// Resultado
// ===========================================================================
function detectorName(id) {
  const d = engine?.config?.detectors.find((x) => x.id === id);
  return d ? d.label.split(" (")[0] : id;
}

function render(res) {
  const im = res.images;
  for (const k of ["mosaico", "justaposicao", "correspondencias", "pontos1", "pontos2", "escala", "pesos"]) {
    const el = $(`#img-${k}`);
    if (el && im[k]) el.src = im[k];
  }

  const facts = [];
  facts.push(`<div class="fact"><div class="k">Translação t = (tx; ty)</div><div class="v">${fmtT(res.t)}</div>
    <div class="s">${res.scale < 1 ? `≈ ${fmtT(res.t_original, 1)} px na escala original` : "pixels; a imagem 2 vai para x₁ = x₂ + t"}</div></div>`);
  facts.push(`<div class="fact"><div class="k">Correspondências válidas (inliers)</div><div class="v">${res.n_inliers} <span style="color:var(--muted);font-size:18px">de ${res.n_matches}</span></div>
    <div class="s">${fmt(100 * res.inlier_ratio, 1)}% dos pares concordam com t</div></div>`);
  if (groundTruth) {
    const err = Math.hypot(res.t_original[0] - groundTruth.tx, res.t_original[1] - groundTruth.ty);
    facts.push(`<div class="fact truth${err > 1 ? " off" : ""}"><div class="k">Erro em relação ao gabarito</div><div class="v">${fmt(err, 2)} px</div>
      <div class="s">gabarito t = ${fmtT([groundTruth.tx, groundTruth.ty], 0)}</div></div>`);
  }
  const meta = [
    ["erro RMS", `${fmt(res.rmse, 3)} px`],
    ["detector", detectorName(res.params.detector)],
    ["pontos", `${res.n_keypoints[0]} / ${res.n_keypoints[1]}`],
    ["iterações", String(res.ransac_iterations)],
    ["mosaico", `${res.mosaic_size[0]} × ${res.mosaic_size[1]}`],
    ["tempo", `${fmt(res.timings_ms.total / 1000, 2)} s`],
  ];
  facts.push(`<dl class="meta">${meta.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`);
  facts.push(`<a class="btn" href="${im.mosaico}" download="mosaico.jpg">Baixar mosaico.jpg</a>`);
  $("#facts").innerHTML = facts.join("");
  $("#resultados").hidden = false;

  renderWalk(res);
}

function renderWalk(res) {
  const d = res.didatica || {};
  const det = detectorName(res.params.detector);
  $("#here-pontos").innerHTML = `Nesta execução: <span class="num">${res.n_keypoints[0]}</span> pontos na imagem 1 e
    <span class="num">${res.n_keypoints[1]}</span> na imagem 2, com o detector ${det}.`;

  if (d.nndr) {
    const n = d.nndr;
    $("#here-nndr").innerHTML = `Nesta execução: <span class="num">${n.aceitos}</span> de <span class="num">${n.total}</span>
      pontos passaram no limiar ${fmt(n.limiar, 2)}${res.params.mutual ? `, e a checagem mútua deixou
      <span class="num">${res.n_matches}</span> pares` : ""}.`;
    drawHistogram($("#chart-nndr"), n);
  }
  if (d.votos) {
    const eps = res.params.ransac_threshold;
    const p = res.inlier_ratio;
    $("#here-ransac").innerHTML = `Nesta execução: <span class="num">${res.n_inliers}</span> dos
      <span class="num">${res.n_matches}</span> votos ficaram a menos de ${fmt(eps, 1)} px de t = ${fmtT(res.t)}.
      Com p = ${fmt(p, 2)} de inliers, o RANSAC precisou de <span class="num">${res.ransac_iterations}</span>
      ${res.ransac_iterations === 1 ? "sorteio" : "sorteios"} para ter 99,9% de confiança.`;
    drawVotes($("#chart-votes"), d.votos, res.t, eps, false);
    drawVotes($("#chart-votes-zoom"), d.votos, res.t, eps, true);
  }
}

// ===========================================================================
// Comparação entre detectores: o mesmo par, os mesmos parâmetros, quatro detectores
// ===========================================================================
const DETECTORS = [
  ["sift", "SIFT (próprio)"],
  ["harris", "Harris + MOPS (próprio)"],
  ["opencv-sift", "SIFT (OpenCV)"],
  ["opencv-orb", "ORB (OpenCV)"],
];
let cmpGen = 0;          // invalida comparações em andamento quando a entrada muda
let cmpStale = true;
let cmpVisible = false;

function setCmpStatus(msg, busy = false) {
  $("#cmp-status").innerHTML = (busy ? '<span class="spinner" aria-hidden="true"></span>' : "") + msg;
}

function markComparisonStale() {
  cmpStale = true;
  cmpGen++;
  if ($("#cmp-table tbody").children.length) setCmpStatus("As entradas mudaram desde a última comparação.");
}

function maybeCompare() {
  if (cmpStale && cmpVisible && lastResult) compare();
}

const median = (xs) => {
  const v = [...xs].sort((a, b) => a - b), m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};

async function compare() {
  if (!files.img1 || !files.img2 || !engine?.ready) return;
  const gen = ++cmpGen;
  cmpStale = false;
  const [f1, f2] = [files.img1, files.img2];
  const base = currentOptions();
  const tbody = $("#cmp-table tbody");
  tbody.innerHTML = DETECTORS.map(([id, name]) =>
    `<tr data-det="${id}"><td>${name}</td><td class="pending" colspan="7">na fila</td></tr>`).join("");
  $("#cmp-grid").innerHTML = "";
  $("#cmp-reading").hidden = true;
  const out = {};
  for (const [id, name] of DETECTORS) {
    const row = $(`tr[data-det="${id}"]`, tbody);
    row.innerHTML = `<td>${name}</td><td class="pending" colspan="7">calculando</td>`;
    setCmpStatus(`Rodando ${name}`, true);
    try { out[id] = await engine.stitch(f1, f2, { ...base, detector: id, figuras: "pontos1" }); }
    catch (err) { out[id] = { error: err.message }; }
    if (gen !== cmpGen) return;
    const r = out[id];
    row.innerHTML = r.error
      ? `<td>${name}</td><td class="off" colspan="7">${r.error}</td>`
      : `<td>${name}</td><td>${r.n_keypoints[0]} / ${r.n_keypoints[1]}</td><td>${r.n_matches}</td>
         <td>${r.n_inliers} (${fmt(100 * r.inlier_ratio, 0)}%)</td><td>${fmtT(r.t)}</td><td class="dev">…</td>
         <td>${fmt(r.rmse, 2)} px</td><td>${fmt(r.timings_ms.deteccao / 1000, 2)} s</td>`;
  }

  const ok = DETECTORS.filter(([id]) => !out[id].error);
  const eps = base.ransac_threshold;
  const tol = Math.max(2 * eps, 3);
  let reading = "";
  if (ok.length) {
    const mt = [median(ok.map(([id]) => out[id].t[0])), median(ok.map(([id]) => out[id].t[1]))];
    const dev = Object.fromEntries(ok.map(([id]) => [id, Math.hypot(out[id].t[0] - mt[0], out[id].t[1] - mt[1])]));
    for (const [id] of ok) {
      const cell = $(`tr[data-det="${id}"] .dev`, tbody);
      cell.textContent = `${fmt(dev[id], 2)} px`;
      cell.classList.toggle("off", dev[id] > tol);
    }
    const off = ok.filter(([id]) => dev[id] > tol);
    const fastest = ok.reduce((a, b) => (out[a[0]].timings_ms.deteccao <= out[b[0]].timings_ms.deteccao ? a : b));
    const most = ok.reduce((a, b) => (out[a[0]].n_inliers >= out[b[0]].n_inliers ? a : b));
    if (!off.length && ok.length > 1) {
      const maxDev = Math.max(...ok.map(([id]) => dev[id]));
      reading = `Os ${ok.length} detectores chegam ao mesmo t, a no máximo ${fmt(maxDev, 2)} px da mediana
        ${fmtT(mt)}. Neste par, trocar o detector muda a quantidade de pontos e o custo, não a resposta:
        ${most[1]} deixou mais inliers (${out[most[0]].n_inliers}) e ${fastest[1]} detectou mais rápido
        (${fmt(out[fastest[0]].timings_ms.deteccao / 1000, 2)} s).`;
    } else if (off.length) {
      reading = `${off.map(([id, n]) => `${n}, com t = ${fmtT(out[id].t)},`).join(" e ")}
        ${off.length > 1 ? "ficam" : "fica"} a mais de ${fmt(tol, 0)} px da mediana ${fmtT(mt)}. Quando detectores
        diferentes chegam a deslocamentos diferentes, pelo menos um deles votou num aglomerado errado; o número
        de inliers e o erro RMS ajudam a julgar qual.`;
    }
  }
  const failed = DETECTORS.filter(([id]) => out[id].error);
  if (failed.length) reading += ` ${failed.map(([, n]) => n).join(" e ")} não ${failed.length > 1 ? "conseguiram" : "conseguiu"} estimar t.`;
  if (reading) { $("#cmp-reading").innerHTML = reading.trim(); $("#cmp-reading").hidden = false; }

  $("#cmp-grid").innerHTML = ok.map(([id, name]) => `
    <figure class="figure"><img src="${out[id].images.pontos1}" alt="Pontos de ${name} na imagem 1">
      <figcaption>${name}: ${out[id].n_keypoints[0]} pontos na imagem 1</figcaption></figure>`).join("");
  setCmpStatus("");
}

cmpBtn.addEventListener("click", () => { cmpStale = true; compare(); });
new IntersectionObserver((entries) => {
  cmpVisible = entries.some((e) => e.isIntersecting);
  maybeCompare();
}, { rootMargin: "200px" }).observe($("#detectores"));

// ===========================================================================
// Gráficos em SVG
// ===========================================================================
function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function svgRoot(box, w, h, label) {
  box.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart", role: "img", "aria-label": label });
  box.appendChild(svg);
  return svg;
}

function text(svg, x, y, s, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, svg);
  t.textContent = s;
  return t;
}

function drawHistogram(box, n) {
  const W = 560, H = 200, L = 40, R = 8, T = 16, B = 28;
  const svg = svgRoot(box, W, H, "Histograma da razão NNDR de todos os pontos");
  const max = Math.max(...n.contagens, 1);
  const x = (v) => L + v * (W - L - R);
  const y = (c) => H - B - (c / max) * (H - T - B);
  el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
  el("line", { x1: L, x2: W - R, y1: y(max), y2: y(max), class: "grid" }, svg);
  text(svg, L - 6, y(max) + 4, String(max), { "text-anchor": "end" });
  text(svg, L - 6, H - B + 4, "0", { "text-anchor": "end" });
  n.contagens.forEach((c, i) => {
    const x0 = x(n.bordas[i]) + 1, x1 = x(n.bordas[i + 1]) - 1;
    const on = n.bordas[i + 1] <= n.limiar + 1e-9;
    el("rect", { x: x0, y: y(c), width: Math.max(0, x1 - x0), height: H - B - y(c), class: "bar" + (on ? " on" : ""), rx: 1 }, svg);
  });
  for (const v of [0, 0.5, 1]) text(svg, x(v), H - 8, fmt(v, 1), { "text-anchor": v === 1 ? "end" : "middle" });
  el("line", { x1: x(n.limiar), x2: x(n.limiar), y1: T - 4, y2: H - B, class: "thr" }, svg);
  text(svg, x(n.limiar) - 6, T + 6, `limiar ${fmt(n.limiar, 2)}`, { "text-anchor": "end", class: "thr-label" });
  text(svg, x(0.75), H - 8, "NNDR = d₁ / d₂", { "text-anchor": "middle" });
}

function drawVotes(box, votos, t, eps, zoom) {
  const W = 300, H = 300, P = 28;
  const svg = svgRoot(box, W, H, zoom ? "Votos próximos de t, com o raio ε" : "Todos os votos de translação");
  const pts = votos.t;
  let x0, x1, y0, y1;
  if (zoom) {
    const r = Math.max(3 * eps, 6);
    [x0, x1, y0, y1] = [t[0] - r, t[0] + r, t[1] - r, t[1] + r];
  } else {
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    x0 = Math.min(...xs, t[0]); x1 = Math.max(...xs, t[0]);
    y0 = Math.min(...ys, t[1]); y1 = Math.max(...ys, t[1]);
    // Mesma escala nos dois eixos, para o aglomerado não ficar distorcido.
    const span = Math.max(x1 - x0, y1 - y0, 20) * 1.08;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    [x0, x1, y0, y1] = [cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2];
  }
  const sx = (v) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P);
  const sy = (v) => P + ((v - y0) / (y1 - y0)) * (H - 2 * P);   // ty cresce para baixo, como na imagem
  el("rect", { x: P, y: P, width: W - 2 * P, height: H - 2 * P, class: "grid", fill: "none" }, svg);
  if (zoom) {
    el("circle", { cx: sx(t[0]), cy: sy(t[1]), r: (eps / (x1 - x0)) * (W - 2 * P), class: "eps" }, svg);
  }
  // Marca de registro no t estimado, por baixo dos votos.
  const g = el("g", { transform: `translate(${sx(t[0])} ${sy(t[1])})` }, svg);
  el("circle", { r: 7, class: "reg" }, g);
  el("path", { d: "M0 -13V13M-13 0H13", class: "reg" }, g);
  const order = pts.map((p, i) => i).sort((a, b) => votos.inlier[a] - votos.inlier[b]);
  for (const i of order) {
    const [vx, vy] = pts[i];
    if (vx < x0 || vx > x1 || vy < y0 || vy > y1) continue;
    el("circle", { cx: sx(vx), cy: sy(vy), r: zoom ? 2.5 : 2, class: votos.inlier[i] ? "vin" : "vout" }, svg);
  }
  if (!zoom) {
    // Na escala geral os inliers viram um único ponto: o rótulo mostra onde estão.
    const nIn = votos.inlier.reduce((a, b) => a + b, 0);
    const right = sx(t[0]) > W / 2;
    text(svg, sx(t[0]) + (right ? -18 : 18), sy(t[1]) - 18, `${nIn} inliers`,
      { "text-anchor": right ? "end" : "start", class: "vin-label" });
  }
  text(svg, P, P - 10, zoom ? `t ± ${fmt(Math.max(3 * eps, 6), 0)} px` : "todos os votos");
  text(svg, W - P, H - 8, "tx →", { "text-anchor": "end" });
  text(svg, 6, H - P, "ty ↓");
}

// ===========================================================================
const galleryReady = loadGallery().catch(() => setStatus("Não foi possível carregar os exemplos", { error: true }));
startEngine().catch((err) => {
  $("#engine").textContent = "indisponível";
  setStatus("Falha ao iniciar o Python: " + err.message, { error: true });
});
