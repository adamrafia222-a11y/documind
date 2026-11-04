"use strict";
if (!localStorage.dmTheme) document.documentElement.dataset.theme = "dark";   // futuristic dark theme by default
// Shell around the extraction workspace (app.js): sign-in, navigation, history, dataset report.
const T = { tok: localStorage.dmTok || "", user: null, mode: "login" }, H = { rows: [] };
const _fetch = window.fetch.bind(window);
window.fetch = (u, o = {}) => {
  const a = T.tok && String(u).startsWith(S.api), p = a ? _fetch(u, { ...o, headers: { ...o.headers, Authorization: "Bearer " + T.tok } }) : _fetch(u, o);
  if (a && String(u).includes("/extract")) p.then(r => r.ok && (toast("Saved to your dataset ✓"), setTimeout(loadRecent, 500))).catch(() => { });
  return p;
};
const J = async (p, o) => {
  let r; try { r = await fetch(S.api + p, o); } catch { throw Error("Cannot reach the backend at " + S.api); }
  const d = await r.json().catch(() => ({}));
  if (r.status === 401 && T.tok && !/^\/auth\/(login|register)/.test(p)) signOut();
  if (!r.ok) throw Error(typeof d.detail === "string" ? d.detail : r.status === 422 ? "Please check the form fields" : r.statusText);
  return d;
};
const nf = x => Number(x || 0).toLocaleString(), pc = x => (100 * x).toFixed(1) + "%", when = t => new Date(t * 1000).toLocaleString();

// ---------- sign in / up ----------
$$("#amode button").forEach(b => b.onclick = () => {
  T.mode = b.dataset.m; $$("#amode button").forEach(x => x.classList.toggle("on", x === b));
  $("#aname").hidden = T.mode !== "register"; $("#asub").textContent = T.mode === "login" ? "Sign in" : "Create account"; $("#aerr").textContent = "";
  $("#apass").autocomplete = T.mode === "login" ? "current-password" : "new-password";
});
$("#aform").onsubmit = async e => {
  e.preventDefault(); const b = $("#asub"); b.disabled = true; $("#aerr").textContent = "";
  try {
    const body = { email: $("#aemail").value, password: $("#apass").value }; if (T.mode === "register") body.name = $("#aname").value;
    const d = await J("/auth/" + T.mode, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    T.tok = localStorage.dmTok = d.token; T.user = d.user; $("#apass").value = ""; enter();
  } catch (x) { $("#aerr").textContent = x.message; }
  b.disabled = false;
};
function enter() {
  $("#auth").hidden = true; $("#site").hidden = true; $("#app").hidden = false;
  $("#uname").textContent = T.user.name; $("#av").textContent = T.user.name.trim()[0].toUpperCase();
  const hr = new Date().getHours(); $("#hi").textContent = (hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening") + ", " + T.user.name.split(" ")[0];
  go(location.hash.slice(1));
}
function signOut() {
  if (T.tok) _fetch(S.api + "/auth/logout", { method: "POST", headers: { Authorization: "Bearer " + T.tok } }).catch(() => { });
  T.tok = localStorage.dmTok = ""; T.user = null; H.rows = []; $("#clear").onclick(); $("#drawer").classList.remove("on");
  $("#app").hidden = true; $("#site").hidden = false; scrollTo(0, 0);
}
$("#out").onclick = signOut;

// ---------- navigation ----------
function go(v) {
  if (!["work", "dash", "hist", "data"].includes(v)) v = "work";
  $$("[data-v]").forEach(s => s.hidden = s.dataset.v !== v); $$("#nav button").forEach(b => b.classList.toggle("on", b.dataset.go === v));
  history.replaceState(null, "", "#" + v); scrollTo(0, 0);
  if (v === "work") loadRecent(); if (v === "dash") loadDash().catch(e => toast(e.message)); if (v === "hist") loadHist().catch(e => toast(e.message)); if (v === "data") loadData();
}
$$("#nav button").forEach(b => b.onclick = () => go(b.dataset.go));

// ---------- history ----------
async function loadHist() { H.rows = await J("/history"); drawHist(); }
function drawHist() {
  const q = $("#hq").value.trim().toLowerCase(), R = H.rows.filter(x => x.file.toLowerCase().includes(q));
  const avg = H.rows.length ? H.rows.reduce((a, x) => a + x.conf, 0) / H.rows.length : 0;
  $("#hkp").innerHTML = [["Documents processed", H.rows.length], ["Avg. confidence", Math.round(avg * 100) + "%"], ["Fields extracted", nf(H.rows.reduce((a, x) => a + x.n_fields, 0))], ["Line items", nf(H.rows.reduce((a, x) => a + x.n_items, 0))]]
    .map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("");
  $("#hbody").innerHTML = R.map(x => `<tr data-id="${x.id}"><td><b>${esc(x.file)}</b></td><td>${when(x.created)}</td><td class="r">${x.n_fields}</td><td class="r">${Math.round(x.conf * 100)}%</td><td class="r">${x.n_items}</td><td class="r">${esc(x.total || "–")}</td><td class="r">${(x.ms / 1000).toFixed(1)} s</td><td class="r"><button class="ghost" data-del="${x.id}">Delete</button></td></tr>`).join("")
    || '<tr><td colspan="8" class="muted center">Nothing here yet. Extract an invoice in the Workspace and it will appear here.</td></tr>';
}
$("#hq").oninput = drawHist;
$("#hbody").onclick = async e => {
  const del = e.target.closest("[data-del]"), tr = e.target.closest("tr[data-id]");
  if (del) { if (!confirm("Delete this extraction from your history?")) return; try { await J("/history/" + del.dataset.del, { method: "DELETE" }); loadHist(); } catch (x) { toast(x.message); } }
  else if (tr) openRow(tr.dataset.id);
};
async function openRow(id) {
  const d = $("#drawer"); d.classList.add("on"); d.innerHTML = '<div class="spin"></div>';
  try {
    const x = await J("/history/" + id), r = x.result, F = r.fields || [], M = r.items || [], base = x.file.replace(/\.[^.]+$/, "");
    const keys = M[0] ? Object.keys(M[0]).filter(k => !["bbox", "page", "_cy", "corrected"].includes(k)) : [];
    d.innerHTML = `<div class="dh"><div><h3>${esc(x.file)}</h3><small class="muted">${when(x.created)} · ${(x.ms / 1000).toFixed(1)} s</small></div><button class="icon" id="dx" aria-label="Close">✕</button></div>
      <div class="dact"><button class="ghost" id="dcsv">⬇ CSV</button><button class="ghost" id="djson">⬇ JSON</button></div><div id="dimg"></div><h4>Fields (${F.length})</h4>` +
      F.map(f => `<div class="f" style="--c:${color(f.label)}"><span class="chip">${esc(f.label)}</span><span class="v">${esc(f.text)}</span></div>`).join("") +
      (M.length ? `<h4>Line items (${M.length})</h4><div style="overflow-x:auto"><table class="tbl"><thead><tr>${keys.map(k => `<th>${esc(k.replace(/_/g, " "))}</th>`).join("")}</tr></thead><tbody>${M.map(m => `<tr>${keys.map(k => `<td>${esc(m[k] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : "");
    $("#dx").onclick = () => d.classList.remove("on");
    if (x.has_img) blobOf(id).then(bl => { if (bl && bl.type.startsWith("image/")) $("#dimg").innerHTML = `<img alt="scan" style="width:100%;border-radius:12px" src="${URL.createObjectURL(bl)}">`; });
    $("#dcsv").onclick = () => download(base + ".csv", csv([r.row]), "text/csv");
    $("#djson").onclick = () => download(base + ".json", JSON.stringify(r, null, 2), "application/json");
  } catch (e) { d.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
addEventListener("keydown", e => e.key === "Escape" && $("#drawer").classList.remove("on"));

// ---------- dataset & model report ----------
const report = () => J("/dataset");
async function loadData() {
  loadScans().catch(e => toast(e.message));
  let r; try { r = await report(); } catch (e) { $("#dkp").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
  const m = r.meta || {}, c = m.corpus || {}, f = r.field_level_test || {}, pf = f.per_field || {};
  const tpl = (/(\d+) templates/.exec(m.split || "") || [])[1] || "–";
  $("#dkp").innerHTML = [["Training invoices", nf(m.n_documents || c.n_documents)], ["Layout templates", tpl], ["OCR words", nf(m.n_words)], ["Macro-F1 (test)", pc(r.macro_f1_fields?.test || 0)]]
    .map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("");
  $("#dbars").innerHTML = Object.entries(pf).sort((a, b) => b[1].token_f1 - a[1].token_f1).map(([k, v]) =>
    `<div class="br" title="exact match ${pc(v.exact_match)} · ${nf(v.docs_with_field)} documents"><span>${esc(k)}</span><i><b style="width:${100 * v.token_f1}%"></b></i><span>${pc(v.token_f1)}</span></div>`).join("");
  $("#dcfg").innerHTML = [["Source", c.source], ["Split", m.split], ["Trained", (m.trained_at || "").slice(0, 10)], ["OCR", `Tesseract · ${m.ocr_lang} · psm ${m.ocr_psm}`], ["Boosting rounds", m.best_iteration],
    ["Validation macro-F1", pc(r.macro_f1_fields?.validation || 0)], ["Exact match (mean)", pc(f.exact_match_mean || 0)], ["Token F1 (mean)", pc(f.token_f1_mean || 0)]]
    .map(([a, b]) => `<tr><td class="muted">${a}</td><td class="r">${esc(b ?? "–")}</td></tr>`).join("");
  $("#dcls").innerHTML = (m.classes || []).filter(x => x !== "O").map(x => `<span>${esc(x)}</span>`).join("");
}
(async () => {   // marketing numbers on the sign-in page come from the real training report
  try { const r = await report(), m = r.meta || {}, t = (/(\d+) templates/.exec(m.split || "") || [])[1];
    fillMq((m.classes || []).filter(x => x !== "O")); if (m.n_documents && t) $("#kpis").innerHTML = `<li><b>${nf(m.n_documents)}</b>training invoices</li><li><b>${t}</b>layout templates</li><li><b>${Math.round(100 * r.macro_f1_fields.test)}%</b>field F1 (test)</li>`; animateNums($("#kpis")); } catch { }
})();

// ---------- boot ----------
(async () => { if (T.tok) { try { T.user = await J("/auth/me"); return enter(); } catch { } } $("#site").hidden = false; })();

// ---------- website behaviour, dataset library ----------
const openAuth = m => { $(`#amode button[data-m=${m}]`).click(); $("#auth").hidden = false; setTimeout(() => $(m === "register" ? "#aname" : "#aemail").focus(), 50); };
$$("[data-open]").forEach(b => b.onclick = () => openAuth(b.dataset.open));
$("#amx").onclick = () => $("#auth").hidden = true;
$("#auth").onclick = e => { if (e.target === $("#auth")) $("#auth").hidden = true; };
addEventListener("keydown", e => e.key === "Escape" && ($("#auth").hidden = true));
const io = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && (e.target.classList.add("vis"), io.unobserve(e.target))), { threshold: .15 });
$$(".reveal").forEach(el => io.observe(el));

async function blobOf(id) { try { const r = await fetch(S.api + "/scans/" + id + "/image"); return r.ok ? await r.blob() : null; } catch { return null; } }
async function loadScans() {
  H.rows = await J("/history"); (H.urls || []).forEach(URL.revokeObjectURL); H.urls = [];
  const R = H.rows, avg = R.length ? R.reduce((a, x) => a + x.conf, 0) / R.length : 0;
  $("#skp").innerHTML = [["Scans saved", R.length], ["Images stored", R.filter(x => x.has_img).length], ["Fields extracted", nf(R.reduce((a, x) => a + x.n_fields, 0))], ["Avg. confidence", Math.round(avg * 100) + "%"]]
    .map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("");
  $("#sgrid").innerHTML = R.map(x => `<div class="sc" data-id="${x.id}"><div class="im" id="im${x.id}">${x.has_img ? "" : "no image"}</div><div class="m"><b>${esc(x.file)}</b><small>${esc(x.total || "–")} · ${Math.round(x.conf * 100)}% · ${new Date(x.created * 1000).toLocaleDateString()}</small></div></div>`).join("")
    || '<p class="muted center">Your dataset is empty. Extract an invoice in the Workspace and it is saved here with its image.</p>';
  for (const x of R.filter(x => x.has_img).slice(0, 60)) blobOf(x.id).then(bl => { const el = $("#im" + x.id); if (!bl || !el) return;
    if (bl.type.startsWith("image/")) { const u = URL.createObjectURL(bl); H.urls.push(u); el.style.backgroundImage = `url(${u})`; } else el.textContent = "PDF"; });
}
$("#sgrid").onclick = e => { const c = e.target.closest(".sc"); if (c) openRow(c.dataset.id); };
$("#dexp").onclick = async () => { try { const r = await fetch(S.api + "/dataset/export"); if (!r.ok) throw Error("Export failed"); download("documind_dataset.zip", await r.blob(), "application/zip"); } catch (e) { toast(e.message); } };

// ---------- more website features: counters, ticker, cursor glow, live demo ----------
function animateNums(root) {
  root.querySelectorAll("b").forEach(b => { const m = /^(\D*)([\d,]+)(.*)$/.exec(b.textContent); if (!m) return; const end = +m[2].replace(/,/g, ""), t0 = performance.now();
    const f = t => { const k = Math.min(1, (t - t0) / 1100); b.textContent = m[1] + Math.round(end * (1 - Math.pow(1 - k, 3))).toLocaleString("en-US") + m[3]; k < 1 && requestAnimationFrame(f); }; requestAnimationFrame(f); });
}
const fillMq = L => { const s = L.map(l => `<span>${esc(l)}</span>`).join(""); $("#mq").innerHTML = `<div>${s}${s}</div>`; };
fillMq(["SELLER_NAME", "BUYER", "DATE", "DUE_DATE", "TOTAL", "TAX", "SUB_TOTAL", "GSTIN", "PO_NUMBER", "PAYMENT_DETAILS", "SELLER_ADDRESS", "DISCOUNT"]);
animateNums($("#kpis"));
addEventListener("pointermove", e => { const g = $("#gfx"); g.style.left = e.clientX + "px"; g.style.top = e.clientY + "px"; });

const dres = $("#dres"), ddrop = $("#ddrop");
dres.innerHTML = '<p class="muted center">Your extraction appears here…</p>';
async function demo(f) {
  if (!f) return;
  dres.innerHTML = `<div class="spin"></div><p class="muted center">Reading ${esc(f.name)}…</p>`; const t0 = performance.now();
  try {
    const fd = new FormData(); fd.append("file", f);
    const r = await fetch(S.api + "/extract?format=json", { method: "POST", body: fd });
    if (!r.ok) { let d = r.statusText; try { d = (await r.json()).detail || d; } catch { } throw Error(d); }
    const x = await r.json(), F = x.fields || [], M = x.items || [], avg = F.length ? F.reduce((a, q) => a + (q.confidence || 0), 0) / F.length : 0, tot = F.find(q => q.label === "TOTAL");
    dres.innerHTML = `<div class="stats mini">${[["Fields found", F.length], ["Confidence", Math.round(avg * 100) + "%"], ["Line items", M.length], ["Total", tot ? esc(tot.text) : "–"]].map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("")}</div>
      <div class="big">${F.slice(0, 16).map(q => `<span style="--c:${color(q.label)}"><i>${esc(q.label)}</i>${esc(q.text)}</span>`).join("")}</div>
      <p class="muted">${M.length ? M.length + " line item" + (M.length > 1 ? "s" : "") + " rebuilt" + (x.items_check?.matches_subtotal ? ' · <span class="ok">✓ matches the subtotal</span>' : "") + " · " : ""}${((performance.now() - t0) / 1000).toFixed(1)} s</p>
      <button class="btn sm" id="dsave">Create an account to keep this scan →</button>`;
    $("#dsave").onclick = () => openAuth("register");
  } catch (e) { dres.innerHTML = `<p class="err">${esc(e.message === "Failed to fetch" ? "Cannot reach the backend at " + S.api : e.message)}</p>`; }
}
$("#dfile").onchange = e => { demo(e.target.files[0]); e.target.value = ""; };
["dragenter", "dragover"].forEach(ev => ddrop.addEventListener(ev, e => { e.preventDefault(); ddrop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => ddrop.addEventListener(ev, e => { e.preventDefault(); ddrop.classList.remove("over"); }));
ddrop.addEventListener("drop", e => demo(e.dataTransfer.files[0]));
ddrop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#dfile").click(); } };

// ---------- dashboard ----------
async function loadDash() {
  H.rows = await J("/history"); const R = H.rows, n = R.length, avg = n ? R.reduce((a, x) => a + x.conf, 0) / n : 0, mins = n * 3;
  $("#bkp").innerHTML = [["Scans processed", nf(n)], ["Avg. confidence", Math.round(avg * 100) + "%"], ["Time saved (est. 3 min/invoice)", mins >= 60 ? Math.floor(mins / 60) + " h " + (mins % 60) + " min" : mins + " min"], ["Avg. processing", n ? (R.reduce((a, x) => a + x.ms, 0) / n / 1000).toFixed(1) + " s" : "–"]]
    .map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("");
  const days = [...Array(14)].map((_, i) => { const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - 13 + i); return d; });
  const cnt = days.map(d => R.filter(x => { const e = new Date(x.created * 1000); e.setHours(0, 0, 0, 0); return +e === +d; }).length), mx = Math.max(1, ...cnt);
  $("#bday").innerHTML = `<svg viewBox="0 0 420 150" width="100%" role="img" aria-label="Scans per day"><defs><linearGradient id="bg1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#22e3ff"/><stop offset="1" stop-color="#8b5cff"/></linearGradient></defs>` +
    cnt.map((c, i) => { const h = c ? Math.max(8, Math.round(100 * c / mx)) : 3, x = i * 30 + 4; return `<rect x="${x}" y="${120 - h}" width="22" height="${h}" rx="5" fill="url(#bg1)" opacity="${c ? 1 : .25}"><title>${days[i].toLocaleDateString()}: ${c}</title></rect><text x="${x + 11}" y="139" text-anchor="middle" font-size="9" fill="currentColor" opacity=".6">${days[i].getDate()}</text>${c ? `<text x="${x + 11}" y="${114 - h}" text-anchor="middle" font-size="10" fill="currentColor">${c}</text>` : ""}`; }).join("") + "</svg>";
  $("#bconf").innerHTML = [["90–100%", .9, 2], ["80–90%", .8, .9], ["60–80%", .6, .8], ["Below 60%", -1, .6]].map(([l, lo, hi]) => { const k = R.filter(x => x.conf >= lo && x.conf < hi || (hi === 2 && x.conf >= .9)).length;
    return `<div class="br"><span>${l}</span><i><b style="width:${n ? 100 * k / n : 0}%"></b></i><span>${k}</span></div>`; }).join("");
  $("#brec").innerHTML = R.slice(0, 6).map(x => `<tr data-id="${x.id}"><td><b>${esc(x.file)}</b></td><td>${when(x.created)}</td><td class="r">${esc(x.total || "–")}</td><td class="r">${Math.round(x.conf * 100)}%</td></tr>`).join("") || '<tr><td colspan="4" class="muted center">No scans yet. Upload an invoice in the Workspace.</td></tr>';
}
$("#brec").onclick = e => { const tr = e.target.closest("tr[data-id]"); if (tr) openRow(tr.dataset.id); };

// ---------- command palette (Ctrl/Cmd + K) ----------
const C = { i: 0, list: [] };
const cmdItems = () => {
  const q = $("#cmdq").value.trim().toLowerCase();
  const acts = [["Go to Workspace", "work"], ["Go to Dashboard", "dash"], ["Go to History", "hist"], ["Go to Dataset", "data"]].map(([t, v]) => ({ t, k: "Page", run: () => go(v) }))
    .concat([{ t: "Toggle theme", k: "Action", run: () => $("#theme").click() }, { t: "Export dataset (.zip)", k: "Action", run: () => { go("data"); $("#dexp").click(); } }, { t: "Try a sample invoice", k: "Action", run: trySample }, { t: "Upload files…", k: "Action", run: () => { go("work"); $("#file").click(); } }, { t: "Toggle background effects", k: "Action", run: () => setFx(localStorage.dmFx === "off") }, { t: "Sign out", k: "Action", run: signOut }]).concat(Object.keys(ACC).map(n => ({ t: "Accent: " + n, k: "Theme", run: () => setAccent(n) })));
  return acts.concat(H.rows.map(x => ({ t: x.file, k: "Scan · " + (x.total || "–"), run: () => openRow(x.id) }))).filter(o => !q || o.t.toLowerCase().includes(q)).slice(0, 10);
};
function drawCmd() { C.list = cmdItems(); C.i = Math.min(C.i, Math.max(0, C.list.length - 1));
  $("#cmdl").innerHTML = C.list.map((o, i) => `<li class="${i === C.i ? "on" : ""}" data-i="${i}"><span>${esc(o.t)}</span><small>${esc(o.k)}</small></li>`).join("") || '<li class="muted">No result</li>'; }
function openCmd() { if ($("#app").hidden) return; $("#cmd").hidden = false; $("#cmdq").value = ""; C.i = 0; drawCmd(); $("#cmdq").focus();
  if (!H.rows.length) J("/history").then(r => { H.rows = r; drawCmd(); }).catch(() => { }); }
const runCmd = i => { const o = C.list[i]; if (o) { $("#cmd").hidden = true; o.run(); } };
$("#cmdq").oninput = () => { C.i = 0; drawCmd(); };
$("#cmdq").onkeydown = e => { if (e.key === "ArrowDown") { C.i = Math.min(C.i + 1, C.list.length - 1); drawCmd(); e.preventDefault(); } else if (e.key === "ArrowUp") { C.i = Math.max(0, C.i - 1); drawCmd(); e.preventDefault(); } else if (e.key === "Enter") runCmd(C.i); };
$("#cmdl").onclick = e => { const li = e.target.closest("li[data-i]"); li && runCmd(+li.dataset.i); };
$("#cmd").onclick = e => { if (e.target === $("#cmd")) $("#cmd").hidden = true; };
$("#kbtn").onclick = openCmd;
addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openCmd(); } else if (e.key === "Escape") $("#cmd").hidden = true; });

// ---------- living background: particles that react to the mouse, plus aurora glow ----------
const root_ = document.documentElement, rgb = s => { s = s.trim().replace("#", ""); if (s.length === 3) s = [...s].map(x => x + x).join(""); return [0, 2, 4].map(i => parseInt(s.substr(i, 2), 16)).join(","); };
const ACC = { Aurora: ["#8b5cff", "#22e3ff"], Emerald: ["#10b981", "#a3e635"], Sunset: ["#f43f5e", "#f59e0b"], Ocean: ["#3b82f6", "#06b6d4"] };
function setAccent(n) { const a = ACC[n]; if (!a) return; root_.style.setProperty("--ac", a[0]); root_.style.setProperty("--ac2", a[1]); localStorage.dmAcc = n; fx.colors(); }
function setFx(on) { localStorage.dmFx = on ? "on" : "off"; document.body.classList.toggle("nofx", !on); if (on) fx.start(); }
const fx = (() => {
  const cv = $("#fx"), g = cv.getContext("2d"), M = { x: -999, y: -999 }; let W, H, P = [], col = ["139,92,255", "34,227,255"], run = false;
  const size = () => { const d = Math.min(devicePixelRatio || 1, 2); W = innerWidth; H = innerHeight; cv.width = W * d; cv.height = H * d; g.setTransform(d, 0, 0, d, 0, 0);
    const n = Math.min(95, Math.round(W * H / 15000)); while (P.length < n) P.push(mk()); P.length = Math.min(P.length, n); };
  const mk = () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35, r: Math.random() * 1.8 + .8, c: Math.random() < .5 ? 0 : 1 });
  const colors = () => { const s = getComputedStyle(root_); col = [rgb(s.getPropertyValue("--ac")), rgb(s.getPropertyValue("--ac2"))]; };
  function tick() {
    if (!run) return; if (localStorage.dmFx === "off") { run = false; return; }
    g.clearRect(0, 0, W, H); const k = root_.dataset.theme === "light" ? .55 : 1;
    for (const p of P) {
      const dx = p.x - M.x, dy = p.y - M.y, d = Math.hypot(dx, dy); if (d < 150) { p.vx += dx / d * .03; p.vy += dy / d * .03; }
      p.vx *= .99; p.vy *= .99; p.vx += (Math.random() - .5) * .01; p.vy += (Math.random() - .5) * .01; p.x += p.vx; p.y += p.vy;
      if (p.life !== undefined) { p.life -= .012; if (p.life <= 0) p.dead = true; }
      if (p.x < -10) p.x = W + 10; else if (p.x > W + 10) p.x = -10; if (p.y < -10) p.y = H + 10; else if (p.y > H + 10) p.y = -10;
    }
    P = P.filter(p => !p.dead);
    for (let i = 0; i < P.length; i++) { const a = P[i];
      for (let j = i + 1; j < P.length; j++) { const b = P[j], d = Math.hypot(a.x - b.x, a.y - b.y); if (d < 125) { g.strokeStyle = `rgba(${col[a.c]},${(1 - d / 125) * .32 * k})`; g.lineWidth = 1; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); } }
      g.fillStyle = `rgba(${col[a.c]},${(a.life !== undefined ? a.life : .85) * k})`; g.shadowColor = `rgba(${col[a.c]},.9)`; g.shadowBlur = 8; g.beginPath(); g.arc(a.x, a.y, a.r, 0, 7); g.fill(); g.shadowBlur = 0; }
    requestAnimationFrame(tick);
  }
  addEventListener("resize", size); addEventListener("pointermove", e => { M.x = e.clientX; M.y = e.clientY; }); addEventListener("pointerleave", () => M.x = -999);
  addEventListener("pointerdown", e => { if (localStorage.dmFx === "off") return; for (let i = 0; i < 10; i++) { const a = Math.random() * 6.28, s = 1 + Math.random() * 2; P.push({ x: e.clientX, y: e.clientY, vx: Math.cos(a) * s, vy: Math.sin(a) * s, r: 1.6, c: i % 2, life: 1 }); } });
  size(); colors();
  return { colors, start() { if (!run && !matchMedia("(prefers-reduced-motion: reduce)").matches) { run = true; colors(); requestAnimationFrame(tick); } } };
})();
if (localStorage.dmAcc) setAccent(localStorage.dmAcc);
setFx(localStorage.dmFx !== "off");
$("#theme").addEventListener("click", () => setTimeout(fx.colors, 0));

// ---------- everyday helpers: sample invoice, recent scans, drop anywhere, shortcuts ----------
function sampleInvoice() {
  const c = document.createElement("canvas"); c.width = 900; c.height = 1180; const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, 900, 1180);
  const t = (s, x, y, f = "22px Arial", a = "left", col = "#111") => { g.fillStyle = col; g.font = f; g.textAlign = a; g.fillText(s, x, y); };
  g.fillStyle = "#1f4e79"; g.fillRect(50, 40, 240, 80); t("ACME Logo", 170, 92, "bold 30px Arial", "center", "#fff"); t("TAX INVOICE", 850, 85, "bold 44px Arial", "right");
  [["Supplier Name: ACME Trading Ltd.", 50, 190], ["Supplier Address: 123 Market Street", 50, 225], ["Casablanca, Morocco", 50, 260], ["Invoice No.: INV-2026-017", 520, 190], ["Date: 01-Nov-2026", 520, 225], ["Due Date: 15-Nov-2026", 520, 260], ["Bill to:", 50, 340], ["Customer Name: Atlas Industries", 50, 375], ["Customer Address: 45 Industrial Road, Marrakech", 50, 410]].forEach(([s, x, y]) => t(s, x, y));
  const X = [50, 120, 440, 540, 700, 850], hd = ["No.", "Description", "Qty", "Unit Price", "Amount"], rows = [["1", "Consulting service", "10", "500.00", "5000.00"], ["2", "Server hosting", "5", "1000.00", "5000.00"], ["3", "Support plan", "2", "750.00", "1500.00"]];
  g.fillStyle = "#e8eef5"; g.fillRect(50, 460, 800, 48); g.strokeStyle = "#555"; g.lineWidth = 1.5;
  for (let r = 0; r <= 4; r++) { g.beginPath(); g.moveTo(50, 460 + r * 48); g.lineTo(850, 460 + r * 48); g.stroke(); }
  X.forEach(x => { g.beginPath(); g.moveTo(x, 460); g.lineTo(x, 460 + 4 * 48); g.stroke(); });
  hd.forEach((s, i) => t(s, X[i] + 12, 491, "bold 21px Arial")); rows.forEach((r, k) => r.forEach((s, i) => t(s, X[i] + 12, 539 + k * 48, "21px Arial")));
  t("Subtotal: 11500.00", 850, 700, "24px Arial", "right"); t("Tax (20%): 2300.00", 850, 740, "24px Arial", "right"); t("Total: 13800.00", 850, 790, "bold 28px Arial", "right");
  t("Total in words: Thirteen Thousand Eight Hundred Dirhams Only", 50, 860, "20px Arial"); t("Payment: bank transfer · Terms: net 14 days", 50, 960, "20px Arial");
  return new Promise(res => c.toBlob(b => res(new File([b], "sample-invoice.png", { type: "image/png" }))));
}
const trySample = async () => {
  try { go("work"); toast("Scanning the sample invoice…"); add([await sampleInvoice()]); } catch (e) { toast("Could not create the sample: " + e.message); }
};
$("#sample").onclick = trySample;
async function loadRecent() {
  try { H.rows = await J("/history"); const R = H.rows.slice(0, 5); $("#recent").innerHTML = R.map(x => `<span class="rc" data-id="${x.id}">${esc(x.file)}<small>${esc(x.total || "")}</small></span>`).join(""); } catch { }
}
$("#recent").onclick = e => { const r = e.target.closest(".rc"); if (r) openRow(r.dataset.id); };

let dragN = 0; const dov = $("#dov");
addEventListener("dragenter", e => { if ($("#app").hidden || !e.dataTransfer || ![...e.dataTransfer.types].includes("Files")) return; dragN++; dov.hidden = false; });
addEventListener("dragleave", () => { dragN = Math.max(0, dragN - 1); if (!dragN) dov.hidden = true; });
addEventListener("dragover", e => { if (!dov.hidden) e.preventDefault(); });
addEventListener("drop", e => { dragN = 0; if (dov.hidden) return; dov.hidden = true; const handled = e.defaultPrevented; e.preventDefault(); if (!handled) { go("work"); add(e.dataTransfer.files); } });


