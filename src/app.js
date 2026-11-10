"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const S = { api: localStorage.dmApi || (location.protocol === "file:" ? "http://localhost:8000" : location.origin), items: [], sel: -1, cols: [], tab: "fields" };

// ---------- helpers ----------
const hue = l => { let h = 0; for (const c of l) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
const color = l => `hsl(${hue(l)} 70% 55%)`;
const toast = m => { const t = $("#toast"); t.textContent = m; t.classList.add("on"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("on"), 2000); };
const kb = n => n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";
const download = (name, text, type) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 500); };
const TYPE_LABEL = { invoice: "Invoice", receipt: "Receipt", cheque: "Cheque" }, TYPE_CSV = { invoice: "invoices.csv", receipt: "receipts.csv", cheque: "cheques.csv" };
const typeOf = r => (r && r.doc_type) || "invoice";
const badge = t => `<span class="tbadge ${esc(t)}">${esc(TYPE_LABEL[t] || t)}</span>`;
const csv = (rows, cols) => {          // cols = the columns of the document type of these rows (result.columns); never mix types in one call
  const base = cols && cols.length ? cols : (S.cols.length ? ["file", ...S.cols.filter(c => c !== "file")] : ["file"]);
  const keys = [...new Set([...base, ...rows.flatMap(r => Object.keys(r))])];
  const q = v => { v = v == null ? "" : String(v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  return "\ufeff" + [keys.join(","), ...rows.map(r => keys.map(k => q(r[k])).join(","))].join("\r\n");
};

const csv2 = (keys, rows) => { const q = v => { v = v == null ? "" : String(v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  return "\ufeff" + [keys.join(","), ...rows.map(r => keys.map(k => q(r[k])).join(","))].join("\r\n"); };

// ---------- backend status ----------
async function ping() {
  const p = $("#health"), t = $("span", p);
  try {
    const h = await (await fetch(S.api + "/health")).json();
    const m = h.models || { invoice: h.model_present }, ready = Object.keys(m).filter(k => m[k]);
    p.className = "pill" + (ready.length ? "" : " warn"); t.textContent = ready.length ? "API online · " + ready.join(" + ") + " ready" : "API online · model missing";
    if (h.model_present && !S.cols.length && (!h.models || h.models.invoice)) { try { S.cols = (await (await fetch(S.api + "/model")).json()).columns || []; } catch { } }
  } catch { p.className = "pill off"; t.textContent = "API offline"; }
}

// ---------- upload & queue ----------
function add(files) {
  const ok = [...files].filter(f => /^image\//.test(f.type) || /\.pdf$/i.test(f.name));
  if (!ok.length) return toast("Choose images or PDF files");
  const start = S.items.length;
  ok.forEach(f => S.items.push({ file: f, url: f.type.startsWith("image/") ? URL.createObjectURL(f) : null, status: "queued" }));
  renderQueue(); select(start); run();
}
let running = false;
async function run() {
  if (running) return; running = true;
  while (true) {
    const it = S.items.find(i => i.status === "queued"); if (!it) break;
    it.status = "running"; renderQueue(); progress(); if (S.items[S.sel] === it) renderResult();
    const t0 = performance.now();
    try {
      const fd = new FormData(); fd.append("file", it.file);
      const r = await fetch(S.api + "/extract?format=json&doc_type=" + encodeURIComponent(it.forceType || $("#dtype").value), { method: "POST", body: fd });
      if (!r.ok) { let d = r.statusText; try { d = (await r.json()).detail || d; } catch { } throw new Error(d); }
      it.result = await r.json(); it.ms = performance.now() - t0; it.status = "done";
    } catch (e) { it.status = "error"; it.error = e.message === "Failed to fetch" ? "Cannot reach the backend at " + S.api : e.message; }
    renderQueue(); progress(); if (S.items[S.sel] === it) renderResult();
  }
  running = false; ping();
}
function progress() {
  const n = S.items.length, d = S.items.filter(i => i.status === "done" || i.status === "error").length;
  $("#bar").hidden = !(n && d < n); $("#bar i").style.width = (n ? 100 * d / n : 0) + "%";
}
function renderQueue() {
  const q = $("#queue"); q.innerHTML = "";
  S.items.forEach((it, i) => {
    const li = document.createElement("li"); li.className = "item" + (i === S.sel ? " on" : "");
    const lab = { queued: "queued", running: "reading…", done: (it.result?.fields.length ?? 0) + " fields", error: "failed" }[it.status];
    const tb = it.status === "done" ? badge(typeOf(it.result)) : "";
    li.innerHTML = `<div class="thumb" style="${it.url ? `background-image:url(${it.url})` : ""}">${it.url ? "" : "PDF"}</div>
      <div class="nm"><b>${esc(it.file.name)}</b><small>${kb(it.file.size)}${tb}</small></div><span class="st ${it.status}">${lab}</span>`;
    li.onclick = () => select(i); q.append(li);
  });
  $("#count").textContent = S.items.length; $("#empty").hidden = !!S.items.length;
  $("#clear").disabled = !S.items.length; $("#csvAll").disabled = !S.items.some(i => i.status === "done");
}
function select(i) { S.sel = i; renderQueue(); renderResult(); }

// ---------- result ----------
function renderResult() {
  const it = S.items[S.sel]; $("#welcome").hidden = !!it; $("#result").hidden = !it; if (!it) return;
  const stage = $("#stage");
  if (it.status !== "done") {
    $("#warn").hidden = true; $("#stats").innerHTML = ""; $("#tab-fields").innerHTML = $("#tab-row").innerHTML = ""; $("#tab-json").textContent = "";
    stage.innerHTML = it.status === "error" ? `<div class="err" style="line-height:1.5"><b>Extraction failed</b><br>${esc(it.error)}</div>` : `<div style="line-height:1.5"><div class="spin"></div><p class="muted center">Reading ${esc(it.file.name)}…</p></div>`;
    return;
  }
  const r = it.result, F = r.fields || [], avg = F.length ? F.reduce((a, f) => a + (f.confidence || 0), 0) / F.length : 0;
  $("#stats").innerHTML = [["Fields found", F.length], ["Avg. confidence", Math.round(avg * 100) + "%"], ["Line items", (r.items || []).length], ["Time", (it.ms / 1000).toFixed(1) + " s"]]
    .map(([a, b]) => `<div class="stat"><small>${a}</small><b>${b}</b></div>`).join("");
  drawTypeBar(it);
  const wn = r.warnings || []; $("#warn").hidden = !wn.length; $("#warn").innerHTML = wn.map(x => "⚠ " + esc(x.message)).join("<br>");
  drawStage(it, F); drawTabs(it, F);
}
function drawTypeBar(it) {
  const r = it.result, t = typeOf(r), auto = r.doc_type_source === "auto", conf = r.routing ? Math.round(r.routing.confidence * 100) : null;
  const others = Object.keys(TYPE_LABEL).filter(k => k !== t);
  $("#typebar").innerHTML = `${badge(t)}<span class="muted">${auto ? "detected automatically" + (conf != null ? " · " + conf + "% sure" : "") : "chosen by you"}</span>` +
    others.map(o => `<button class="ghost retype" data-t="${o}">Read as ${TYPE_LABEL[o].toLowerCase()} instead</button>`).join("");
  $$("#typebar .retype").forEach(b => b.onclick = () => { it.forceType = b.dataset.t; it.status = "queued"; it.result = null; renderQueue(); renderResult(); run(); });
}
function drawStage(it, F) {
  const stage = $("#stage");
  if (!it.url) { stage.innerHTML = '<div class="noprev">PDF preview is not available.<br>Fields are listed on the right.</div>'; return; }
  stage.innerHTML = `<img alt="document" src="${it.url}">`; const img = $("img", stage);
  const paint = () => {
    const pg = (it.result.pages || [])[0] || {}; const W = pg.width || img.naturalWidth, H = pg.height || img.naturalHeight;
    F.forEach((f, i) => { if (f.page && f.page !== 1) return; const [x0, y0, x1, y1] = f.bbox; const b = document.createElement("div");
      b.className = "box"; b.dataset.i = i; b.style = `--c:${color(f.label)};left:${100 * x0 / W}%;top:${100 * y0 / H}%;width:${100 * (x1 - x0) / W}%;height:${100 * (y1 - y0) / H}%`;
      b.innerHTML = `<span>${esc(f.label)}</span>`; b.onmouseenter = () => hl(i, true); b.onmouseleave = () => hl(i, false); b.onclick = () => copy(f.text); stage.append(b); });
    (it.result.items || []).forEach((m, j) => { if (m.page && m.page !== 1) return; const [x0, y0, x1, y1] = m.bbox; const b = document.createElement("div");
      b.className = "box item"; b.dataset.i = "i" + j; b.style = `left:${100 * x0 / W}%;top:${100 * y0 / H}%;width:${100 * (x1 - x0) / W}%;height:${100 * (y1 - y0) / H}%`;
      b.innerHTML = `<span>item ${j + 1}</span>`; b.onmouseenter = () => hl("i" + j, true); b.onmouseleave = () => hl("i" + j, false); stage.append(b); });
  };
  img.complete ? paint() : (img.onload = paint);
}
function hl(i, on) { $$(`[data-i="${i}"]`).forEach(e => e.classList.toggle("hl", on)); $("#stage").classList.toggle("dim", on); }
const $$ = s => [...document.querySelectorAll(s)];
const copy = t => navigator.clipboard?.writeText(t).then(() => toast("Copied: " + (t.length > 40 ? t.slice(0, 40) + "…" : t)));
function drawTabs(it, F) {
  const q = $("#q").value.trim().toLowerCase();
  $("#tab-fields").innerHTML = F.map((f, i) => ({ f, i })).filter(({ f }) => !q || (f.label + " " + f.text).toLowerCase().includes(q)).map(({ f, i }) => {
    const c = Math.round((f.confidence || 0) * 100), k = c > 85 ? "var(--ok)" : c > 60 ? "var(--warn)" : "var(--bad)";
    return `<div class="f" data-i="${i}" style="--c:${color(f.label)}"><span class="chip">${esc(f.label)}</span><span class="v">${esc(f.text)}</span>
      <div class="cf"><i><b style="width:${c}%;background:${k}"></b></i>${c}% · p${f.page || 1}${f.source === "rule" ? " · read from layout" : f.source === "computed" ? " · computed: total − subtotal" : ""}</div></div>`; }).join("") || '<p class="muted center">No field matches.</p>';
  $$("#tab-fields .f").forEach(e => { const i = +e.dataset.i; e.onmouseenter = () => hl(i, true); e.onmouseleave = () => hl(i, false); e.onclick = () => copy(F[i].text); });
  $("#tab-row").innerHTML = "<table>" + Object.entries(it.result.row || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + "</table>";
  $("#tab-json").textContent = JSON.stringify(it.result, null, 2);
  drawItems(it);
}
function drawItems(it) {
  const M = it.result.items || [], chk = it.result.items_check || {}, el = $("#tab-items");
  if (!M.length) { el.innerHTML = '<p class="muted center">No line-item table was found on this document.</p>'; return; }
  const base = [["id", "#"], ["qty", "Qty"], ["description", "Description"], ["unit_price", "Unit price"], ["amount", "Amount"]];
  const decl = (chk.columns && chk.columns.length) ? chk.columns : base.map(([key, label]) => ({ key, label }));
  const cols = decl.filter(c => M.some(m => (m[c.key] || "") !== "") || ["description", "amount"].includes(c.key));
  const num = k => M.filter(m => m[k]).every(m => /^[-(]?[^a-z]*\d[\d.,\s%)]*$/i.test(String(m[k])));
  const head = cols.map(c => `<th class="${c.key === "description" ? "" : num(c.key) ? "r" : ""}">${esc(c.label)}</th>`).join("");
  const cell = (m, c) => `<td class="${c.key === "description" ? "" : num(c.key) ? "r" : ""}">${esc(m[c.key] || "")}${c.key === "amount" && m.corrected ? '<span class="fix" title="OCR damaged this amount; recomputed as qty × unit price">fixed</span>' : ""}</td>`;
  const foot = chk.footer && Object.keys(chk.footer).length ? chk.footer : null;
  const sum = chk.items_sum != null ? chk.items_sum.toFixed(2) : "";
  const verdict = chk.matches_subtotal === true ? '<span class="ok">✓ items match the subtotal</span>' : chk.matches_subtotal === false ? '<span class="bad">⚠ items differ from the subtotal: check the amounts</span>' : "";
  el.innerHTML = `<div style="overflow-x:auto"><table class="items"><thead><tr><th>#</th>${head}</tr></thead><tbody>` +
    M.map((m, j) => `<tr data-i="i${j}"><td>${j + 1}</td>${cols.map(c => cell(m, c)).join("")}</tr>`).join("") +
    `</tbody><tfoot>${foot ? `<tr><td></td>${cols.map((c, i) => `<td class="${num(c.key) ? "r" : ""}">${i === 0 && !(c.key in foot) ? "Total" : esc(foot[c.key] || "")}</td>`).join("")}</tr>`
      : `<tr><td colspan="${cols.length}">Sum of items</td><td class="r">${sum}</td></tr>`}</tfoot></table></div><p style="margin:10px 2px">${verdict}</p>`;
  [...el.querySelectorAll("tbody tr")].forEach(tr => { const k = tr.dataset.i; tr.onmouseenter = () => hl(k, true); tr.onmouseleave = () => hl(k, false); });
}

// ---------- events ----------
const drop = $("#drop");
$("#file").onchange = e => { add(e.target.files); e.target.value = ""; };
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => add(e.dataTransfer.files));
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } };
addEventListener("paste", e => { const f = [...(e.clipboardData?.files || [])]; if (f.length) add(f); });
$$(".tabs button").forEach(b => b.onclick = () => { $$(".tabs button").forEach(x => x.classList.toggle("on", x === b)); S.tab = b.dataset.tab;
  ["fields", "items", "row", "json"].forEach(t => $("#tab-" + t).hidden = t !== S.tab); $("#q").hidden = S.tab !== "fields"; });
$("#q").oninput = () => { const it = S.items[S.sel]; if (it?.result) drawTabs(it, it.result.fields); };
$("#csvOne").onclick = () => { const it = S.items[S.sel]; if (it?.result) download(it.file.name.replace(/\.[^.]+$/, "") + ".csv", csv([it.result.row], it.result.columns), "text/csv"); };
$("#csvItems").onclick = () => { const it = S.items[S.sel], M = it?.result?.items || []; if (!M.length) return toast("No line items on this document");
  const decl = it.result.items_check?.columns || (typeOf(it.result) === "receipt" ? [{ key: "qty" }, { key: "description" }, { key: "amount" }] : [{ key: "qty" }, { key: "description" }, { key: "unit_price" }, { key: "amount" }]);
  const keys = ["file", "line", ...decl.map(c => c.key).filter(k => M.some(m => (m[k] || "") !== "") || ["description", "amount"].includes(k))];
  download(it.file.name.replace(/\.[^.]+$/, "") + "_items.csv", csv2(keys, M.map((m, j) => ({ file: it.file.name, line: j + 1, ...m }))), "text/csv"); };
$("#csvAll").onclick = () => {   // one CSV per document type: invoices.csv and receipts.csv are never merged
  const groups = {}; S.items.filter(i => i.status === "done").forEach(i => (groups[typeOf(i.result)] ||= []).push(i.result));
  const types = Object.keys(groups); if (!types.length) return;
  types.forEach((t, k) => setTimeout(() => download(TYPE_CSV[t] || t + ".csv", csv(groups[t].map(r => r.row), groups[t][0].columns), "text/csv"), k * 350));
  toast(types.length > 1 ? `${types.length} files: ` + types.map(t => TYPE_CSV[t] || t + ".csv").join(" + ") : "Downloaded " + (TYPE_CSV[types[0]] || types[0] + ".csv"));
};
$("#jsonCopy").onclick = () => { const it = S.items[S.sel]; if (it?.result) copy(JSON.stringify(it.result, null, 2)); };
$("#clear").onclick = () => { S.items.forEach(i => i.url && URL.revokeObjectURL(i.url)); S.items = []; S.sel = -1; renderQueue(); renderResult(); progress(); };
const api = $("#api"); api.value = S.api;
api.onchange = () => { S.api = api.value.replace(/\/+$/, ""); localStorage.dmApi = S.api; S.cols = []; ping(); };
const root = document.documentElement; root.dataset.theme = localStorage.dmTheme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
$("#theme").onclick = () => { root.dataset.theme = localStorage.dmTheme = root.dataset.theme === "dark" ? "light" : "dark"; };
renderQueue(); ping(); setInterval(ping, 15000);
