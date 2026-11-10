// MedRAG clinical workspace - dependency-free single-page client.
"use strict";

const S = { token: localStorage.getItem("medrag_token"), user: null, patient: null, study: null, report: null };
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const headers = opts.headers || {};
  if (S.token) headers.Authorization = "Bearer " + S.token;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401 && S.token) { logout(); throw new Error("Session expired"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.blob();
}

function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 4500);
}
async function guard(fn, btn) {
  if (btn) btn.disabled = true;
  try { return await fn(); } catch (e) { toast(e.message, true); } finally { if (btn) btn.disabled = false; }
}

// ------------------------------------------------------------------ auth
async function boot() {
  if (!S.token) return showLogin();
  try {
    S.user = await api("/api/me");
    const h = await api("/api/health");
    $("#mode-badge").textContent = h.mode + " mode · " + h.vlm;
  } catch (_) { return showLogin(); }
  $("#login-view").classList.add("hidden");
  $("#hdr-right").classList.remove("hidden");
  $("#who").textContent = `${S.user.full_name} (${S.user.role})`;
  const can = (p) => S.user.permissions.includes(p);
  document.querySelectorAll(".ingest-only").forEach((e) => e.classList.toggle("hidden", !can("ingest")));
  document.querySelectorAll(".analyze-only").forEach((e) => e.classList.toggle("hidden", !can("analyze")));
  document.querySelectorAll(".signoff-only").forEach((e) => e.classList.toggle("hidden", !can("sign_off")));
  document.querySelectorAll(".admin-only").forEach((e) => e.classList.toggle("hidden", !can("audit")));
  show("workspace");
  loadSamples(); loadPatients(); loadStudies();
}
function showLogin() {
  $("#login-view").classList.remove("hidden"); $("#hdr-right").classList.add("hidden");
  ["workspace", "search", "audit"].forEach((v) => $(`#${v}-view`).classList.add("hidden"));
}
function logout() { S.token = null; localStorage.removeItem("medrag_token"); showLogin(); }
$("#logout").onclick = logout;
$("#login-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    const r = await api("/api/auth/login", { method: "POST", json: { username: f.get("username"), password: f.get("password") } });
    S.token = r.token; localStorage.setItem("medrag_token", r.token); $("#login-error").textContent = ""; boot();
  } catch (err) { $("#login-error").textContent = err.message; }
};

function show(view) {
  ["workspace", "search", "audit"].forEach((v) => $(`#${v}-view`).classList.toggle("hidden", v !== view));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  if (view === "search") loadKB();
  if (view === "audit") loadAudit();
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => show(t.dataset.view)));

// -------------------------------------------------------------- patients
async function loadPatients() {
  const ps = await api("/api/patients");
  $("#patients").innerHTML = ps.map((p) => `<li data-ref="${esc(p.patient_ref)}" class="${S.patient === p.patient_ref ? "sel" : ""}">
    <span>${esc(p.patient_ref.slice(0, 14))}… <span class="tag">${esc(p.age_band)}</span><span class="tag">${esc(p.sex)}</span></span>
    <span class="muted small">${p.studies} studies</span></li>`).join("") || `<li class="muted">No patients yet</li>`;
  document.querySelectorAll("#patients li[data-ref]").forEach((li) => (li.onclick = () => selectPatient(li.dataset.ref)));
}
function selectPatient(ref) {
  S.patient = ref; $("#np-label").textContent = "for " + ref.slice(0, 14) + "…";
  loadPatients(); loadStudies();
}
$("#patient-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  const f = new FormData(e.target);
  const p = await api("/api/patients", { method: "POST", json: { patient_id: f.get("patient_id"), age: f.get("age") ? +f.get("age") : null, sex: f.get("sex") } });
  e.target.reset(); toast("Patient registered as pseudonym " + p.patient_ref.slice(0, 18) + "… (raw ID not stored)");
  selectPatient(p.patient_ref);
}, e.submitter);

// --------------------------------------------------------------- studies
async function loadSamples() {
  const s = await api("/api/samples");
  $("#sample-pick").innerHTML = `<option value="">…or use a synthetic sample</option>` +
    Object.entries(s).flatMap(([m, fs]) => fs.map((f) => `<option value="${m}/${f}">${m}: ${f.replace("_", " ")}</option>`)).join("");
}
$("#sample-pick").onchange = (e) => { if (e.target.value) $("#study-form [name=modality]").value = e.target.value.split("/")[0]; };

async function loadStudies() {
  const qs = S.patient ? "?patient_ref=" + encodeURIComponent(S.patient) : "";
  const st = await api("/api/studies" + qs);
  $("#studies").innerHTML = st.map((s) => `<li data-id="${s.id}" class="${S.study === s.id ? "sel" : ""}">
    <span>${esc(s.id)} <span class="tag">${esc(s.modality)}</span></span>
    <span class="status ${esc(s.latest_report_status || "")}" style="font-size:10px">${esc(s.latest_report_status || "new")}</span></li>`).join("")
    || `<li class="muted">No studies</li>`;
  document.querySelectorAll("#studies li[data-id]").forEach((li) => (li.onclick = () => selectStudy(li.dataset.id)));
}
$("#study-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  if (!S.patient) throw new Error("Select or register a patient first");
  const f = new FormData(e.target);
  const fd = new FormData();
  fd.append("patient_ref", S.patient); fd.append("modality", f.get("modality")); fd.append("clinical_note", f.get("clinical_note") || "");
  const sample = $("#sample-pick").value;
  if (sample) {
    const blob = await api(`/api/samples/${sample}?seed=${Math.floor(Math.random() * 1000)}`);
    fd.append("file", blob, "sample.png");
  } else if (f.get("file") && f.get("file").size) {
    fd.append("file", f.get("file"));
  } else throw new Error("Choose an image or a synthetic sample");
  const st = await api("/api/studies", { method: "POST", body: fd });
  e.target.reset(); $("#sample-pick").value = "";
  toast(`Study ${st.id} stored after de-identification (${st.deid_summary.masked_regions} pixel regions masked)`);
  await loadStudies(); loadPatients(); selectStudy(st.id);
}, e.submitter);

async function selectStudy(id) {
  S.study = id; loadStudies();
  const st = await api("/api/studies/" + id);
  $("#study-title").textContent = `${st.id} · ${st.modality} · age ${st.age_band}`;
  const blob = await api(`/api/studies/${id}/image`);
  $("#study-img").src = URL.createObjectURL(blob);
  const d = st.deid_summary;
  $("#deid-summary").innerHTML = `De-identified: metadata removed [${esc(d.metadata_removed.join(", ") || "none")}], `
    + `${d.masked_regions} burned-in text regions masked, note redactions ${esc(JSON.stringify(d.note_phi_redactions))}.`
    + (st.clinical_note ? `<br>Note: <i>${esc(st.clinical_note)}</i>` : "");
  if (st.reports.length) loadReport(st.reports[0].id, st.reports); else { S.report = null; renderReport(null, []); }
}

$("#analyze-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  if (!S.study) throw new Error("Select a study first");
  const f = new FormData(e.target);
  const q = (f.get("question") || "").trim() || "Please review this study.";
  $("#report-body").innerHTML = `<p class="muted">Agent running: route → retrieve → inspect image → draft → verify…</p>`;
  const rep = await api(`/api/studies/${S.study}/analyze`, { method: "POST", json: { question: q, variant: f.get("variant") } });
  await loadStudies();
  const st = await api("/api/studies/" + S.study);
  loadReport(rep.id, st.reports);
}, $("#analyze-btn"));

// --------------------------------------------------------------- reports
async function loadReport(id, history) {
  const rep = await api("/api/reports/" + id);
  S.report = rep; renderReport(rep, history || []);
}
const SECTIONS = [["findings", "Image findings"], ["impression", "Impression"], ["differential", "Differential considerations"], ["recommendations", "Recommendations"]];

function renderReport(rep, history) {
  $("#report-history").innerHTML = history.length > 1 ? "Versions: " + history.map((h) =>
    `<a href="#" data-id="${h.id}" class="${rep && h.id === rep.id ? "" : "muted"}">${h.id}</a> <span class="tag">${h.status}</span>`).join(" ") : "";
  document.querySelectorAll("#report-history a").forEach((a) => (a.onclick = (ev) => { ev.preventDefault(); loadReport(a.dataset.id, history); }));
  if (!rep) {
    $("#report-id").textContent = ""; $("#report-status").textContent = ""; $("#report-status").className = "status";
    $("#report-body").innerHTML = `<p class="muted">Run an analysis to generate an evidence-grounded draft.</p>`;
    $("#report-actions").classList.add("hidden"); return;
  }
  const c = rep.content, ev = c.evidence || [], claims = (c.draft && c.draft.claims) || [];
  $("#report-id").textContent = rep.id + " · " + (c.variant || "full");
  $("#report-status").textContent = rep.status.replace("_", " ");
  $("#report-status").className = "status " + rep.status;
  const cite = (i) => { const e = ev[i - 1]; return `<span class="cite" title="${esc(e ? e.title + ": " + e.text : "")}">${i}</span>`; };
  let html = `<p class="small"><b>Question:</b> ${esc(c.question)}</p>`;
  if (c.draft && c.draft.summary) html += `<p>${esc(c.draft.summary)}</p>`;
  for (const [key, label] of SECTIONS) {
    const items = claims.filter((x) => x.section === key);
    if (!items.length) continue;
    html += `<div class="section"><h4>${label}</h4>` + items.map((x) => {
      const cls = x.source === "image" ? "img" : x.supported ? "ok" : "bad";
      return `<div class="claim ${cls}">${esc(x.text)} ${x.citations.map(cite).join("")}`
        + (x.source === "image" ? `<span class="tag">image</span>` : "")
        + (x.supported === false ? `<span class="tag" style="color:#c0392b">unsupported</span>` : "") + `</div>`;
    }).join("") + `</div>`;
  }
  const v = c.verification;
  if (v) html += `<p class="small">Self-verification: <b>${v.unsupported}</b> unsupported of ${v.supported + v.unsupported} claims · ungrounded rate ${v.hallucination_rate.toFixed(2)} · ${c.iterations} drafting iteration(s)</p>`;
  if (rep.signed_by) html += `<p class="small"><b>${rep.status === "signed_off" ? "Signed off" : "Rejected"}</b> by ${esc(rep.signed_by)} at ${esc(rep.signed_at)}${rep.comment ? " - " + esc(rep.comment) : ""}</p>`;
  html += `<details><summary>Evidence (${ev.length})</summary>` + ev.map((e, i) =>
    `<div class="ev"><b>[${i + 1}] ${esc(e.title)}</b> <span class="tag">${esc(e.chunk_id)}</span> <span class="tag">score ${e.score.toFixed(3)}</span><br>${esc(e.text)}</div>`).join("") + `</details>`;
  if ((c.similar_cases || []).length) html += `<details><summary>Similar prior cases (${c.similar_cases.length})</summary>` + c.similar_cases.map((s) =>
    `<div class="ev">${esc(s.study_id)} · ${esc(s.finding_summary)} · similarity ${s.similarity}</div>`).join("") + `</details>`;
  html += `<details><summary>Agent trace</summary><div class="trace">` + (c.trace || []).map((t) =>
    `<div>${t.step}. <b>${esc(t.node)}</b> ${esc(JSON.stringify(Object.fromEntries(Object.entries(t).filter(([k]) => !["step", "node"].includes(k)))))}</div>`).join("") + `</div></details>`;
  $("#report-body").innerHTML = html;
  const terminal = ["signed_off", "rejected"].includes(rep.status);
  $("#report-actions").classList.remove("hidden");
  $("#signoff-btn").disabled = terminal; $("#reject-btn").disabled = terminal; $("#decision-comment").disabled = terminal;
}

async function decide(kind) {
  if (!S.report) return;
  const comment = $("#decision-comment").value;
  if (kind === "signoff" && !confirm("Sign off this report? You accept clinical responsibility for its content.")) return;
  const rep = await api(`/api/reports/${S.report.id}/${kind}`, { method: "POST", json: { comment } });
  $("#decision-comment").value = "";
  toast(kind === "signoff" ? "Report signed off" : "Report rejected");
  const st = await api("/api/studies/" + S.study);
  loadStudies(); loadReport(rep.id, st.reports);
}
$("#signoff-btn").onclick = (e) => guard(() => decide("signoff"), e.target);
$("#reject-btn").onclick = (e) => guard(() => decide("reject"), e.target);
$("#pdf-btn").onclick = (e) => guard(async () => {
  const blob = await api(`/api/reports/${S.report.id}/pdf`);
  window.open(URL.createObjectURL(blob), "_blank");
}, e.target);

// ------------------------------------------------------------- assistant
$("#ask-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  const q = e.target.question.value.trim(); if (!q) return;
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", `<div class="msg q">${esc(q)}</div>`);
  e.target.reset();
  let modality = null;
  if (S.study) { const st = await api("/api/studies/" + S.study); modality = st.modality; }
  const r = await api("/api/ask", { method: "POST", json: { question: q, modality } });
  const refs = r.citations.map((i) => { const ev = r.evidence[i - 1]; return ev ? `<div class="small muted">[${i}] ${esc(ev.title)}</div>` : ""; }).join("");
  chat.insertAdjacentHTML("beforeend", `<div class="msg a">${esc(r.answer)}${refs}</div>`);
  chat.scrollTop = chat.scrollHeight;
}, e.submitter);

// -------------------------------------------------------- knowledge base
async function loadKB() {
  const k = await api("/api/kb");
  $("#kb-count").textContent = `${k.documents} documents · ${k.chunks} chunks · ${k.embedder} + ${k.reranker}`;
  $("#kb-titles").innerHTML = k.titles.map((t) => `<li>${esc(t)}</li>`).join("");
}
$("#search-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  const f = new FormData(e.target);
  const r = await api("/api/search", { method: "POST", json: { query: f.get("query"), strategy: f.get("strategy"), modality: f.get("modality") || null, k: 5 } });
  const t = Object.entries(r.timings_ms).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ");
  $("#search-results").innerHTML = `<p class="small muted">${esc(r.strategy)} · ${t}</p>` + r.hits.map((h) =>
    `<div class="hit"><b>${h.rank}. ${esc(h.title)}</b><div class="meta">${esc(h.chunk_id)} · score ${h.score.toFixed(4)} · dense rank ${h.dense_rank ?? "-"} · sparse rank ${h.sparse_rank ?? "-"}</div>${esc(h.text)}</div>`).join("");
}, e.submitter);
$("#kb-form").onsubmit = (e) => guard(async () => {
  e.preventDefault();
  const f = new FormData(e.target);
  const r = await api("/api/kb/documents", { method: "POST", json: { title: f.get("title"), source: f.get("source") || "uploaded", text: f.get("text"), modality: f.get("modality").split(",").map((s) => s.trim()).filter(Boolean) } });
  e.target.reset(); toast(`Indexed ${r.chunks} chunk(s) as ${r.doc_id}`); loadKB();
}, e.submitter);

// ------------------------------------------------------------------ audit
async function loadAudit() {
  const rows = await api("/api/audit?limit=300");
  $("#audit-rows").innerHTML = rows.map((r) => `<tr><td>${r.seq}</td><td>${esc(r.ts)}</td><td>${esc(r.actor)} <span class="tag">${esc(r.actor_kind)}</span></td>
    <td>${esc(r.action)}</td><td>${esc(r.entity)}</td><td>${esc(r.details)}</td></tr>`).join("");
}
$("#verify-chain").onclick = (e) => guard(async () => {
  const v = await api("/api/audit/verify");
  $("#chain-status").innerHTML = v.valid ? `<span style="color:#1d8a4f">✔ Chain intact - ${v.entries} entries, head ${esc(v.head.slice(0, 16))}…</span>`
    : `<span class="error">✘ Chain broken at entry ${v.broken_at}</span>`;
}, e.target);

boot();
