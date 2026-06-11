// STEM2Crystal-Bench leaderboard — reads benchmark data live from GitHub.
const RAW = "https://raw.githubusercontent.com/PEESEgroup/STEM2Crystal-Bench/main/leaderboard.json";
const COLORS = {sccd:"#5b8cff", diffcsp:"#3ad29f", microscopygpt:"#ffd166", mattergen:"#ff7a90", automat:"#9aa7c4"};
const $ = (s, r=document) => r.querySelector(s);

let DATA, state = {bench:"synthetic", split:"low"};

async function load(){
  // live from GitHub (cache-busted); fall back to the co-located copy
  for (const url of [RAW + "?t=" + Date.now(), "leaderboard.json"]) {
    try { const r = await fetch(url, {cache:"no-store"}); if (r.ok) return r.json(); } catch(e){}
  }
  throw new Error("could not load leaderboard.json");
}

function fmt(v){ return v === null || v === undefined ? "—" : (Math.abs(v) >= 100 ? v.toFixed(2) : v.toFixed(4)); }
function escapeHtml(s){ return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function renderCitation(){
  const c = DATA.citation; if(!c) return;
  const el = document.getElementById("cite-block"); if(!el) return;
  el.innerHTML = `<p><b>${escapeHtml(c.title)}</b><br>${escapeHtml(c.authors)} · ${escapeHtml(c.venue)}</p>
    <pre class="code"><code>${escapeHtml(c.bibtex)}</code></pre>`;
}

function mlabel(key){ return (DATA.methods[key] && DATA.methods[key].name) || key; }

function bestPerColumn(rows, cols){
  const best = {};
  for (const c of cols){
    const hb = DATA.metrics[c].higher_better;
    let b = null;
    for (const r of rows){ const v = r[c]; if (v===null||v===undefined) continue; b = b===null ? v : (hb ? Math.max(b,v) : Math.min(b,v)); }
    best[c] = b;
  }
  return best;
}

function renderControls(){
  const bseg = $("#bench-seg"); bseg.innerHTML = "";
  Object.entries(DATA.benchmarks).forEach(([key,b])=>{
    const btn = document.createElement("button");
    btn.textContent = b.name; btn.className = key===state.bench ? "active":"";
    btn.onclick = ()=>{ state.bench = key; state.split = Object.keys(b.splits)[0]; render(); };
    bseg.appendChild(btn);
  });
  const sseg = $("#split-seg"); sseg.innerHTML = "";
  const splits = Object.keys(DATA.benchmarks[state.bench].splits);
  sseg.style.display = splits.length > 1 ? "" : "none";
  splits.forEach(s=>{
    const btn = document.createElement("button");
    btn.textContent = s; btn.className = s===state.split ? "active":"";
    btn.onclick = ()=>{ state.split = s; render(); };
    sseg.appendChild(btn);
  });
}

function renderTable(){
  const b = DATA.benchmarks[state.bench];
  $("#bench-desc").textContent = b.description || "";
  const cols = b.columns;
  let rows = b.splits[state.split].slice();
  // sort by the primary "higher-better" column (Hit@5 if present) for a sensible ranking
  const primary = cols.find(c=>DATA.metrics[c].higher_better) || cols[0];
  rows.sort((x,y)=> (y[primary]??-1) - (x[primary]??-1));
  const best = bestPerColumn(rows, cols);

  const thead = $("#lb-table thead"), tbody = $("#lb-table tbody");
  thead.innerHTML = "<tr><th class='rank'>#</th><th>Method</th>" +
    cols.map(c=>{ const m=DATA.metrics[c]; return `<th title="${c}">${m.label} ${m.higher_better?"↑":"↓"}</th>`; }).join("") + "</tr>";
  tbody.innerHTML = rows.map((r,i)=>{
    const mk = r.method, col = COLORS[mk]||"#9aa7c4", meth = DATA.methods[mk]||{};
    const tag = meth.venue ? `<span class="tag ${meth.tag==="ours"?"ours":"baseline"}">${meth.venue}</span>` : "";
    const cells = cols.map(c=>{
      const v = r[c], isBest = (v!==null && v!==undefined && v===best[c]);
      return `<td class="${isBest?"best":""}">${fmt(v)}</td>`;
    }).join("");
    return `<tr><td class="rank">${i+1}</td>` +
      `<td><span class="mname" data-m="${mk}"><span class="dot" style="background:${col}"></span>${mlabel(mk)} ${tag}</span></td>` +
      cells + `</tr>`;
  }).join("");
  tbody.querySelectorAll(".mname").forEach(el=> el.onclick = ()=>openDrawer(el.dataset.m));
}

function renderMethodCards(){
  const wrap = $("#method-cards"); wrap.innerHTML = "";
  Object.entries(DATA.methods).forEach(([key,m])=>{
    const col = COLORS[key]||"#9aa7c4";
    const div = document.createElement("div");
    div.className = "mcard"; div.onclick = ()=>openDrawer(key);
    div.innerHTML = `<h3><span class="dot" style="background:${col}"></span>${m.name}
        <span class="tag ${m.tag==="ours"?"ours":"baseline"}">${m.venue||""}</span></h3>
      <div class="who">${m.authors||""} · ${m.conditioning||""}</div>
      <p>${m.summary||""}</p>`;
    wrap.appendChild(div);
  });
}

function openDrawer(key){
  const m = DATA.methods[key]; if(!m) return;
  const links = Object.entries(m.links||{}).map(([k,v])=>`<a href="${v}" target="_blank" rel="noopener">${k}</a>`).join(" · ");
  const highlights = (m.highlights||[]).map(h=>`<li>${h}</li>`).join("");
  $("#drawer-body").innerHTML =
    `<h2><span class="dot" style="background:${COLORS[key]||'#9aa7c4'}"></span> ${m.full_name||m.name}
        ${m.tag==="ours"?'<span class="tag ours">ours</span>':''}</h2>
     <div class="kv">
        <span>${m.authors||""}</span>
        <span>${m.venue||""}</span>
        <span>conditioning: ${m.conditioning||"—"}</span>
     </div>
     ${m.figure?`<figure class="mfig"><img src="${m.figure}" alt="${m.name} architecture" loading="lazy"/>
        <figcaption>${m.figure_note||""}</figcaption></figure>`:""}
     <h3>Method</h3>
     <p>${m.description || m.summary || ""}</p>
     ${highlights?`<h3>Highlights</h3><ul>${highlights}</ul>`:""}
     ${key==="sccd"?`<h3>Run it</h3><pre class="code"><code>python scripts/download_models.py --model sccd
python scripts/generate.py --method sccd --benchmark synthetic --noise low
python scripts/evaluate.py  --method sccd --benchmark synthetic --noise low</code></pre>
     ${DATA.citation?`<h3>Citation</h3><pre class="code"><code>${escapeHtml(DATA.citation.bibtex)}</code></pre>`:""}`:""}
     <p class="muted">${links}</p>`;
  const d = $("#drawer"); d.classList.add("open"); d.setAttribute("aria-hidden","false");
}
function closeDrawer(){ const d=$("#drawer"); d.classList.remove("open"); d.setAttribute("aria-hidden","true"); }

function render(){ renderControls(); renderTable(); }

function wireStatic(){
  const L = DATA.links||{};
  $("#lnk-paper").href = L.paper||"#"; $("#lnk-data").href = L.dataset||"#"; $("#lnk-code").href = L.code||"#";
  $("#lnk-add").href = (L.code||"#") + "/blob/main/docs/add_a_method.md";
  $("#updated").textContent = DATA.updated || "—";
  if(DATA.figure_credit){ const fc=document.getElementById("fig-credit"); if(fc) fc.textContent=DATA.figure_credit; }
  $("#foot-links").innerHTML =
    `<a href="${L.dataset||'#'}">dataset</a> · <a href="${L.code||'#'}">code</a> · <a href="${L.paper||'#'}">paper</a>`;
  $("#drawer-close").onclick = closeDrawer;
  $("#drawer").addEventListener("click", e=>{ if(e.target.id==="drawer") closeDrawer(); });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") closeDrawer(); });
}

(async ()=>{
  try{
    DATA = await load();
    wireStatic(); renderMethodCards(); renderCitation(); render();
  }catch(e){
    $("#lb-table tbody").innerHTML = `<tr><td colspan="9" class="muted">Failed to load leaderboard data.</td></tr>`;
    console.error(e);
  }
})();
