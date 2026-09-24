#!/usr/bin/env python
"""Static HTML gallery for a directory tree of plot images.

Scans a plots directory (as written by makePlots.py: <root>/<year>/region_<X>/<var>.png,
with optional per-process subdirectories for 2D plots) and writes a single
self-contained ``index.html`` next to the images.  No server, no external
resources: it works from CERNBox www, from a local file:// URL and from any
static host.

Features of the page:
  * thumbnails grouped by sub-directory (e.g. RunII/region_SR), lazy-loaded
  * text filter (space-separated terms, all must match name or group)
  * "by variable" view: the same histogram from every group side by side
    (SR next to SB, data next to ttbar, ...)
  * optional "summary" section pinned at the top: the handful of plots that
    tell you whether the run is healthy
  * click for full size, arrow keys to step, Esc to close, filter kept in
    the URL hash so a view can be linked

Usage:
    python src/plotting/make_gallery.py <plots_dir> [--title T] [-m plots.yml]
        [--summary var1 var2 ...] [--output index.html]

Summary plots come from ``--summary`` and/or a ``summary:`` list in the plot
metadata yaml (``-m``, the same file makePlots.py consumed).  Names are matched
against the image basename with ``.`` and ``_`` treated as equivalent, so
``canJet0.pt`` matches ``canJet0_pt.png`` and ``canJet0_pt_logy.png``.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}
LINK_EXT = {".pdf"}

logger = logging.getLogger("make_gallery")


def _norm(name: str) -> str:
    return name.replace(".", "_").lower()


def scan(root: Path, output_name: str) -> list:
    items = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == output_name:
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            kind = "img"
        elif ext in LINK_EXT:
            kind = "link"
        else:
            continue
        rel = p.relative_to(root)
        group = rel.parent.as_posix()
        items.append({
            "path": rel.as_posix(),
            "group": "" if group == "." else group,
            "name": p.stem,
            "kind": kind,
        })
    return items


def load_summary_names(metadata: str | None, extra: list) -> list:
    names = []
    if metadata:
        try:
            import yaml
            with open(metadata) as f:
                cfg = yaml.safe_load(f) or {}
            s = cfg.get("summary") or []
            if isinstance(s, str):
                s = s.split()
            names.extend(str(x) for x in s)
        except Exception as e:  # missing yaml module or file: not fatal
            logger.warning(f"Could not read summary list from {metadata}: {e}")
    names.extend(extra or [])
    return list(dict.fromkeys(names))


def mark_summary(items: list, summary_names: list) -> int:
    if not summary_names:
        return 0
    keys = [_norm(n) for n in summary_names]
    n = 0
    for it in items:
        nm = _norm(it["name"])
        rank = next((i for i, k in enumerate(keys) if nm == k or nm.startswith(k + "_")), None)
        if rank is not None:
            it["summary"] = rank
            n += 1
    return n


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f7f7f8; --fg:#1b1b1f; --muted:#6b6b76; --card:#fff; --line:#e2e2e8; --accent:#2f6fdb; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
header h1 { font-size:16px; margin:0 8px 0 0; font-weight:600; }
header .meta { color:var(--muted); }
header input[type=search] { flex:1 1 260px; min-width:180px; padding:6px 10px; border:1px solid var(--line); border-radius:6px; font-size:14px; }
header label { color:var(--muted); display:flex; align-items:center; gap:6px; white-space:nowrap; }
header .seg button { border:1px solid var(--line); background:var(--card); padding:5px 10px; cursor:pointer; }
header .seg button:first-child { border-radius:6px 0 0 6px; } header .seg button:last-child { border-radius:0 6px 6px 0; margin-left:-1px; }
header .seg button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
main { padding:12px 16px 60px; }
section.group { margin-bottom:22px; }
section.group > h2 { font-size:14px; font-weight:600; margin:0 0 8px; cursor:pointer; user-select:none; display:flex; gap:8px; align-items:baseline; }
section.group > h2 .n { color:var(--muted); font-weight:400; }
section.group > h2::before { content:"▾"; color:var(--muted); }
section.group.closed > h2::before { content:"▸"; }
section.group.closed .grid, section.group.closed .rows { display:none; }
section.summary > h2 { color:var(--accent); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(var(--w,220px),1fr)); gap:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:zoom-in; }
.card img { display:block; width:100%; height:auto; aspect-ratio:1/1; object-fit:contain; background:#fff; }
.card .cap { padding:5px 8px; font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card .cap b { color:var(--fg); font-weight:500; }
.card.link .cap { padding:14px 8px; }
.rows .row { display:flex; gap:10px; align-items:flex-start; margin-bottom:10px; overflow-x:auto; padding-bottom:4px; }
.rows .row .lbl { flex:0 0 160px; font-size:13px; padding-top:4px; word-break:break-all; }
.rows .row .card { flex:0 0 var(--w,220px); }
.empty { color:var(--muted); padding:40px; text-align:center; }
#lb { position:fixed; inset:0; background:rgba(10,10,14,.92); display:none; z-index:10; flex-direction:column; align-items:center; justify-content:center; }
#lb.on { display:flex; }
#lb img { max-width:96vw; max-height:88vh; background:#fff; }
#lb .cap { color:#ddd; margin-top:8px; font-size:13px; }
#lb .cap a { color:#9cc4ff; }
#lb .x { position:absolute; top:10px; right:16px; color:#ccc; font-size:26px; cursor:pointer; }
#lb .nav { position:absolute; top:50%; transform:translateY(-50%); color:#ccc; font-size:40px; cursor:pointer; padding:0 12px; user-select:none; }
#lb .prev { left:0; } #lb .next { right:0; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta" id="count"></span>
  <input type="search" id="q" placeholder="filter: e.g.  canJet SB   or   m4j" autocomplete="off">
  <span class="seg"><button id="vGroups" class="on">by group</button><button id="vVars">by variable</button></span>
  <label>size <input type="range" id="size" min="120" max="520" step="20" value="220"></label>
  <label><input type="checkbox" id="expandAll" checked> expand all</label>
</header>
<main id="main"></main>
<div id="lb"><span class="x" title="close (Esc)">&times;</span><span class="nav prev">&#8249;</span><img id="lbimg" alt=""><div class="cap" id="lbcap"></div><span class="nav next">&#8250;</span></div>
<script>
const ITEMS = __ITEMS__;
const $ = s => document.querySelector(s);
const main = $('#main'), q = $('#q'), count = $('#count');
let view = 'groups', current = [], lbIdx = -1;

function parseHash(){ const h = decodeURIComponent(location.hash.slice(1)); const m = h.match(/^(?:(groups|vars):)?(.*)$/); if(m){ view = m[1]||'groups'; q.value = m[2]||''; } }
function setHash(){ history.replaceState(null,'', '#'+encodeURIComponent((view==='vars'?'vars:':'')+q.value)); }

function filtered(){
  const terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
  return ITEMS.filter(it => { const hay = (it.group+'/'+it.name).toLowerCase(); return terms.every(t => hay.includes(t)); });
}
function card(it, idx, showGroup){
  const d = document.createElement('div'); d.className = 'card'+(it.kind==='link'?' link':''); d.dataset.idx = idx;
  if(it.kind==='img'){ const im = document.createElement('img'); im.loading='lazy'; im.src = it.path; im.alt = it.name; d.appendChild(im); }
  const c = document.createElement('div'); c.className='cap'; c.innerHTML = '<b>'+esc(it.name)+'</b>' + (showGroup ? '<br>'+esc(it.group||'.') : ''); c.title = it.path; d.appendChild(c);
  d.onclick = () => it.kind==='img' ? openLb(idx) : window.open(it.path,'_blank');
  return d;
}
function esc(s){ return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function groupSection(title, items, cls){
  const s = document.createElement('section'); s.className = 'group'+(cls?' '+cls:'');
  if(!$('#expandAll').checked && !cls) s.classList.add('closed');
  const h = document.createElement('h2'); h.innerHTML = esc(title)+' <span class="n">'+items.length+'</span>'; h.onclick = () => s.classList.toggle('closed'); s.appendChild(h);
  return s;
}
function render(){
  current = filtered(); main.innerHTML = '';
  count.textContent = current.length + ' / ' + ITEMS.length + ' plots';
  if(!current.length){ main.innerHTML = '<div class="empty">nothing matches</div>'; return; }
  const idxOf = new Map(current.map((it,i)=>[it,i]));
  const summ = current.filter(it => it.summary !== undefined).sort((a,b)=> a.summary-b.summary || a.group.localeCompare(b.group));
  if(summ.length){
    const s = groupSection('summary', summ, 'summary');
    const rowsEl = document.createElement('div'); rowsEl.className='rows';
    const byName = new Map(); summ.forEach(it => { const k=it.name; if(!byName.has(k)) byName.set(k,[]); byName.get(k).push(it); });
    byName.forEach((its,name) => { const r=document.createElement('div'); r.className='row'; const l=document.createElement('div'); l.className='lbl'; l.textContent=name; r.appendChild(l); its.forEach(it=>r.appendChild(card(it, idxOf.get(it), true))); rowsEl.appendChild(r); });
    s.appendChild(rowsEl); main.appendChild(s);
  }
  if(view==='groups'){
    const groups = new Map(); current.forEach(it => { if(!groups.has(it.group)) groups.set(it.group,[]); groups.get(it.group).push(it); });
    groups.forEach((its, g) => { const s = groupSection(g||'.', its); const grid=document.createElement('div'); grid.className='grid'; its.forEach(it=>grid.appendChild(card(it, idxOf.get(it)))); s.appendChild(grid); main.appendChild(s); });
  } else {
    const byName = new Map(); current.forEach(it => { if(!byName.has(it.name)) byName.set(it.name,[]); byName.get(it.name).push(it); });
    const s = groupSection('by variable', current); const rowsEl=document.createElement('div'); rowsEl.className='rows';
    byName.forEach((its,name) => { const r=document.createElement('div'); r.className='row'; const l=document.createElement('div'); l.className='lbl'; l.textContent=name; r.appendChild(l); its.forEach(it=>r.appendChild(card(it, idxOf.get(it), true))); rowsEl.appendChild(r); });
    s.appendChild(rowsEl); main.appendChild(s);
  }
  setHash();
}
function openLb(i){ lbIdx = i; const it = current[i]; $('#lbimg').src = it.path; $('#lbcap').innerHTML = esc(it.group+'/'+it.name)+' &nbsp; <a href="'+it.path+'" target="_blank">open</a>'; $('#lb').classList.add('on'); }
function stepLb(d){ if(lbIdx<0) return; let i = lbIdx; for(let k=0;k<current.length;k++){ i=(i+d+current.length)%current.length; if(current[i].kind==='img'){ openLb(i); return; } } }
function closeLb(){ $('#lb').classList.remove('on'); lbIdx=-1; }
$('#lb .x').onclick = closeLb; $('#lb .prev').onclick = e => { e.stopPropagation(); stepLb(-1); }; $('#lb .next').onclick = e => { e.stopPropagation(); stepLb(1); };
$('#lb').onclick = e => { if(e.target.id==='lb') closeLb(); };
document.addEventListener('keydown', e => { if(lbIdx<0) return; if(e.key==='Escape') closeLb(); else if(e.key==='ArrowRight') stepLb(1); else if(e.key==='ArrowLeft') stepLb(-1); });
q.oninput = render;
$('#vGroups').onclick = () => { view='groups'; $('#vGroups').classList.add('on'); $('#vVars').classList.remove('on'); render(); };
$('#vVars').onclick = () => { view='vars'; $('#vVars').classList.add('on'); $('#vGroups').classList.remove('on'); render(); };
$('#size').oninput = e => document.documentElement.style.setProperty('--w', e.target.value+'px');
$('#expandAll').onchange = render;
parseHash(); if(view==='vars'){ $('#vVars').classList.add('on'); $('#vGroups').classList.remove('on'); }
render();
</script>
</body>
</html>
"""


def write_gallery(root: Path, items: list, title: str, output: Path) -> None:
    page = (PAGE
            .replace("__TITLE__", html.escape(title))
            .replace("__ITEMS__", json.dumps(items, separators=(",", ":"))))
    output.write_text(page)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plots_dir", help="directory containing the plots (scanned recursively)")
    ap.add_argument("--title", default=None, help="page title (default: directory name)")
    ap.add_argument("-m", "--metadata", default=None, help="plot metadata yaml; its `summary:` list pins key plots at the top")
    ap.add_argument("--summary", nargs="*", default=[], help="additional summary plot names (e.g. canJet0.pt m4j)")
    ap.add_argument("-o", "--output", default="index.html", help="output file name, relative to plots_dir unless absolute")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    root = Path(args.plots_dir)
    if not root.is_dir():
        logger.error(f"{root} is not a directory")
        return 1
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    items = scan(root, output.name)
    if not items:
        logger.warning(f"no images found under {root}")
    n_summary = mark_summary(items, load_summary_names(args.metadata, args.summary))
    title = args.title or root.resolve().name
    write_gallery(root, items, title, output)
    logger.info(f"wrote {output}: {len(items)} plots, {n_summary} in summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
