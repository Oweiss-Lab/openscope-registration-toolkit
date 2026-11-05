"""Manual-review GUI builder: self-contained HTML for cross-session track review.

Generates a single static HTML file (no server) showing, per candidate track, the
aligned image crops from every session with the matched footprint outline and a
provenance-colored border. The reviewer marks accept / reject / unsure per track;
decisions persist in browser localStorage and export to CSV for the frozen manifest.
"""

from __future__ import annotations

import base64
import html as html_module
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

CATEGORY_COLORS = {
    "matched": "#2066a8",
    "rescued_classifier_override": "#f6a600",
    "present_but_undetected": "#5ab06c",
    "absent_ambiguous": "#c8c8c8",
}


def encode_crop_png(gray: np.ndarray, outline: np.ndarray | None = None, scale: int = 3) -> str:
    """Encode a [0,1] grayscale crop (optional yellow outline) as a base64 PNG string."""
    from PIL import Image
    from scipy import ndimage

    rgb = np.repeat(np.clip(gray, 0.0, 1.0)[:, :, None], 3, axis=2)
    if outline is not None and outline.any():
        edge = ndimage.binary_dilation(outline) ^ outline.astype(bool)
        rgb[edge] = [1.0, 0.9, 0.1]
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_manual_review_html(
    tracks: list[dict[str, Any]],
    session_labels: list[str],
    output_path: Path,
    *,
    title: str,
    context_note: str = "",
    export_slug: str = "",
) -> None:
    """Write the review GUI.

    Each track dict: ``{"track_id": int, "metrics": {name: value}, "cells":
    {session_label: {"image_b64": str, "category": str, "tooltip": str}},
    "review_required": bool (default True)}``. Tracks with
    ``review_required=False`` (triage policy) default to an automatic accept
    that the reviewer can override; exports record decision source
    (manual/auto). ``export_slug`` is appended to the exported CSV filename
    (``manual_review_decisions<_slug>.csv``).
    """
    metric_names = sorted({k for t in tracks for k in t["metrics"]})
    rows = []
    for track in tracks:
        cells = []
        for label in session_labels:
            cell = track["cells"].get(label)
            if cell is None:
                cells.append('<td class="crop missing">-</td>')
                continue
            color = CATEGORY_COLORS.get(cell["category"], "#000")
            tooltip = html_module.escape(cell.get("tooltip", cell["category"]))
            cells.append(
                f'<td class="crop"><img src="data:image/png;base64,{cell["image_b64"]}" '
                f'style="border:3px solid {color}" title="{tooltip}"></td>'
            )
        metrics = "".join(f"<td class=metric>{track['metrics'].get(m, '')}</td>" for m in metric_names)
        review_flag = 1 if track.get("review_required", True) else 0
        rows.append(
            f'<tr class="track" id="track-{track["track_id"]}" data-track="{track["track_id"]}" data-review="{review_flag}">'
            f'<td class="tid">{track["track_id"]}</td>{metrics}{"".join(cells)}'
            f'<td class="decision" id="dec-{track["track_id"]}">undecided</td></tr>'
        )
    legend = " ".join(
        f'<span class="chip" style="background:{c}">{html_module.escape(k)}</span>'
        for k, c in CATEGORY_COLORS.items()
    )
    header_cells = "".join(f"<th>{html_module.escape(m)}</th>" for m in metric_names)
    header_sessions = "".join(f'<th class="sess">{html_module.escape(s)}</th>' for s in session_labels)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_module.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 12px; background: #fafafa; }}
 h1 {{ font-size: 18px; }} .note {{ color: #444; max-width: 1100px; font-size: 13px; }}
 .chip {{ color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 4px; }}
 .toolbar {{ position: sticky; top: 0; background: #fafafa; padding: 8px 0; z-index: 5;
            border-bottom: 1px solid #ddd; margin-bottom: 8px; }}
 button {{ margin-right: 6px; padding: 4px 10px; }}
 table {{ border-collapse: collapse; }}
 th {{ font-size: 11px; padding: 2px 4px; position: sticky; top: 46px; background: #fafafa; }}
 td {{ padding: 2px; text-align: center; font-size: 12px; }}
 td.crop img {{ display: block; image-rendering: pixelated; }}
 tr.track.selected {{ outline: 3px solid #333; }}
 tr.track.accept td.decision {{ background: #c8e6c9; }} tr.track.reject td.decision {{ background: #ffcdd2; }}
 tr.track.unsure td.decision {{ background: #fff9c4; }}
 .hidden {{ display: none; }}
 #progress {{ font-weight: 600; margin-left: 12px; }}
</style></head><body>
<h1>{html_module.escape(title)}</h1>
<p class="note">{html_module.escape(context_note)}
 Keyboard: <b>j</b>/<b>k</b> next/previous track, <b>a</b> accept, <b>r</b> reject, <b>u</b> unsure.
 Yellow outline = matched footprint (warped into reference coordinates). Border colors: {legend}
 Decisions persist in this browser (localStorage) and survive reload. Export opens a save dialog starting in your
 Documents folder; after you pick your export folder once (e.g. Documents/allen_institute/exported), all review GUIs
 remember and reuse it.</p>
<div class="toolbar">
 <button onclick="setFilter('all')">All</button>
 <button onclick="setFilter('undecided')">Undecided</button>
 <button onclick="setFilter('accept')">Accepted</button>
 <button onclick="setFilter('reject')">Rejected</button>
 <button onclick="setFilter('unsure')">Unsure</button>
 <button onclick="exportCsv()"><b>Export decisions CSV</b></button>
 <input type="file" id="importFile" accept=".csv" onchange="importCsv(event)" title="Import a previously exported CSV">
 <span id="progress"></span>
</div>
<table>
<thead><tr><th>track</th>{header_cells}{header_sessions}<th>decision</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
<script>
const KEY = {json.dumps("review::" + title)};
let decisions = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let selected = null;
const tracks = Array.from(document.querySelectorAll("tr.track"));
function record(id, row) {{
  // normalize: stored string (legacy) -> manual record; absent + auto track -> implicit auto-accept
  const raw = decisions[id];
  if (typeof raw === "string") return {{d: raw, src: "manual"}};
  if (raw) return raw;
  if (row.dataset.review === "0") return {{d: "accept", src: "auto"}};
  return null;
}}
function render() {{
  let counts = {{accept: 0, reject: 0, unsure: 0}};
  let needed = 0, decided = 0, autos = 0;
  for (const row of tracks) {{
    const id = row.dataset.track;
    const rec = record(id, row);
    row.classList.remove("accept", "reject", "unsure");
    if (rec) {{ row.classList.add(rec.d); counts[rec.d] = (counts[rec.d] || 0) + 1; }}
    if (rec && rec.src === "auto") autos++;
    if (row.dataset.review === "1") {{ needed++; if (decisions[id]) decided++; }}
    document.getElementById("dec-" + id).textContent = rec ? rec.d + (rec.src === "auto" ? " (auto)" : "") : "undecided";
  }}
  document.getElementById("progress").textContent =
    `${{decided}}/${{needed}} requiring review done  (accept ${{counts.accept}} / reject ${{counts.reject}} / ` +
    `unsure ${{counts.unsure}}; ${{autos}} auto-accepted)`;
  localStorage.setItem(KEY, JSON.stringify(decisions));
}}
function select(i) {{
  if (selected !== null) tracks[selected].classList.remove("selected");
  selected = Math.max(0, Math.min(tracks.length - 1, i));
  tracks[selected].classList.add("selected");
  tracks[selected].scrollIntoView({{block: "center"}});
}}
function decide(value) {{
  if (selected === null) return;
  decisions[tracks[selected].dataset.track] = {{d: value, src: "manual"}};
  render();
  select(selected + 1);
}}
function setFilter(mode) {{
  for (const row of tracks) {{
    const rec = record(row.dataset.track, row);
    const show = mode === "all"
      || (mode === "undecided" ? (row.dataset.review === "1" && !decisions[row.dataset.track]) : rec && rec.d === mode);
    row.classList.toggle("hidden", !show);
  }}
}}
async function exportCsv() {{
  let csv = "track_id,decision,source\\n";
  for (const row of tracks) {{
    const rec = record(row.dataset.track, row);
    csv += `${{row.dataset.track}},${{rec ? rec.d : "undecided"}},${{rec ? rec.src : "none"}}\\n`;
  }}
  const name = {json.dumps("manual_review_decisions" + (("_" + export_slug.strip("_")) if export_slug else "") + ".csv")};
  if (window.showSaveFilePicker) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        id: "allen_review_exports",
        startIn: "documents",
        suggestedName: name,
        types: [{{description: "CSV", accept: {{"text/csv": [".csv"]}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(csv);
      await writable.close();
      return;
    }} catch (err) {{ if (err.name === "AbortError") return; }}
  }}
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = name;
  a.click();
}}
function importCsv(event) {{
  const reader = new FileReader();
  reader.onload = () => {{
    for (const line of reader.result.split("\\n").slice(1)) {{
      const [id, d, src] = line.trim().split(",");
      if (id && ["accept", "reject", "unsure"].includes(d) && src !== "auto") decisions[id] = {{d: d, src: "manual"}};
    }}
    render();
  }};
  reader.readAsText(event.target.files[0]);
}}
tracks.forEach((row, i) => row.addEventListener("click", () => select(i)));
document.addEventListener("keydown", (e) => {{
  if (e.key === "j") select(selected === null ? 0 : selected + 1);
  else if (e.key === "k") select(selected === null ? 0 : selected - 1);
  else if (e.key === "a") decide("accept");
  else if (e.key === "r") decide("reject");
  else if (e.key === "u") decide("unsure");
}});
render(); select(0);
</script></body></html>
"""
    output_path.write_text(page, encoding="utf-8")


def build_review_index_html(entries: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    """Write a batch index page listing review GUIs with live progress.

    Each entry: ``{"label": str, "href": str, "gui_title": str, "n_tracks": int}``.
    Progress is read from the same browser's localStorage (key ``review::<gui_title>``),
    so it reflects decisions made in GUIs opened from this index.
    """
    rows = "".join(
        f'<tr><td><a href="{html_module.escape(e["href"])}">{html_module.escape(e["label"])}</a></td>'
        f'<td>{e["n_tracks"]}</td><td class="prog" data-key={json.dumps("review::" + e["gui_title"])} '
        f'data-total="{e["n_tracks"]}">-</td></tr>'
        for e in entries
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_module.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 20px; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: 6px 14px; border-bottom: 1px solid #ddd; font-size: 14px; }}
 .done {{ background: #c8e6c9; }}
</style></head><body>
<h1>{html_module.escape(title)}</h1>
<p>Progress reflects decisions stored by this browser. Open each GUI, review, export its CSV, then return here.</p>
<table><thead><tr><th>Review GUI</th><th>Tracks</th><th>Progress (accept/reject/unsure)</th></tr></thead>
<tbody>{rows}</tbody></table>
<script>
for (const cell of document.querySelectorAll("td.prog")) {{
  const decisions = JSON.parse(localStorage.getItem(cell.dataset.key) || "{{}}");
  const counts = {{accept: 0, reject: 0, unsure: 0}};
  for (const v of Object.values(decisions)) {{
    const d = typeof v === "string" ? v : v.d;
    if (counts[d] !== undefined) counts[d]++;
  }}
  const done = counts.accept + counts.reject + counts.unsure;
  cell.textContent = `${{done}}/${{cell.dataset.total}} (${{counts.accept}}/${{counts.reject}}/${{counts.unsure}})`;
  if (done >= Number(cell.dataset.total)) cell.classList.add("done");
}}
</script></body></html>
"""
    output_path.write_text(page, encoding="utf-8")
