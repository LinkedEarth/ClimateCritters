"""Pre-render script: generate docs/tutorials/_tutorials_table.qmd.

Walks the tutorials sidebar in _quarto.yml, extracts title and description
from each notebook's markdown header cell, and writes a Quarto include file
containing the section headers and tables.

tutorials/index.qmd includes this file via {{< include _tutorials_table.qmd >}}
so the table stays in sync with the sidebar without manual maintenance.
"""

import json
import re
import textwrap
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent
REPO_ROOT = DOCS_DIR.parent
OUT = DOCS_DIR / "tutorials" / "_tutorials_table.qmd"

# ---------------------------------------------------------------------------
# Sidebar structure — mirrors the tutorials sidebar in _quarto.yml.
# Each entry is either a notebook dict or a section dict.
# Keeping this in Python avoids a PyYAML dependency and is easy to update.
# ---------------------------------------------------------------------------

SIDEBAR = [
    {
        "section": "Getting started",
        "contents": [
            {"text": "CCModel Basics",   "href": "notebooks/base_classes/ccmodel_basics.ipynb"},
            {"text": "Forcing Basics",   "href": "notebooks/base_classes/forcing.ipynb"},
        ],
    },
    {
        "section": "Model demos",
        "contents": [
            {
                "section": "Energy Balance Models",
                "contents": [
                    {"text": "0D",            "href": "notebooks/model_demos/ebm0d.ipynb"},
                    {"text": "1D (latitude)", "href": "notebooks/model_demos/ebm1d_lat.ipynb"},
                ],
            },
            {
                "section": "Oscillators",
                "contents": [
                    {"text": "SimplePendulum & Spring", "href": "notebooks/model_demos/pendulums_and_spring.ipynb"},
                    {"text": "Double Pendulum",         "href": "notebooks/model_demos/double_pendulum.ipynb"},
                ],
            },
            {
                "section": "Climate",
                "contents": [
                    {"text": "ENSO",                  "href": "notebooks/model_demos/enso_recharge.ipynb"},
                    {"text": "Ganopolski 2024, Model 3", "href": "notebooks/model_demos/g24.ipynb"},
                    {"text": "Stommel",               "href": "notebooks/model_demos/stommel.ipynb"},
                ],
            },
            {
                "section": "Attractors",
                "contents": [
                    {"text": "Roessler",  "href": "notebooks/model_demos/roessler.ipynb"},
                    {"text": "Lorenz63",  "href": "notebooks/model_demos/lorenz63.ipynb"},
                    {"text": "Lorenz96",  "href": "notebooks/model_demos/lorenz96.ipynb"},
                ],
            },
            {
                "section": "Box Models",
                "contents": [
                    {"text": "Carbon Cycle Demo", "href": "notebooks/model_demos/box_models_carbon.ipynb"},
                ],
            },
        ],
    },
    {
        "section": "Functionality demos",
        "contents": [
            {
                "section": "Noise",
                "contents": [
                    {"text": "Noise",       "href": "notebooks/functionality_demos/noise_demo.ipynb"},
                    {"text": "Model noise", "href": "notebooks/functionality_demos/model_noise_starter.ipynb"},
                ],
            },
            {"text": "Downsampling", "href": "notebooks/functionality_demos/downsample_starter.ipynb"},
            {"text": "Solvers",      "href": "notebooks/functionality_demos/solver_demo.ipynb"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _parse_header(source: list[str]) -> dict:
    """Extract title and description from a markdown header cell."""
    text = "".join(source)
    title_m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    desc_m = re.search(r"_Description:_\s*(.+)", text)
    return {
        "title": title_m.group(1).strip() if title_m else None,
        "description": desc_m.group(1).strip() if desc_m else None,
    }


def _notebook_meta(href: str) -> dict:
    """Return title and description for a notebook, given its docs-relative href."""
    nb_path = REPO_ROOT / "notebooks" / Path(href).relative_to("notebooks")
    if not nb_path.exists():
        return {"title": href, "description": ""}
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if not nb["cells"]:
        return {"title": href, "description": ""}
    return _parse_header(nb["cells"][0]["source"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _table_row(entry: dict) -> str:
    meta = _notebook_meta(entry["href"])
    text = entry.get("text") or meta["title"] or Path(entry["href"]).stem
    desc = meta["description"] or ""
    link = f"[{text}](../{entry['href']})"
    return f"| {link} | {desc} |"


def _render_section(section: dict, level: int) -> list[str]:
    """Recursively render a section as markdown lines."""
    hashes = "#" * level
    lines = [f"{hashes} {section['section']}", ""]

    # Walk contents in order, grouping consecutive notebook entries into one table
    pending_notebooks: list[dict] = []

    def _flush_notebooks():
        if pending_notebooks:
            lines.extend(["| Notebook | What it shows |", "|---|---|"])
            lines.extend(_table_row(nb) for nb in pending_notebooks)
            lines.append("")
            pending_notebooks.clear()

    for entry in section["contents"]:
        if "href" in entry:
            pending_notebooks.append(entry)
        else:
            _flush_notebooks()
            lines += _render_section(entry, level + 1)
    _flush_notebooks()

    return lines


def generate():
    lines = [
        "<!-- AUTO-GENERATED by scripts/generate_tutorials_index.py — do not edit by hand -->",
        "",
    ]
    for entry in SIDEBAR:
        lines += _render_section(entry, level=2)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"generate_tutorials_index: wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate()
