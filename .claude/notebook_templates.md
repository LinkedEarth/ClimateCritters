# Notebook Template Specifications

---

## Signal Model Notebook

**Purpose:** document a single signal model for a reader who wants to understand what the model represents, where it comes from, and how to run it. The model is the subject; framework features are not the focus.

### Cell structure

| # | Type | Content |
|---|------|---------|
| 1 | Markdown | `# Model Name (Author Year)` + one-sentence description |
| 2 | Markdown | **Overview**: 2–4 sentences on the physical system, phenomena targeted, and why this is a useful minimal model |
| 3 | Markdown | **Literature**: full citation(s); which paper figures are reproduced below |
| 4 | Markdown | **Equations**: LaTeX-rendered ODEs; define every symbol on first use |
| 5 | Markdown | **Parameters**: table — symbol, `param_values` key, default, units/range, physical meaning |
| 6 | Markdown | **State variables**: table — name, integrated (Y/N), description; **Implementation notes**: 1–2 sentences on non-obvious choices (solver preference, hooks used, known constraints) |
| 7 | Code | Imports |
| 8 | Markdown | `## Setup` — 1–2 sentences on canonical parameters, y0, and any non-obvious initialization |
| 9 | Code | Model instantiation and integration (canonical paper parameters, y0 matching the paper) |
| 10 | Markdown | `## Reproducing [Author Year] Figure N: brief caption` — describe what the figure shows and why it matters |
| 11 | Code | Figure N |
| 12 | Markdown | `## Reproducing [Author Year] Figure M: brief caption` — optional second figure |
| 13 | Code | Figure M |
| 14 | Markdown | **Notes** — limitations, known issues, related models; omit if nothing substantive to add |

### Conventions

- All parameters set to paper-default values in the setup cell; no unexplained magic numbers anywhere
- One figure per code cell; `plt.tight_layout(); plt.show()` at the end of every figure cell
- Figure captions live in the preceding markdown cell as the section heading — not in `ax.set_title()`. `ax.set_title()` is reserved for subplot-level labels within a multi-panel figure where the section heading alone is insufficient
- Section heading naming: `## Reproducing AuthorYear Figure N: brief description` so the heading alone tells you what you're looking at
- Figures that aren't directly reproducing a paper figure but demonstrate a key result stated in the paper (e.g., sensitive dependence) belong in the signal model notebook; label them accordingly rather than forcing a "reproducing Figure N" heading that doesn't fit
- Parameter sweeps, bifurcation diagrams, and ensemble runs are **not** part of a signal model notebook — those belong in functionality notebooks
- If the model requires a specific solver (e.g., Euler for regime-switching models), state that in the Implementation notes cell, not buried in a code comment
- The Setup markdown cell (cell 8) should explain the choice of initial conditions, not just list them — this is non-obvious for most models and frequently the source of confusion

---

## Functionality Notebook

**Purpose:** demonstrate one framework capability. The model(s) are vehicles; the feature is the subject. A reader should be able to use the notebook without having read the corresponding signal model notebook.

### Cell structure

| # | Type | Content |
|---|------|---------|
| 1 | Markdown | `# Feature Name` + one-sentence description of what it enables |
| 2 | Markdown | **What this does**: 2–3 sentences on the problem it solves and when to reach for it |
| 3 | Markdown | **Models used**: which model(s) and one sentence on why (what property makes each one illustrative) |
| 4 | Code | Imports and shared setup (forcing objects, time grids, helper functions) |
| 5 | Markdown | `## [Case or concept 1]` — explain the concept and what to look for in the output |
| 6 | Code | Minimal working example for concept 1 (integration call(s)) |
| 7 | Code | Visualization for concept 1 |
| 8–N | Markdown + Code | Repeat the markdown → run → figure pattern for each additional concept or case |
| N+1 | Markdown | **When to use which**: concrete decision criteria as a table or bullets; "use X when Y", not "consider X if you care about Y" |
| N+2 | Markdown | **See also**: signal model notebooks and related functionality notebooks |

### Conventions

- Create fresh model instances per section rather than mutating a shared instance
- Each section is self-contained: its own integration call(s) and its own figure
- No more than two models as vehicles; if more are needed, split into separate notebooks
- If a model helper function or forcing constructor is reused across sections, define it once in the shared setup cell (cell 4), not inline in each section — but keep model-specific constants (n, F, t_span, y0) named and grouped per model in that cell, not scattered globally
- Guidance table must be actionable; omit any row that would say "it depends on your use case"
- When a model has structural constraints that restrict solver choice (e.g., regime-switching models), the explanation must come in a markdown cell **before** the code that runs the model — the reader needs the "why" before they see the consequence, not after
- Integration runs that are logically grouped (e.g., three dt values for a convergence study) belong in a single code cell; the figure gets the next cell. Don't split a group of related `run_X()` calls across cells
