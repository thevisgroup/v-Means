# V-Means

Clean desktop package for the visible and explainable clustering application.

This directory is the runnable, reorganized copy of the dissertation app. The
original working files in the parent `dissertation/` folder are left untouched.
There is one public entry point:

```bash
python3 app.py
```

`app.py` starts the Qt desktop application from `vmeans.gui.main_window`.

## Overview Figure

The overview diagram used in the paper is included here:
[`figures/overview.pdf`](figures/overview.pdf).

It summarizes the v-Means workflow: load or generate a 2D dataset, construct
the visible silhouette, detect gradient boundaries, compute cluster centers,
optionally recurse into child regions, and inspect the final result through
the details/AI feedback view.

## Requirements

Tested on macOS with Python 3.12.9. The Python dependencies are listed in
`requirements.txt`:

```text
numpy
pandas
matplotlib
PyQt6
pyqtgraph
scipy
scikit-learn
openpyxl
```

`openpyxl` is needed for reading Excel files through pandas. The AI feedback
panel uses Python's standard library for HTTP calls, so OpenAI/Gemini/Claude
SDK packages are not required.

Ollama is optional and is not installed by `requirements.txt`. If you want the
local AI feedback panel, install and run Ollama separately, then make sure a
model such as `qwen2.5:14b-instruct` or `llama3.2:latest` is available.

## Install

Recommended clean setup:

```bash
cd /Users/slian/Desktop/VisKMean/dissertation/V-Means
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If you are using the existing dissertation environment, this path is already
known to work on this machine:

```bash
cd /Users/slian/Desktop/VisKMean/dissertation/V-Means
/Users/slian/miniforge3/bin/python3 app.py
```

If `python3 app.py` reports missing packages such as `numpy` or `PyQt6`, your
shell is pointing to a different Python interpreter. Activate the environment
where the requirements are installed, or use the miniforge command above.

## Optional AI Providers

The Hover Details panel can send the current clustering context to an AI model.
Local Ollama keeps the data on your machine. Cloud providers require API keys:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

For Ollama, the default endpoint is `http://localhost:11434`. Override it if
needed:

```bash
export OLLAMA_HOST=http://localhost:11434
```

## Run

From this directory:

```bash
python3 app.py
```

or, using the tested local environment:

```bash
/Users/slian/miniforge3/bin/python3 app.py
```

## Included Data

The package includes local benchmark data files so the built-in generated-data
choices work without downloading:

- `Aggregation.npz` for `aggregation`
- `Compound.npz` and `Compound.txt` for `zahn_compound`
- `hosp-epis-stat-admi-diag-2023-24-tab.xlsx` for the hospital admissions view

Generated datasets such as `blobs`, `cross`, `ring`, `moons`, and
`anisotropic_blobs` are produced in code.

## Package Layout

```text
vmeans/
  animation/
    recursive.py        StepFrame and recursive child analysis helpers
    builder.py          animation frame sequence builder
  gui/
    main_window.py      main Qt window and application entry point
    standard_tab.py     Standard Analysis tab
    step_animation_*.py Step Animation tab split into UI/build/render mixins
    hover_*.py          hover/select/AI feedback viewer
    data_preview.py     uploaded-data preview and cleaning dialog
  rendering/
    base.py             shared Matplotlib constants, options, and helpers
    dispatch.py         frame dispatch
    parent_*.py         top-level animation frames
    child_frames.py     child/grandchild recursive frames
    recursive_frames.py final recursive composition helpers
    export_frames.py    export and GIF helpers
    colored*.py         colored-silhouette override layer
  core_analysis.py      core gradient/region analysis
  data.py               generated and benchmark dataset loading
  colors.py             cluster color utilities
  segment.py            angular and Cartesian segmentation helpers
  interface.py          shared plotting options
  ai_client.py          Ollama/API provider client
```

## Smoke Tests

Basic syntax/import check:

```bash
python -m compileall .
```

Runtime sanity check:

```bash
python - <<'PY'
from vmeans.data import generate_structured_points
from vmeans.animation import build_enhanced_visible_frames

points = generate_structured_points("blobs", 1000)
frames = build_enhanced_visible_frames(
    points,
    segments=60,
    center_method="centroid",
    gradient_threshold_ratio=0.25,
    enable_recursion=True,
    max_recursion_depth=1,
    circle_animation_frames=1,
)
print(len(points), len(frames), frames[0].name, frames[-1].name)
PY
```

## Notes

- All Python implementation files in this cleaned package are kept below 1000
  lines.
- The top-level directory intentionally contains only one Python entry point:
  `app.py`.
- The app writes Matplotlib cache files through a temporary writable cache
  directory configured at runtime.
