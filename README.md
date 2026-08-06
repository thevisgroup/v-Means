# V-Means

This repository contains two independently runnable research components:

1. the V-Means visible and explainable clustering desktop application; and
2. `vlm_userstudy`, a standalone toolkit that evaluates clustering videos with
   multiple visual-language models as study participants.

This directory is the runnable, reorganized copy of the dissertation app. The
original working files in the parent `dissertation/` folder are left untouched.
The desktop application's public entry point is:

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

## Standalone VLM User Study

[`vlm_userstudy/`](vlm_userstudy/) is an independent command-line research
module. It does not import or launch the Qt GUI, and it keeps its own
dependencies, tests, model registry, prompts, transcripts, CSV outputs, scoring
tools, and optional Google Sheets uploader.

The runner uses visual-language models as study participants in the same
four-video questionnaire flow used for the human study. It talks to an
OpenAI-compatible Chat Completions endpoint with native video input. The module
currently ships formal configurations and vLLM launch scripts for five models
across the Qwen, MiniCPM-V, InternVL, and GLM families:

- Qwen3-VL-8B-Instruct
- Qwen3.5-9B
- MiniCPM-V 4.5
- InternVL3.5-38B
- GLM-4.6V

These are the fixed models for the current experiment, not a hard-coded
provider limit. Additional OpenAI-compatible VLMs can be added through the
registry in `vlm_userstudy/config.py`, together with their sampling settings,
frame limits, and server launcher. Model-specific changes should be validated
with the included dry run, regression tests, and an isolated pilot before a
formal run.

The module is installed and run separately from the desktop app:

```bash
cd vlm_userstudy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python runner.py --help
python push_to_sheet.py --help
```

The four study MP4 files are intentionally not duplicated in this repository.
Download them before a pilot or formal run:

```bash
bash download_videos.sh
python runner.py --tag qwen3vl-8b --dry-run
```

Formal sessions require a compatible model endpoint, normally started on the
trusted GPU server with the matching `serve/*.sh` script. These launchers are
research scripts, not hardened public services: do not expose port 8000 to an
untrusted network; prefer localhost or an SSH tunnel plus firewall and access
controls. Google Sheets publishing is optional and uses
`GOOGLE_APPLICATION_CREDENTIALS` and `STUDY_SHEET_ID`. Keep the service-account
JSON outside the repository. Transcripts, CSV results, downloaded media, and
other local runtime artifacts under the module are ignored by Git. See
[`vlm_userstudy/README.md`](vlm_userstudy/README.md) for the full experimental
protocol, model matrix, security boundary, scoring rules, pilot workflow, and
reproducibility metadata.

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
app.py                  desktop GUI entry point
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

vlm_userstudy/          independent VLM-as-participant research toolkit
  config.py             videos, run settings, and extensible model registry
  questionnaire.py      prompts, answer schemas, and validation
  runner.py             dry-run, pilot, and formal multi-turn sessions
  score.py              response scoring and format-compliance metrics
  push_to_sheet.py      optional Google Sheets publishing
  serve/                formal and legacy/reference vLLM launch scripts
  tests/                standalone regression suite
  videos/               local study stimuli (downloaded, not tracked)
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

Standalone VLM module checks (run after installing its own requirements):

```bash
cd vlm_userstudy
python -m unittest discover -s tests -v
python runner.py --tag qwen3vl-8b --dry-run
# After a pilot or formal run:
python score.py outputs/pilot/vlm_responses.csv
```

## Notes

- All Python implementation files in this cleaned package are kept below 1000
  lines.
- The desktop GUI keeps one top-level Python entry point, `app.py`; the
  independent VLM study commands live under `vlm_userstudy/`.
- The app writes Matplotlib cache files through a temporary writable cache
  directory configured at runtime.
