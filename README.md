# TattooAudit

**A benchmark-grounded interactive expert-review system for open-set tattoo analysis**

Artifact repository for the paper:

> **TattooAudit: A Benchmark-Grounded Interactive Expert-Review System for Open-Set Tattoo Analysis**

TattooAudit combines a controlled Vision–Language Model (VLM) benchmark with a Streamlit-based expert-review workspace. The system transforms precomputed model outputs and benchmark discrepancies into a transparent workflow for case triage, visual inspection, cross-model comparison, structured review actions, decision history, and exportable audit records.

The reference annotation remains authoritative. Model predictions provide supporting evidence for expert inspection and are not treated as final decisions.

---

## Quick start

### Requirements

For the dashboard-only installation:

- Git
- Python 3.11 or a compatible recent Python 3 version
- Approximately 1 GB of available disk space for the repository and environment
- No GPU is required to inspect the precomputed results
- No VLM weights need to be downloaded to run the interactive dashboard

### 1. Download the repository

Using Git:

```bash
git clone https://github.com/professorvargas/tattoo-audit-artifact.git
cd tattoo-audit-artifact
```

Alternatively, use **Code → Download ZIP** on GitHub, extract the archive, and open a terminal inside the extracted `tattoo-audit-artifact` directory.

### 2. Create the software environment

#### Option A — Lightweight dashboard environment

This option installs only the packages required to open the interactive TattooAudit workspace:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit pandas pillow
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

#### Option B — Full Conda environment

The repository also provides a broader environment for the research artifact:

```bash
conda env create -f environment.yml
conda activate tattoo
```

The Conda environment includes the packages used by the benchmark pipeline and is substantially larger than the lightweight dashboard environment.

### 3. Run TattooAudit

From the repository root:

```bash
streamlit run mvp_audit_streamlit.py
```

Streamlit normally opens the application automatically in the default browser. When it does not, open:

```text
http://localhost:8501
```

To stop the application, return to the terminal and press:

```text
Ctrl+C
```

---

## What is included

The public repository includes the artifacts required to inspect the study results:

- the TattooAudit Streamlit source code;
- public TSSD2023 test images and reference masks;
- class-specific crops generated from the reference masks;
- black- and white-background crop conditions;
- precomputed outputs from the three evaluated VLMs;
- normalized prediction tables and case-level metrics;
- scripts used for preprocessing, parsing, evaluation, and analysis; and
- support files for review-queue construction and audit-priority calculation.

Because the predictions and evaluation outputs are already included under `runs/`, a user can open the dashboard without installing or executing Gemma, Qwen, LLaMA, Ollama, or Hugging Face model weights.

---

## What the dashboard does

TattooAudit opens a fixed expert-review workspace over the public `test_open` split. The interface allows the reviewer to:

1. inspect a ranked and filterable review queue;
2. select a tattoo case;
3. view the original image and reference mask;
4. inspect class-specific crops on black and white backgrounds;
5. compare predictions from all evaluated VLMs and input conditions;
6. inspect the indicators that produced the audit-priority score;
7. confirm the current reference annotation;
8. propose a revised reference;
9. mark a case as ambiguous;
10. route a case to re-annotation;
11. exclude a case from a high-confidence pool;
12. record reviewer confidence, operational relevance, and rationale; and
13. export review and audit records.

---

## System boundary

TattooAudit separates the offline benchmark pipeline from the interactive review application.

### Offline benchmark layer

The offline layer:

- reads TSSD2023 images, masks, and reference labels;
- creates class-specific crops;
- renders each crop on black and white backgrounds;
- executes the VLMs;
- parses and normalizes textual outputs;
- aggregates crop-level predictions at image level;
- computes semantic metrics; and
- stores the resulting artifacts.

### Interactive review layer

The Streamlit application:

- loads the precomputed images, predictions, and metrics;
- constructs the review queue;
- displays visual and model evidence;
- calculates transparent priority indicators;
- records structured expert decisions; and
- exports review logs.

The current dashboard does **not**:

- run segmentation models;
- generate masks during review;
- create experimental crops during review;
- invoke the VLMs during the interactive session;
- process new operational images end to end;
- infer identity, criminal affiliation, intent, ethnicity, religion, or cultural membership; or
- represent a validated operational forensic system.

---

## Dataset

The study uses the public test sets of the **Tattoo Semantic Segmentation Dataset 2023 (TSSD2023)**:

```text
https://github.com/Brilhador/tssd2023
```

TSSD2023 provides original tattoo images, pixel-level semantic masks, and reference labels for closed-set and open-set evaluation.

Only publicly redistributable test data and derived artifacts are included in this repository. The non-public TSSD2023 training and validation images are not included.

---

## Experimental protocol

The study evaluates zero-shot, multi-label semantic naming under oracle-segmentation conditions.

Reference masks serve as oracle segmentations so that the experiment can analyze semantic naming behavior without conflating naming errors with mask-prediction errors.

Each case is evaluated under three visual conditions:

- **Original image:** the unmodified full image;
- **GT-crop black:** a class-specific reference-mask foreground rendered on a black background; and
- **GT-crop white:** the same class-specific foreground rendered on a white background.

For images containing multiple reference classes, the pipeline processes each class-specific crop independently and forms the image-level prediction from the union of normalized crop-level labels.

---

## Evaluated models

The benchmark includes outputs from:

- **Gemma3:12B**
- **Qwen2.5-VL-7B**
- **LLaMA 3.2 Vision 11B**

All models operate under a controlled 35-label output vocabulary that excludes `background` and includes `unknown`.

The repository contains precomputed outputs and evaluation artifacts, not the model weights.

---

## Metrics

For each image:

- `GT` is the set of reference labels;
- `Pred` is the normalized set of predicted labels.

The evaluation computes:

```text
TP = |GT ∩ Pred|
FP = |Pred \ GT|
FN = |GT \ Pred|
```

Dataset-level micro-F1 is computed as:

```text
micro-F1 = 2TP / (2TP + FP + FN)
```

Additional measures include:

- precision;
- recall;
- Jaccard index;
- exact-set agreement;
- false positives per image;
- false negatives per image;
- `unknown` frequency;
- prediction-set size and overprediction;
- black/white exact stability;
- black/white flip rate; and
- inference runtime.

The reported F1 evaluates semantic label sets. It is not a segmentation Dice score or mIoU because the evaluated VLMs do not predict masks in this protocol.

---

## Audit priority

TattooAudit uses a deterministic and inspectable priority heuristic to rank cases for expert attention.

Current indicators include:

- presence of the `unknown` label;
- disagreement between black- and white-background crops;
- divergence between original-image and crop predictions; and
- high false-positive counts.

The audit-priority score is a relative triage index. It must not be interpreted as an error probability or an automated decision.

The interface presents the score through the following qualitative bands:

- Very low
- Low
- Medium
- High
- Critical

---

## Review records

When a reviewer saves a decision, TattooAudit creates the `audit_logs/` directory automatically when needed.

Structured review records are stored under this directory, including:

```text
audit_logs/expert_review_log.csv
```

A review record may contain:

- timestamp;
- case identifier;
- reviewer identifier;
- review action and status;
- original reference labels;
- proposed or reviewed labels;
- reviewer confidence;
- operational relevance;
- audit-priority score and band;
- triggered indicators; and
- decision rationale.

Review records should be anonymized before public release when they contain reviewer names or sensitive comments.

---

## Repository structure

```text
tattoo-audit-artifact/
├── data_meta/                    # Class mapping and small metadata files
├── datasets/
│   ├── test_closed/              # Public closed-set images and masks
│   ├── test_open/                # Public open-set images and masks
│   ├── crops_gt/                 # Reference class crops
│   ├── crops_gt_black/           # Reference crops on black background
│   └── crops_gt_white/           # Reference crops on white background
├── experiments/                  # Experimental and evaluation code
├── exploration/                  # Exploratory notebooks and scripts
├── runs/
│   ├── gemma3/
│   ├── qwen2_5_vl/
│   ├── llama3_2_vision/
│   └── _shared_links/
├── scripts/                      # Supporting analysis scripts
├── environment.yml               # Full Conda environment
├── mvp_audit_streamlit.py        # Main TattooAudit application
├── run_experiments.py            # Benchmark orchestration
└── README.md
```

---

## Troubleshooting

### `No test_open cases were found`

Run the command from the repository root:

```bash
cd tattoo-audit-artifact
streamlit run mvp_audit_streamlit.py
```

Then confirm that the following directories exist:

```text
datasets/test_open/
datasets/crops_gt/test_open/
datasets/crops_gt_black/test_open/
datasets/crops_gt_white/test_open/
runs/gemma3/
runs/qwen2_5_vl/
runs/llama3_2_vision/
```

### `ModuleNotFoundError`

Activate the environment before launching Streamlit:

```bash
source .venv/bin/activate
```

or:

```bash
conda activate tattoo
```

Then reinstall the dashboard packages when necessary:

```bash
python -m pip install streamlit pandas pillow
```

### Port 8501 is already in use

Run the application on another port:

```bash
streamlit run mvp_audit_streamlit.py --server.port 8502
```

### Images or predictions are missing

Confirm that the repository download completed successfully and that `datasets/` and `runs/` contain their tracked files. Running only the Python source without the precomputed artifacts does not reproduce the interactive review workspace.

---

## Reproducibility notes

- The dashboard uses precomputed benchmark artifacts and does not require live VLM inference.
- The interactive review queue currently focuses on `test_open`.
- The reference annotation remains authoritative.
- Proposed revisions are stored separately rather than silently replacing benchmark labels.
- The audit-priority rules and weights are manually specified and should be treated as transparent triage logic, not as calibrated risk estimates.
- The current artifact validates the technical workflow, but it does not establish usability, decision quality, or operational effectiveness with domain experts.

---

## Paper

**Ricardo Alexandre Vargas Pereira, Rodrigo Tchalski da Silva, Clayton Kossoski, Heitor Silvério Lopes.**

*TattooAudit: A Benchmark-Grounded Interactive Expert-Review System for Open-Set Tattoo Analysis.*

Citation metadata will be updated after publication.

---

## Acknowledgments

This artifact builds on the public TSSD2023 benchmark and its open-set tattoo semantic-segmentation protocol.

The authors acknowledge the Federal University of Technology – Paraná (UTFPR) and the LABIC research infrastructure used in the experimental study.
