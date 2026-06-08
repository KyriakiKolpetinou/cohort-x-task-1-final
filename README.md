# Cohort X — Task 1 — reproduce submission v29

A fresh run of `reproduce_v29.py` rebuilds our submission and scores **0.72864** on the public
leaderboard (slightly above the original v29 at 0.72797). It extracts 6 fields from PMC articles
using **article text only** — no external registries, CPU-capable, nothing larger than a 7B model.

## How each field is produced

| field | method |
|---|---|
| `conditions` | Mistral-7B LLM-RAG — few-shot from the most similar train articles (`conditions.py`) |
| `eligibility_criteria` | fine-tuned BART (`reproduce_v29.py`) |
| `study_type` | fine-tuned PubMedBERT classifier (`study_type_and_sex.py`) |
| `sex` | keyword rules (`study_type_and_sex.py`) |
| `minimum_age` | constant `"18 Years"` |
| `maximum_age` | constant `"Not Specified"` |

Ages are constants because the ground-truth ages come from the trial registry, not the paper — on
the train set the constants beat every extraction attempt.

## Run it

```bash
PY=/home/kkolpetinou/miniconda3/bin/python      # has llama-cpp-python; the default `python` does not

# 1) get the two trained models (GitHub Release 'weights-v29')
BASE=https://github.com/KyriakiKolpetinou/cohort-x-task-1-final/releases/download/weights-v29
curl -L $BASE/study_type_classifier.tar.gz | tar -xz -C models      # -> models/study_type_classifier/
curl -L $BASE/bart_raft_v17_final.tar.gz | tar -xz && export BART_DIR="$PWD/final"
# Mistral GGUF: download from HF (bartowski/Mistral-7B-Instruct-v0.3-GGUF) and set $MISTRAL_GGUF

# 2) run  (GPU. For competition-compliant pure CPU: CUDA_VISIBLE_DEVICES="" and drop N_GPU_LAYERS, ~11h)
N_GPU_LAYERS=33 BART_DEVICE=cuda $PY reproduce_v29.py               # -> submission_v29.csv
```

Check it matches the submitted file:
`diff <(sort submission_v29.csv) <(sort reference_outputs/submission_v29d.csv)`

Deps: `llama-cpp-python==0.3.20, transformers, torch, scikit-learn, openpyxl, numpy, beautifulsoup4, lxml`

## Files

- **Inference (all you run):** `reproduce_v29.py` (driver) + `nxml_parser.py`, `conditions.py`, `study_type_and_sex.py`
- **Data:** `Task_1.xlsx`, `PMC_NXML_Archives/`, `train_index.json`
- **Submitted file:** `reference_outputs/submission_v29d.csv` (LB 0.72864)
- **Model-training scripts:** see *Rebuilding the models* below

## Rebuilding the models (optional — only if not using the Release weights)

```bash
$PY build_train_index.py             # -> train_index.json  (PubMedBERT index of train articles)
$PY train_study_type_classifier.py   # -> models/study_type_classifier/  (SEED=7, the 0.72864 model)
# eligibility BART:  facebook/bart-base --train_bart_eligibility.py(SFT)--> then RAFT
#                    (raft_step1..4_*.py) --> bart_raft_v17/final
```

## Notes

- **Weights live in the GitHub Release `weights-v29`**, not in git (each >100 MB). Mistral GGUF is a
  separate HF download.
- **study_type** — the original classifier weights were lost. `study_type_seed_sweep.py` retrained
  under 10 seeds and kept the one closest to the original's predictions (**seed 7**, now pinned in
  `train_study_type_classifier.py`); it scores 0.72864. The other 5 columns reproduce byte-for-byte.
- Mistral decodes greedily (`temperature=0`), so output is stable run-to-run apart from rare token
  differences in `conditions`.
