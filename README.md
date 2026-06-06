# Cohort X — Task 1 — Reproducing submission **v29** (public LB 0.72797)

Extract 6 structured fields from PMC journal articles (NXML):
`conditions, study_type, sex, minimum_age, maximum_age, eligibility_criteria`.

> **Compliance:** article text only — no external registries (ClinicalTrials.gov / AACT / NCT)
> are queried at inference. Pretrained LLMs are allowed. Everything is CPU-capable and fits the
> ≥16 GB-RAM rule (the 7B GGUF is the largest model used).

---

## 1. The pipeline (one pass, one method per field)

`reproduce_v29.py` runs a **single pass** over the 500 test articles. Each field is produced
exactly once, by one method:

| Field | Method | Code |
|---|---|---|
| `conditions` | **Mistral-7B-Instruct-v0.3 Q4**, LLM-RAG: k=4 PubMedBERT few-shot examples retrieved from `train_index.json`; "diseases only" prompt | `cond_llm_train.extract_conditions_llm` → `llm_extractor_rag_v28` |
| `eligibility_criteria` | **RAFT-tuned BART** (`bart_raft_v17/final`), 4-beam decode | `reproduce_v29.py` + `prepare_ft_data.build_input_text` |
| `study_type` | **fine-tuned PubMedBERT** classifier (INTERVENTIONAL / OBSERVATIONAL) | `extractors.extract_study_type` |
| `sex` | rule-based regex over eligibility/abstract text | `extractors.extract_sex` |
| `minimum_age` | constant **`"18 Years"`** | `reproduce_v29.py` |
| `maximum_age` | constant **`"Not Specified"`** | `reproduce_v29.py` |

**Why the ages are constants:** the official age metric is Jaccard on extracted numbers with no
partial credit, and the ground-truth ages come from the trial registry, *not* the paper. On the
416-row train set the constants are optimal (`minimum_age="18 Years"` 70.4%,
`maximum_age="Not Specified"` 57.9%); every text/LLM extraction we tried scored lower.

```
Task_1.xlsx ('Test' sheet, 500 ids) + PMC_NXML_Archives/
        │
        ▼   reproduce_v29.py   (one pass; the 4 models/rules above)
        ▼
   submission_v29.csv
```

That is the whole inference pipeline. (Sections 4–5 cover how the two trained models were built.)

---

## 2. Reproduce

```bash
PY=/home/kkolpetinou/miniconda3/bin/python    # interpreter WITH llama-cpp-python 0.3.20
                                              # (the default tb-env `python` does NOT have it)

# competition-compliant pure CPU (~78 s/article for the 7B → ~11 h for 500 rows):
CUDA_VISIBLE_DEVICES="" $PY reproduce_v29.py

# dev speed, identical output — offload both models to a GPU:
N_GPU_LAYERS=33 BART_DEVICE=cuda $PY reproduce_v29.py
```

Output `submission_v29.csv` should match `reference_outputs/submission_v29.csv`:

```bash
diff <(sort submission_v29.csv) <(sort reference_outputs/submission_v29.csv)
```

**Determinism:** Mistral runs greedy (`temperature=0.0`); llama.cpp is near-deterministic but not
guaranteed byte-identical across builds/thread counts, so a few `conditions` tokens may differ.
See §5 for the one expected difference in `study_type`.

### Python deps
`llama-cpp-python==0.3.20, transformers, torch, scikit-learn, openpyxl, numpy, beautifulsoup4, lxml`

### Models
| Model | Field | Default path | How to get it |
|---|---|---|---|
| `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (4.1 GB) | conditions | `$MISTRAL_GGUF` or `/mnt/extra_storage/kkolpetinou/mistral7b_dl/` | download from HF `bartowski/Mistral-7B-Instruct-v0.3-GGUF` |
| `bart_raft_v17/final` (533 MB) | eligibility | `$BART_DIR` or `/mnt/extra_storage/kkolpetinou/bart_raft_v17/final` | **too large for git** — GitHub Release asset, or retrain (§4) |
| `models/study_type_classifier/` (533 MB) | study_type | repo-local `models/study_type_classifier` | ⏳ **PENDING** — weights were lost; retrain (§4/§5) |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | retrieval + study_type base | HF cache | auto-download |

Model paths are overridable by env var (`MISTRAL_GGUF`, `BART_DIR`); data paths resolve relative
to the repo, so it runs from any location.

---

## 3. File manifest

**Inference (the §1 pipeline):**
`reproduce_v29.py`, `nxml_parser.py`, `prepare_ft_data.py`, `cond_llm_train.py`,
`llm_extractor_rag_v28.py`, `llm_extractor_rag.py`, `pipeline_v6.py`, `extractors.py`,
`evaluate.py` (metrics / verification).

**Model training (§4):**
`build_train_index.py`, `train_bart_eligibility.py`, `raft_step1_sample.py … raft_step4_validate.py`,
`train_study_type_classifier.py`.

**Data:** `Task_1.xlsx`, `PMC_NXML_Archives/` (950 articles), `train_index.json`,
`training_data/{ft_train,ft_val,ft_data}.jsonl`, `training_data/raft_train_best.jsonl`.

**Reference:** `reference_outputs/submission_v29.csv` (the authoritative final file).

---

## 4. How the two trained models are made (provenance)

### `train_index.json` — RAG retrieval index
`build_train_index.py` embeds title+abstract of each **train** article with PubMedBERT, storing
embeddings + GT labels. Consumed by the conditions retriever.
```bash
$PY build_train_index.py            # Task_1.xlsx + PMC_NXML_Archives/ -> train_index.json
```

### Eligibility BART — two stages: SFT → RAFT
```
facebook/bart-base
   │  train_bart_eligibility.py   (SFT on training_data/ft_train.jsonl + ft_val.jsonl)
   ▼  bart_eligibility_v1/final
   │  RAFT:  raft_step1_sample.py  -> sample best-of-N eligibility candidates
   │         raft_step2_score.py   -> training_data/raft_train_best.jsonl
   │         raft_step3_train.py    -> continue-train on the best candidates
   │         raft_step4_validate.py
   ▼  bart_raft_v17/final          (the eligibility model used at inference)
```
`ft_train.jsonl` / `ft_val.jsonl` are derived from `Task_1.xlsx` by `prepare_ft_data.py`.

### Study_type classifier — ⏳ PENDING
```bash
$PY train_study_type_classifier.py  # PubMedBERT, 416 train rows, 5 epochs (~14 s GPU)
                                    # -> models/study_type_classifier/   (train acc ~97.8%)
```

---

## 5. Known reproduction caveats

- ⏳ **study_type classifier weights are lost.** They were written to the original repo's
  git-ignored `models/` folder (not `/mnt/extra_storage`) and deleted. Retrain with
  `train_study_type_classifier.py`. A retrained net is **not** byte-identical to the original, so
  a small number of test `study_type` labels may differ from `reference_outputs/submission_v29.csv`
  — which remains the authoritative record. If `models/study_type_classifier/` is absent at run
  time, `extract_study_type` silently falls back to a runtime TF-IDF+LR model (different output).
- **Large weights aren't in git** (533 MB each > GitHub's 100 MB limit). Use the Release assets or
  retrain via §4.
- **Interpreter:** use `/home/kkolpetinou/miniconda3/bin/python` (has `llama_cpp`); the default
  `python` (tb-env) does not.
- **True-CPU compliance:** export `CUDA_VISIBLE_DEVICES=""` — the CUDA build of llama.cpp reserves
  ~2 GB GPU even at `N_GPU_LAYERS=0`.
