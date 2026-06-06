# Cohort X — Task 1 — Reproducing submission **v29** (best, public LB 0.72797)

This repository contains everything needed to **fully reproduce `submission_v29.csv`**, our
best submission for MICCAI *Cohort X* Task 1 (extract 7 structured trial fields from PMC
journal articles: `conditions, study_type, sex, minimum_age, maximum_age, eligibility_criteria`).

> **Compliance rule (organizer-enforced):** extraction uses **only the supplied article text**
> (PMC NXML). No external registries (ClinicalTrials.gov / AACT / NCT) are queried at inference.
> Pretrained LLMs are permitted. All inference is CPU-capable and fits the ≥16 GB-RAM rule
> (the 7B GGUF is the largest model; nothing larger than 7B is used at inference).

---

## 1. What v29 is, in one sentence

v29 is built by a **4-stage pipeline**. Each stage takes the previous stage's CSV and overwrites
specific columns, so every column in the final file traces to exactly one stage:

```
Task_1.xlsx + PMC_NXML_Archives/            (organizer inputs)
        │
 STAGE 1  pipeline_v28.py ──────────────►  reference_outputs/submission_v28_base.csv
        │   conditions(vocab+LLM), study_type(BERT clf), sex(rules),
        │   eligibility+min_age+max_age (Mistral-7B RAG)
        │
 STAGE 2  build_v28.py ─────────────────►  reference_outputs/submission_v28.csv
        │   STEP A: re-do conditions (high-confidence vocab merge, build_v7_merge)
        │   STEP B: rewrite eligibility_criteria with the RAFT-tuned BART
        │
 STAGE 3  build_v28b.py ────────────────►  reference_outputs/submission_v28b.csv
        │   force minimum_age = "18 Years", maximum_age = "Not Specified"  (train-optimal)
        │
 STAGE 4  build_v29.py ─────────────────►  reference_outputs/submission_v29.csv   ★ FINAL
            regenerate conditions with Mistral-7B LLM-RAG (diseases-only prompt)
```

### Final provenance of every column in `submission_v29.csv`

| Column | Produced by (final) | Method |
|---|---|---|
| `conditions` | **Stage 4** — `build_v29.py` → `cond_llm_train.py` → `llm_extractor_rag_v28.py` | Mistral-7B-Instruct-v0.3 Q4, k=4 PubMedBERT few-shot from `train_index.json`, "diseases only" prompt |
| `eligibility_criteria` | **Stage 2 (STEP B)** — `build_v28.py` + `prepare_ft_data.py` | RAFT-tuned BART (`bart_raft_v17/final`), 4-beam; falls back to Stage-1 text if degenerate |
| `minimum_age` | **Stage 3** — `build_v28b.py` | constant `"18 Years"` (70.4% train-optimal; ages are not recoverable from article text) |
| `maximum_age` | **Stage 3** — `build_v28b.py` | constant `"Not Specified"` (57.9% train-optimal) |
| `study_type` | **Stage 1** — `extractors.extract_study_type` | fine-tuned PubMedBERT classifier → TF-IDF+LR fallback → keyword rule |
| `sex` | **Stage 1** — `extractors.extract_sex` | rule-based regex over eligibility/abstract text |
| `pmcids` | given | Test sheet of `Task_1.xlsx` |

> Stage 1's conditions and Stage 2's STEP-A conditions are both **overwritten** by Stage 4 —
> they still run (they're in the chain) but do not affect the final `conditions` column.

---

## 2. Environment

```bash
# Interpreter MUST have llama-cpp-python (the project used miniconda base, llama_cpp 0.3.20).
PY=/home/kkolpetinou/miniconda3/bin/python      # NOT the tb-env `python`, which lacks llama_cpp

# Python deps: llama-cpp-python==0.3.20, transformers, torch, scikit-learn,
#              openpyxl, numpy, beautifulsoup4 + lxml
```

### External / pretrained models (not committed — too large for git)

| Model | Role | Location used by the code | Status |
|---|---|---|---|
| `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (4.1 GB) | conditions + eligibility/ages LLM | `/mnt/extra_storage/kkolpetinou/mistral7b_dl/` | ✅ present (re-downloadable from HF: `bartowski/Mistral-7B-Instruct-v0.3-GGUF`) |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | retrieval + BioBERT scoring + study_type base | HF cache (auto-download) | ✅ auto |
| `bart_raft_v17/final` (537 MB) | eligibility rewriter (Stage 2) | `/mnt/extra_storage/kkolpetinou/bart_raft_v17/final` | ✅ present (recreate via §4) |
| `bart_eligibility_v1/final` (intermediate SFT, "v13") | RAFT starting checkpoint | `/mnt/extra_storage/kkolpetinou/bart_eligibility_v1/final` | ✅ present (recreate via §4) |
| **`models/study_type_classifier/`** | study_type classifier (Stage 1) | repo-local `models/study_type_classifier` | ⏳ **PENDING — weights lost, must be retrained (see §4)** |

---

## 3. Reproduce v29 (inference)

Run from the canonical working directory (see *Path note* below). Pure-CPU is the
competition-compliant mode; `N_GPU_LAYERS`/`BART_DEVICE` only speed up dev and give identical output.

```bash
# Stage 1  (~Mistral-7B RAG over 500 test rows)
CUDA_VISIBLE_DEVICES="" $PY pipeline_v28.py          # -> submission_v28_base.csv

# Stage 2  (BART eligibility rewrite + vocab conditions merge)
CUDA_VISIBLE_DEVICES="" $PY build_v28.py             # -> submission_v28.csv

# Stage 3  (constant ages — pure python, instant)
                        $PY build_v28b.py            # -> submission_v28b.csv

# Stage 4  (regenerate conditions with Mistral-7B LLM-RAG)  ★ produces the final file
CUDA_VISIBLE_DEVICES="" $PY build_v29.py             # -> submission_v29.csv
```

**Runtime:** pure CPU ≈ 78 s/article for the 7B (full 500-row Stage 1 or Stage 4 ≈ 11 h each).
With `N_GPU_LAYERS=33` (RTX 4090) each LLM stage is ~3–4 min. Decoding is greedy
(`temperature=0.0`); llama.cpp is near-deterministic but not guaranteed byte-identical across
builds/thread counts, so `conditions` may differ by a token here and there.

**Shortcut (regenerate only the final file):** Stage 4 reads `submission_v28b.csv` (provided in
`reference_outputs/`). Copy it to the working dir and run only `build_v29.py`.

---

## 4. How the trained models are made (from-scratch provenance)

### (a) `train_index.json` — RAG retrieval index
`build_train_index.py` embeds title+abstract of each **train** article with PubMedBERT and stores
embeddings + GT labels. Consumed by `llm_extractor_rag_v28._retrieve` and `cond_llm_train`.
```bash
$PY build_train_index.py        # Task_1.xlsx + PMC_NXML_Archives/ -> train_index.json
```

### (b) BART eligibility model (two stages: SFT → RAFT)
```
facebook/bart-base
   │  train_bart_eligibility.py   (SFT on training_data/ft_train.jsonl + ft_val.jsonl)
   ▼
bart_eligibility_v1/final   ("v13")
   │  RAFT refinement:
   │    raft_step1_sample.py   sample best-of-N eligibility candidates from v13
   │    raft_step2_score.py    score candidates -> training_data/raft_train_best.jsonl
   │    raft_step3_train.py    continue-train v13 on raft_train_best.jsonl (+ ft_val.jsonl)
   │    raft_step4_validate.py validate
   ▼
bart_raft_v17/final         (the eligibility rewriter used in Stage 2)
```
`ft_train.jsonl` / `ft_val.jsonl` are built from `Task_1.xlsx` by `prepare_ft_data.py`
(`build_input_text` is also imported at inference by `build_v28.py`).

### (c) ⏳ PENDING — `models/study_type_classifier/`  (study_type, Stage 1)
The weights were saved into the **git-ignored local `models/` folder** of the original repo and
have since been **lost** (only `study_type_train.log` survives in `/mnt/extra_storage`). They are
**not** in `/mnt/extra_storage`. Recreate with:
```bash
$PY train_study_type_classifier.py     # PubMedBERT, 416 train rows, 5 epochs (~14 s GPU)
                                       # -> models/study_type_classifier/  (train acc ~97.8%)
```
**Reproduction impact:** if Stage 1 runs *without* this directory, `extract_study_type` silently
falls back to a runtime TF-IDF+LR model → **different** study_type predictions than the submitted
file. For an exact match, either retrain (a retrained net won't be byte-identical, so a few
predictions may still differ) **or** treat the submitted `study_type` column in
`reference_outputs/submission_v28b.csv` as a fixed artifact.

---

## 5. File manifest

### Code — pipeline (Stages 1–4)
| File | Role | Status |
|---|---|---|
| `pipeline_v28.py` | Stage 1 driver (all 7 fields, base CSV) | ✅ |
| `build_v28.py` | Stage 2 (conditions vocab merge + BART eligibility) | ✅ |
| `build_v28b.py` | Stage 3 (constant ages) | ✅ |
| `build_v29.py` | Stage 4 (LLM-RAG conditions) — **final** | ✅ |
| `cond_llm_train.py` | Stage 4 conditions extractor + leakage-free eval | ✅ |
| `build_v7_merge.py` | Stage 2 STEP-A high-confidence conditions | ✅ |
| `extractors.py` | study_type + sex + conditions vocab | ✅ |
| `llm_extractor_rag_v28.py` | Mistral-7B RAG (eligibility/ages + `_get_llm`/`_retrieve`) | ✅ |
| `llm_extractor.py`, `llm_extractor_rag.py` | conditions LLM fallback / shared RAG utils | ✅ |
| `nxml_parser.py` | NXML → title/abstract/body/keywords (stdlib) | ✅ |
| `prepare_ft_data.py` | `build_input_text` for BART + builds ft_*.jsonl | ✅ |
| `pipeline_v6.py` | `_is_negated`, `_EXTRA_BLOCKLIST` (used by build_v7_merge) | ✅ |
| `evaluate.py` | official-style metrics (BioBERT sim, FM3S, age Jaccard) | ✅ |

### Code — model training / index building
| File | Role | Status |
|---|---|---|
| `build_train_index.py` | builds `train_index.json` | ✅ |
| `train_bart_eligibility.py` | SFT `facebook/bart-base` → `bart_eligibility_v1` | ✅ |
| `raft_step1_sample.py` … `raft_step4_validate.py` | RAFT → `bart_raft_v17` | ✅ |
| `run_bart_v17.py` | standalone BART eligibility inference | ✅ |
| `train_study_type_classifier.py` | builds the (PENDING) study_type classifier | ✅ |
| `simulate_age_policies.py`, `validate_age_detector.py` | evidence for the constant-age choice | ✅ |

### Data / inputs
| Path | Role | Status |
|---|---|---|
| `Task_1.xlsx` | organizer data (Train 416 / Test 500) | ✅ |
| `PMC_NXML_Archives/` | 950 article NXML files | ✅ |
| `train_index.json` | PubMedBERT RAG index (recreatable via `build_train_index.py`) | ✅ |
| `training_data/ft_train.jsonl`, `ft_val.jsonl`, `ft_data.jsonl` | BART SFT data | ✅ |
| `training_data/raft_train_best.jsonl` | RAFT best-of-N training set | ✅ |

### Reference outputs (verification checkpoints)
| Path | Status |
|---|---|
| `reference_outputs/submission_v28_base.csv` (Stage 1) | ✅ |
| `reference_outputs/submission_v28.csv` (Stage 2) | ✅ |
| `reference_outputs/submission_v28b.csv` (Stage 3 — input to `build_v29.py`) | ✅ |
| `reference_outputs/submission_v29.csv` (**final, expected output**) | ✅ |

### Models — see §2 table (Mistral GGUF, BART RAFT/SFT, PubMedBERT = external/large; **study_type_classifier = ⏳ PENDING**)

---

## 6. Path note & known gotchas

- **Absolute paths:** these are the *exact* scripts that produced v29. Several hardcode the
  original working dir `/home/kkolpetinou/cohort-x-task-1/...` and the model dirs under
  `/mnt/extra_storage/kkolpetinou/...`. To reproduce, run with this repo placed at (or symlinked
  to) `/home/kkolpetinou/cohort-x-task-1`, and the external models at their documented paths.
  Files that use `os.path.dirname(__file__)` are already location-independent.
- **`nct_cache.json` is dead code.** `pipeline_v28.py` defines `_get_nct_cache` /
  `_normalise_ct_age` but never calls them, and the file is intentionally **not** shipped — it
  confirms no external registry is read at inference.
- **Interpreter:** the default `python` here (tb-env) lacks `llama_cpp`; use
  `/home/kkolpetinou/miniconda3/bin/python`.
- **True-CPU compliance:** export `CUDA_VISIBLE_DEVICES=""` — the CUDA build of llama.cpp reserves
  ~2 GB GPU even at `N_GPU_LAYERS=0` otherwise.

## 7. Verify

```bash
diff <(sort reference_outputs/submission_v29.csv) <(sort submission_v29.csv)
# or score against train GT with evaluate.py (BioBERT == official metric, confirmed)
```
