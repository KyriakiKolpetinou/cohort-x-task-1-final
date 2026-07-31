# Cohort X — Task 1 — 2nd place — reproduce submission v29f

**Final (private) leaderboard score: 0.71095 — 2nd place in Task 1.** (Public: 0.72941.)

A run of `reproduce_v29.py` rebuilds the submitted file, `submission_v29f.csv`. It
extracts 6 fields from PMC articles using **article text only** — no external
registries, CPU-capable.

## How each field is produced

| field | method |
|---|---|
| `conditions` | Qwen2.5-3B LLM-RAG — few-shot from the most similar train articles, over title+abstract+keywords (`conditions.py`) |
| `eligibility_criteria` | fine-tuned BART (`reproduce_v29.py`) |
| `study_type` | fine-tuned PubMedBERT classifier (`study_type_and_sex.py`) |
| `sex` | keyword rules (`study_type_and_sex.py`) |
| `minimum_age` | constant `"18 Years"` |
| `maximum_age` | extracted where a genuine age bound fires, else `"Not Specified"` (`age_extractor.py`) |

### Ages

The gold ages come from the trial registry, not the paper — so the train-optimal
constants (`"18 Years"` / `"Not Specified"`) are a strong baseline, right ~70% of the
time for min and ~58% for max. But for many trials the age criterion is *also* stated
in the paper's eligibility/methods text, so `age_extractor.py` overrides the constant
**only** when a high-precision, age-anchored pattern fires.

The metric is Jaccard over numbers, so precision beats recall — a wrong extra number
costs as much as a missing one. Measured on train: extracting `maximum_age` gains
+0.0028 overall; extracting `minimum_age` **hurts** (adult trials cite subgroup ages
that mislead it), so `EXTRACT_MIN = False` and min stays constant. On the 500 test
rows the extractor fires on 49, all of them `Not Specified` → a real bound.

## Run it

```bash
# 0) deps (llama-cpp-python is the one the system `python` usually lacks)
pip install "llama-cpp-python==0.3.20" transformers torch scikit-learn openpyxl \
            numpy beautifulsoup4 lxml

# 1) fetch the two trained models (GitHub Release 'weights-v29') into models/
mkdir -p models/bart_raft_v17_final
BASE=https://github.com/KyriakiKolpetinou/cohort-x-task-1-final/releases/download/weights-v29
curl -L $BASE/study_type_classifier.tar.gz | tar -xz -C models                    # -> models/study_type_classifier/
curl -L $BASE/bart_raft_v17_final.tar.gz  | tar -xz -C models/bart_raft_v17_final # -> models/bart_raft_v17_final/final/

# 2) Qwen GGUF from HuggingFace (bartowski/Qwen2.5-3B-Instruct-GGUF)
curl -L -o models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# 3) run — competition-compliant pure CPU (~3 h for the 3B over 500 rows)
CUDA_VISIBLE_DEVICES="" python reproduce_v29.py                     # -> submission_v29f.csv

# ...or dev speed, identical output: offload both models to GPU
N_GPU_LAYERS=33 BART_DEVICE=cuda python reproduce_v29.py
```

Every model path defaults to `models/` inside the repo, so no env vars are needed if you
follow the steps above. To keep weights elsewhere, set `CONDITIONS_GGUF`, `BART_DIR`, or
`NXML_DIR`.

Check it matches the submitted file:
`diff <(sort submission_v29f.csv) <(sort reference_outputs/submission_v29f.csv)`

## Files

- **Inference (all you run):** `reproduce_v29.py` (driver) + `nxml_parser.py`, `conditions.py`, `study_type_and_sex.py`, `age_extractor.py`
- **Data:** `Task_1.xlsx`, `PMC_NXML_Archives/`, `train_index.json`
- **Submitted file:** `reference_outputs/submission_v29f.csv` (private 0.71095, 2nd place)
- **Model-training scripts:** see *Rebuilding the models* below

## Rebuilding the models (optional — only if not using the Release weights)

```bash
python build_train_index.py             # -> train_index.json  (PubMedBERT index of train articles)
python train_study_type_classifier.py   # -> models/study_type_classifier/  (SEED=7, the 0.72864 model)

# eligibility BART: facebook/bart-base --SFT--> bart_eligibility_v1 --RAFT--> bart_raft_v17_final
python train_bart_eligibility.py        # -> models/bart_eligibility_v1/final   (v13, the SFT base)
python raft_step1_sample.py             # -> raft_candidates.jsonl
python raft_step2_score.py              # -> raft_train_best.jsonl, raft_stats.json
python raft_step3_train.py              # -> models/bart_raft_v17_final/final   (v17, used by the driver)
python raft_step4_validate.py           # compares v13 vs v17 on held-out train
```

These write into `models/` inside the repo, matching the defaults the driver reads.
Override with `OUT_DIR`, `MODEL_DIR`, `START_FROM`, `BART_DIR` to store weights elsewhere.

## Notes

- **Weights live in the GitHub Release `weights-v29`**, not in git (each >100 MB). The Qwen GGUF is a
  separate HF download.
- **study_type** — `study_type_seed_sweep.py` trained under 10 seeds and kept the best (**seed 7**, now pinned in
  `train_study_type_classifier.py`); it scores 0.72864 on this field.
- Qwen decodes greedily (`temperature=0`), so output is stable run-to-run apart from rare token
  differences in `conditions`.
- **`conditions` prompt context** — `conditions.py` reads `CONDITIONS_EXTRA`, default `''`
  (title+abstract+keywords), which is the submitted v29f setting. Setting it to `methods`
  appends the methods text; that variant is v29g. See *Public vs private* below.
- The current leaderboard view shows `v29d` and `v29d_qwencond` as *Error*; that is an
  organiser-side artefact introduced when the rankings were revised. Both scored normally
  at submission time and the figures below are the ones they returned.

## Submission lineage

Each submission changes exactly **one field** relative to its parent, so the whole family
is a small grid. Two suffixes carry the meaning:

- **`_qwencond`** — the `conditions` column regenerated with **Qwen2.5-3B** instead of the
  original Mistral-7B (commit `c964bce`).
- **`d`** — the **seed-7** `study_type` retrain (`study_type_seed_sweep.py` swept 10 seeds
  and kept the best).

| submission | conditions | study_type | maximum_age |
|---|---|---|---|
| v29 | Mistral-7B | original | constant |
| v29d | Mistral-7B | **seed-7** | constant |
| v29_qwencond | **Qwen2.5-3B** | original | constant |
| v29d_qwencond | **Qwen2.5-3B** | **seed-7** | constant |
| **v29f (submitted)** | Qwen2.5-3B | seed-7 | **extracted** |
| v29g | **Qwen2.5-3B +methods** | seed-7 | constant |

v29d_qwencond is the common parent; **v29f and v29g are two independent branches off it.**

> **Filename note:** `reference_outputs/submission_v29e.csv` is byte-identical to the file
> uploaded as `submission_v29d_qwencond.csv` (md5 `5be00f7c`). The repo and the leaderboard
> used different names for the same submission.

## Public vs private

Public and private rank the scored submissions in **near-inverted** order:

| submission | public | private |
|---|---|---|
| v29 | 0.72773 | 0.70927 |
| v29_qwencond | 0.73001 | 0.71063 |
| v29d_qwencond | 0.73112 | 0.70772 |
| **v29f (submitted)** | 0.72941 | **0.71095** |
| v29g | **0.73219** | 0.70784 |

Because each step changes one field, every delta is attributable to a single change:

| change | rows | public | private |
|---|---|---|---|
| `conditions` Mistral-7B → Qwen2.5-3B (v29 → v29_qwencond) | 314/500 | +0.00228 | +0.00136 |
| `study_type` seed-7 retrain (v29_qwencond → v29d_qwencond) | 110/500 | **+0.00111** | **−0.00291** |
| `maximum_age` extractor (v29d_qwencond → **v29f**) | 49/500 | **−0.00171** | **+0.00323** |
| `conditions` +methods prompt (v29d_qwencond → v29g) | 144/500 | +0.00107 | +0.00012 |

Three things follow. The **age extractor won the private board** — 49 changed cells, the
largest private gain of any change we made, and the only one that moved public and private
in opposite directions. The **Mistral → Qwen swap was the only unambiguous win**, gaining on
both splits. And the `study_type` seed sweep is the cautionary tale: picking the best of 10
seeds on public gained +0.0011 there and lost −0.0029 on private — it selected for the
split, not the task.

Not tested: the `+methods` conditions prompt is roughly private-neutral (+0.00012), so
combining it with the age extractor would plausibly have scored a little above v29f. That
combination was never submitted, so this is an inference from the table above, not a result.
