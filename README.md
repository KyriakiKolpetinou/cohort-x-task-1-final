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
PY=/home/kkolpetinou/miniconda3/bin/python      # has llama-cpp-python; the default `python` does not

# 1) get the two trained models (GitHub Release 'weights-v29')
BASE=https://github.com/KyriakiKolpetinou/cohort-x-task-1-final/releases/download/weights-v29
curl -L $BASE/study_type_classifier.tar.gz | tar -xz -C models      # -> models/study_type_classifier/
curl -L $BASE/bart_raft_v17_final.tar.gz | tar -xz && export BART_DIR="$PWD/final"
# Qwen GGUF: download from HF (bartowski/Qwen2.5-3B-Instruct-GGUF), file
# Qwen2.5-3B-Instruct-Q4_K_M.gguf, and place at models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
# (or set $CONDITIONS_GGUF to point at it elsewhere)

# 2) run  (GPU. For competition-compliant pure CPU: CUDA_VISIBLE_DEVICES="" and drop N_GPU_LAYERS, ~3h)
N_GPU_LAYERS=33 BART_DEVICE=cuda $PY reproduce_v29.py               # -> submission_v29f.csv
```

Check it matches the submitted file:
`diff <(sort submission_v29f.csv) <(sort reference_outputs/submission_v29f.csv)`

Deps: `llama-cpp-python==0.3.20, transformers, torch, scikit-learn, openpyxl, numpy, beautifulsoup4, lxml`

## Files

- **Inference (all you run):** `reproduce_v29.py` (driver) + `nxml_parser.py`, `conditions.py`, `study_type_and_sex.py`, `age_extractor.py`
- **Data:** `Task_1.xlsx`, `PMC_NXML_Archives/`, `train_index.json`
- **Submitted file:** `reference_outputs/submission_v29f.csv` (private 0.71095, 2nd place)
- **Model-training scripts:** see *Rebuilding the models* below

## Rebuilding the models (optional — only if not using the Release weights)

```bash
$PY build_train_index.py             # -> train_index.json  (PubMedBERT index of train articles)
$PY train_study_type_classifier.py   # -> models/study_type_classifier/  (SEED=7, the 0.72864 model)
# eligibility BART:  facebook/bart-base --train_bart_eligibility.py(SFT)--> then RAFT
#                    (raft_step1..4_*.py) --> bart_raft_v17/final
```

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

## Public vs private

Public and private rank our four submissions in **exactly inverted** order:

| submission | public | private |
|---|---|---|
| v29_qwencond | 0.73001 | 0.71063 |
| v29d_qwencond | 0.73112 | 0.70772 |
| **v29f (submitted)** | 0.72941 | **0.71095** |
| v29g | **0.73219** | 0.70784 |

Each submission differs from the previous one in exactly one field, so the deltas are
attributable:

| change | rows | public | private |
|---|---|---|---|
| `study_type` seed-7 retrain (v29_qwencond → v29d) | 110/500 | **+0.00111** | **−0.00291** |
| `maximum_age` extractor (v29d → **v29f**) | 49/500 | **−0.00171** | **+0.00323** |
| `conditions` +methods prompt (v29d → v29g) | 144/500 | +0.00107 | +0.00012 |

Two things follow. The **age extractor is what won the private board** — 49 changed cells,
the largest private gain of any change we made, and the only one that moved public and
private in opposite directions. And the `study_type` seed sweep is the cautionary tale:
picking the best of 10 seeds on public gained +0.0011 there and lost −0.0029 on private,
i.e. it selected for the split, not the task.

Not tested: the `+methods` conditions prompt is roughly private-neutral (+0.00012), so
combining it with the age extractor would plausibly have scored a little above v29f. That
combination was never submitted, so this is an inference from the table above, not a result.
