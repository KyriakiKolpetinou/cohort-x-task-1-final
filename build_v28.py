"""
=== PIPELINE PROVENANCE — v28 final assembly (analogue of v17, smaller LLM) ===
Produces submission_v28.csv = the v28 base (Mistral-7B eligibility/ages, rule-based
study_type/sex, vocab conditions) with the SAME two post-steps that turned v3 into v17:

  STEP A  conditions upgrade (the v7 merge):  for each row, run safer_conditions()
          from build_v7_merge.py — a high-confidence title/keyword/abstract vocab
          lookup with blocklist + negation filter; keep the base value otherwise.
  STEP B  eligibility rewrite (the v17 BART): regenerate eligibility_criteria with
          the RAFT-tuned BART at bart_raft_v17/final, falling back to the base
          eligibility text when the generation looks degenerate.

This mirrors v17 exactly; ONLY the upstream eligibility/ages LLM differs
(Mistral-7B here vs Mistral-24B in v17). Everything else is byte-for-byte the
same code path, so any score delta is attributable to the model swap alone.

INPUT : submission_v28_base.csv   (from pipeline_v28.py)
OUTPUT: submission_v28.csv
MODELS: BART RAFT (~557MB, bart_raft_v17/final) — small, CPU-friendly.
        NXML parsed by nxml_parser (stdlib, CPU). No external/network sources.

CPU NOTE: device defaults to CPU (competition-compliant). Set BART_DEVICE=cuda to
use a GPU during dev; output is identical.

REPRODUCE (run after pipeline_v28.py has written submission_v28_base.csv):
  python build_v28.py
  # dev speed with GPU:
  BART_DEVICE=cuda python build_v28.py
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('HF_HOME', '/mnt/extra_storage/kkolpetinou/torch_cache/huggingface')

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration

from nxml_parser import parse_nxml
from prepare_ft_data import build_input_text
from build_v7_merge import safer_conditions

BART_DIR = '/mnt/extra_storage/kkolpetinou/bart_raft_v17/final'
BASE = '/home/kkolpetinou/cohort-x-task-1/submission_v28_base.csv'
OUT = '/home/kkolpetinou/cohort-x-task-1/submission_v28.csv'
PREFIX = 'Extract eligibility criteria: '
FIELDNAMES = ['pmcids', 'conditions', 'study_type', 'sex',
              'minimum_age', 'maximum_age', 'eligibility_criteria']


def main():
    device = os.environ.get('BART_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('  (cuda requested but unavailable — using cpu)')
        device = 'cpu'

    with open(BASE) as f:
        rows = list(csv.DictReader(f))
    print(f'Loaded {len(rows)} rows from {BASE}', flush=True)

    # ── STEP A: v7 conditions upgrade ──────────────────────────────────────
    cond_changed = 0
    for r in rows:
        parsed = parse_nxml(r['pmcids'])
        if parsed is None:
            continue
        new_cond = safer_conditions(parsed, r['conditions'])
        if new_cond != r['conditions']:
            cond_changed += 1
        r['conditions'] = new_cond
    print(f'STEP A: conditions upgraded on {cond_changed}/{len(rows)} rows', flush=True)

    # ── STEP B: v17 BART eligibility rewrite ───────────────────────────────
    print(f'Loading RAFT BART from {BART_DIR} on {device}...', flush=True)
    tokenizer = BartTokenizerFast.from_pretrained(BART_DIR)
    model = BartForConditionalGeneration.from_pretrained(BART_DIR).eval().to(device)

    t0 = time.time(); fallbacks = 0
    for i, r in enumerate(rows, 1):
        parsed = parse_nxml(r['pmcids'])
        if parsed is None:
            continue
        try:
            inp = PREFIX + build_input_text(parsed)
            enc = tokenizer(inp, max_length=1024, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_length=384, num_beams=4,
                                     no_repeat_ngram_size=3, early_stopping=True)
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            if not gen or len(gen.strip()) < 80 or 'Inclusion' not in gen:
                gen = r.get('eligibility_criteria', '')
                fallbacks += 1
        except Exception as e:
            print(f'  PMC{r["pmcids"]} error: {e}')
            gen = r.get('eligibility_criteria', '')
            fallbacks += 1
        r['eligibility_criteria'] = gen.strip()
        if i % 50 == 0:
            elapsed = time.time() - t0; rate = i / max(elapsed, 1)
            eta = (len(rows) - i) / max(rate, 0.001)
            print(f'  [{i}/{len(rows)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s, fallbacks={fallbacks}', flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in FIELDNAMES} for r in rows])
    print(f'\nWrote -> {OUT} (BART fallbacks: {fallbacks}/{len(rows)})')


if __name__ == '__main__':
    main()
