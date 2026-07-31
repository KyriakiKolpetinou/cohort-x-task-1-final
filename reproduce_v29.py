"""
Reproduce submission_v29f.csv — Cohort X Task 1, the 2nd-place submission
(final/private LB 0.71095; public 0.72941).

ONE clean pass over the 500 test articles (Task_1.xlsx 'Test' sheet). Each of the
six output fields is produced exactly once, by its final method — no version lineage,
no intermediate CSVs:

  conditions           : Qwen2.5-3B-Instruct Q4, LLM-RAG (k=4 PubMedBERT few-shot
                         from train_index.json; "diseases only" prompt over
                         title+abstract+keywords)                        [conditions.py]
  eligibility_criteria : RAFT-tuned BART (bart_raft_v17/final), 4-beam     [this file]
  study_type           : fine-tuned PubMedBERT classifier                  [extractors]
  sex                  : rule-based                                        [extractors]
  minimum_age          : constant "18 Years"       (extraction measured worse — the
                                                    constant is right ~70% of the time)
  maximum_age          : extracted from the article's eligibility/methods text where a
                         genuine age bound fires, else "Not Specified"  [age_extractor.py]

Article text only — no external registries (ClinicalTrials.gov / AACT / NCT). CPU-capable.

MODELS — all default to paths inside this repo; override via env if stored elsewhere:
  CONDITIONS_GGUF  models/Qwen2.5-3B-Instruct-Q4_K_M.gguf      (conditions)
  BART_DIR         models/bart_raft_v17_final/final            (eligibility)
  NXML_DIR         PMC_NXML_Archives/                          (article source)
  models/study_type_classifier/                                (study_type; PubMedBERT)
PubMedBERT (retrieval + study_type base) is auto-downloaded from HuggingFace.

REPRODUCE:
  # competition-compliant pure CPU (~3 h for the 3B over 500 rows):
  CUDA_VISIBLE_DEVICES="" python reproduce_v29.py
  # dev speed (identical output): offload both models to GPU
  N_GPU_LAYERS=33 BART_DEVICE=cuda python reproduce_v29.py
OUTPUT: submission_v29f.csv  (compare to reference_outputs/submission_v29f.csv)
"""
import os, sys, csv, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration

from nxml_parser import parse_nxml
from conditions import extract_conditions_llm                    # Qwen2.5-3B LLM-RAG conditions
from study_type_and_sex import extract_sex, extract_study_type   # PubMedBERT study_type + rule sex
from age_extractor import extract_ages                           # regex age bounds (min stays constant)

TASK_XLSX = os.path.join(HERE, 'Task_1.xlsx')
BART_DIR  = os.environ.get('BART_DIR', os.path.join(HERE, 'models', 'bart_raft_v17_final', 'final'))
OUT       = os.environ.get('OUT_CSV', os.path.join(HERE, 'submission_v29f.csv'))
PREFIX    = 'Extract eligibility criteria: '
FIELDNAMES = ['pmcids', 'conditions', 'study_type', 'sex',
              'minimum_age', 'maximum_age', 'eligibility_criteria']

# Fallbacks. minimum_age is always the constant: extracting it measured WORSE than
# "18 Years" (right ~70% of the time; adult trials cite subgroup ages that mislead the
# extractor), so age_extractor sets EXTRACT_MIN=False. maximum_age falls back to
# "Not Specified" only when no genuine age bound fires — see age_extractor.py.
MIN_AGE = '18 Years'
MAX_AGE = 'Not Specified'


def read_test_pmcids():
    import openpyxl
    ws = openpyxl.load_workbook(TASK_XLSX)['Test']
    rows = list(ws.iter_rows(values_only=True))
    return [str(int(float(r[0]))) for r in rows[1:] if r[0]]   # xlsx stores ids as floats


def build_input_text(parsed, max_chars=4000):
    """Format a parsed article into the BART eligibility-generation prompt."""
    title    = (parsed.get('title', '') or '').strip()
    abstract = (parsed.get('abstract_text', '') or '').strip()
    methods  = (parsed.get('methods_text', '') or '').strip()
    elig     = (parsed.get('eligibility_text', '') or '').strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract[:1500]}")
    if elig and len(elig) > 50:
        parts.append(f"Eligibility section: {elig[:1500]}")
    if methods and not elig:
        parts.append(f"Methods: {methods[:2000]}")
    elif methods:
        parts.append(f"Methods: {methods[:1500]}")
    return '\n\n'.join(parts)[:max_chars]


def main():
    device = os.environ.get('BART_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('  (cuda requested but unavailable — using cpu)'); device = 'cpu'

    print(f'Loading RAFT BART from {BART_DIR} on {device}...', flush=True)
    tok = BartTokenizerFast.from_pretrained(BART_DIR)
    bart = BartForConditionalGeneration.from_pretrained(BART_DIR).eval().to(device)

    def eligibility(parsed):
        enc = tok(PREFIX + build_input_text(parsed), max_length=1024,
                  truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out = bart.generate(**enc, max_length=384, num_beams=4,
                                no_repeat_ngram_size=3, early_stopping=True)
        gen = tok.decode(out[0], skip_special_tokens=True).strip()
        return gen or 'Not Specified'

    pmcids = read_test_pmcids()
    print(f'Processing {len(pmcids)} test articles -> {OUT}', flush=True)
    rows = []; t0 = time.time(); parse_fail = 0
    for i, pmcid in enumerate(pmcids, 1):
        parsed = parse_nxml(pmcid)
        if parsed is None:
            parse_fail += 1
            rows.append({'pmcids': pmcid, 'conditions': str(['Not Specified']),
                         'study_type': 'INTERVENTIONAL', 'sex': 'ALL',
                         'minimum_age': MIN_AGE, 'maximum_age': MAX_AGE,
                         'eligibility_criteria': 'Not Specified'})
            continue
        min_age, max_age = extract_ages(parsed)   # min is always MIN_AGE by design
        rows.append({
            'pmcids': pmcid,
            'conditions': extract_conditions_llm(parsed, exclude_pmcid=pmcid),
            'study_type': extract_study_type(parsed),
            'sex': extract_sex(parsed),
            'minimum_age': min_age,
            'maximum_age': max_age,
            'eligibility_criteria': eligibility(parsed),
        })
        if i % 25 == 0:
            el = time.time() - t0
            print(f'  [{i}/{len(pmcids)}] {el:.0f}s, ETA {(len(pmcids)-i)/max(i/el,1e-3):.0f}s', flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {OUT} ({len(rows)} rows, {parse_fail} parse-fail defaults)')


if __name__ == '__main__':
    main()
