"""Generate submission_v17.csv using RAFT-tuned BART (v17)."""
import os, sys, csv, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('HF_HOME', '/mnt/extra_storage/kkolpetinou/torch_cache/huggingface')

from transformers import BartTokenizerFast, BartForConditionalGeneration
from nxml_parser import parse_nxml
from prepare_ft_data import build_input_text

MODEL_DIR = '/mnt/extra_storage/kkolpetinou/bart_raft_v17/final'
BASE = '/home/kkolpetinou/cohort-x-task-1/submission_v7.csv'
OUT = '/home/kkolpetinou/cohort-x-task-1/submission_v17.csv'
PREFIX = 'Extract eligibility criteria: '


def main():
    print(f'Loading v17 RAFT BART from {MODEL_DIR}...', flush=True)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).eval().to('cuda')

    with open(BASE) as f:
        rows = list(csv.DictReader(f))
    print(f'Loaded {len(rows)} from {BASE}', flush=True)
    out_rows = []; t0 = time.time(); fallbacks = 0

    for i, r in enumerate(rows, 1):
        parsed = parse_nxml(r['pmcids'])
        if parsed is None:
            out_rows.append(r); continue
        try:
            inp = PREFIX + build_input_text(parsed)
            enc = tokenizer(inp, max_length=1024, truncation=True, return_tensors='pt').to('cuda')
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
        out = dict(r); out['eligibility_criteria'] = gen.strip()
        out_rows.append(out)
        if i % 50 == 0:
            elapsed = time.time() - t0; rate = i/max(elapsed, 1)
            eta = (len(rows)-i)/max(rate, 0.001)
            print(f'  [{i}/{len(rows)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s, fallbacks={fallbacks}', flush=True)

    fieldnames = ['pmcids','conditions','study_type','sex','minimum_age','maximum_age','eligibility_criteria']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(out_rows)
    print(f'\nWrote → {OUT} (fallbacks: {fallbacks}/{len(rows)})')


if __name__ == '__main__':
    main()
