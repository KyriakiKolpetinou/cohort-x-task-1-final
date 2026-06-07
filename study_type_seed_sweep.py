"""
Seed sweep for the study_type PubMedBERT classifier.

The original classifier weights were lost; a single retrain diverges from the v29
study_type column (which scored ~0.865 on the LB). This trains the SAME classifier
under several seeds and, for each, measures agreement with the original v29 study_type
(reference_outputs/submission_v29.csv) over the 500 test rows — our local proxy for
"closest to the original score", with no leaderboard guessing.

Keeps the best-agreeing model at models/study_type_classifier/ and writes
submission_v29d.csv (= reproduced columns + best-seed study_type).
"""
import os, re, sys, csv, json, shutil
os.environ['LD_LIBRARY_PATH'] = ('/usr/local/cuda-12.1/targets/x86_64-linux/lib:'
                                 + os.environ.get('LD_LIBRARY_PATH', ''))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)
from bs4 import BeautifulSoup
from nxml_parser import parse_nxml

BERT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
NXML_DIR = os.path.join(HERE, 'PMC_NXML_Archives')
OUT_MODEL = os.path.join(HERE, 'models', 'study_type_classifier')
REF = os.path.join(HERE, 'reference_outputs', 'submission_v29.csv')
LABELS = ['INTERVENTIONAL', 'OBSERVATIONAL']; L2I = {l: i for i, l in enumerate(LABELS)}
SEEDS = [42, 1337, 2024, 0, 7, 13, 123, 777, 2025, 99]
FIELDS = ['pmcids', 'conditions', 'study_type', 'sex', 'minimum_age', 'maximum_age', 'eligibility_criteria']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def soup_text(pmcid):                      # training representation (matches train_study_type_classifier)
    p = os.path.join(NXML_DIR, f'PMC{pmcid}.nxml')
    if not os.path.exists(p): return ''
    try:
        with open(p) as f: return BeautifulSoup(f, 'lxml-xml').get_text(' ', strip=True)[:4000]
    except Exception: return ''


def infer_text(parsed):                    # inference representation (matches extract_study_type)
    full = (parsed.get('full_body_text', '') or parsed.get('abstract_text', '') or '')
    t = (parsed.get('title', '') + ' ' + full).lower()
    return re.sub(r'\s+', ' ', t).strip()[:4000]


class DS(Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, y
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}; d['labels'] = torch.tensor(self.y[i]); return d
    def __len__(self): return len(self.y)


def main():
    tok = AutoTokenizer.from_pretrained(BERT)

    # ---- train data (once) ----
    gt = pd.read_excel(os.path.join(HERE, 'Task_1.xlsx'))
    tr_texts, tr_y = [], []
    for _, r in gt.iterrows():
        t = soup_text(str(r['pmcids']))
        st = str(r['study_type']).strip().upper()
        if t and st in L2I:
            tr_texts.append(t.lower()); tr_y.append(L2I[st])
    enc = tok(tr_texts, truncation=True, padding=True, max_length=512)
    print(f'train: {len(tr_y)} (INT {tr_y.count(0)}, OBS {tr_y.count(1)})', flush=True)

    # ---- test inference texts + reference study_type (once) ----
    sub_rows = list(csv.DictReader(open(REF)))
    test_ids = [r['pmcids'] for r in sub_rows]
    ref_st = {r['pmcids']: r['study_type'].strip() for r in sub_rows}
    test_texts = [infer_text(parse_nxml(i) or {}) for i in test_ids]
    test_enc = tok(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    print(f'test: {len(test_ids)} | reference INT={list(ref_st.values()).count("INTERVENTIONAL")}', flush=True)

    best = {'agree': -1, 'seed': None, 'preds': None}
    results = []
    for seed in SEEDS:
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT, num_labels=2, id2label={i: l for l, i in L2I.items()}, label2id=L2I)
        args = TrainingArguments(output_dir=f'/tmp/styp_{seed}', num_train_epochs=5,
                                 per_device_train_batch_size=8, learning_rate=2e-5,
                                 fp16=(DEV == 'cuda'), save_strategy='no', report_to='none',
                                 logging_strategy='no', seed=seed)
        Trainer(model=model, args=args, train_dataset=DS(enc, tr_y)).train()

        # predict test in the inference representation, batched
        model.eval().to(DEV)
        preds = []
        with torch.no_grad():
            for s in range(0, len(test_ids), 32):
                batch = {k: v[s:s+32].to(DEV) for k, v in test_enc.items()}
                logits = model(**batch).logits
                preds += [LABELS[j] for j in logits.argmax(-1).cpu().tolist()]
        agree = sum(preds[k] == ref_st[test_ids[k]] for k in range(len(test_ids)))
        int_frac = preds.count('INTERVENTIONAL')
        results.append((seed, agree, int_frac))
        print(f'seed {seed:5}: agree {agree}/{len(test_ids)} = {agree/len(test_ids)*100:.1f}%  | INT={int_frac}', flush=True)
        if agree > best['agree']:
            best.update(agree=agree, seed=seed, preds=list(preds))
            shutil.rmtree(OUT_MODEL, ignore_errors=True); os.makedirs(OUT_MODEL, exist_ok=True)
            model.save_pretrained(OUT_MODEL); tok.save_pretrained(OUT_MODEL)
        del model; torch.cuda.empty_cache()

    print('\n=== sweep results (sorted) ===')
    for seed, agree, intf in sorted(results, key=lambda x: -x[1]):
        print(f'  seed {seed:5}: {agree/len(test_ids)*100:5.1f}% agreement, INT={intf}')
    print(f'\nBEST seed {best["seed"]}: {best["agree"]}/{len(test_ids)} = {best["agree"]/len(test_ids)*100:.1f}% (saved to models/study_type_classifier)')

    # build submission with best-seed study_type spliced onto the reproduced columns
    by_id = {r['pmcids']: dict(r) for r in sub_rows}
    for k, i in enumerate(test_ids):
        by_id[i]['study_type'] = best['preds'][k]
    with open(os.path.join(HERE, 'submission_v29d.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        w.writerows([{c: by_id[i][c] for c in FIELDS} for i in test_ids])
    print('wrote submission_v29d.csv')


if __name__ == '__main__':
    main()
