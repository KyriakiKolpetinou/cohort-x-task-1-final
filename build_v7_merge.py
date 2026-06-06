"""
Build submission_v7.csv by merging:
  - submission_v3.csv as baseline (LB 0.690 — LLM eligibility was working)
  - safer-v6 conditions: vocab + blocklist + NER, NO noisy RAG
  - keep v3's study_type, sex, min/max age, eligibility unchanged

Strategy: only replace conditions when we have a HIGH-CONFIDENCE vocab match
(title vocab is most reliable). Otherwise keep v3's original.
"""
import os, sys, csv, ast, re
sys.path.insert(0, os.path.dirname(__file__))
os.environ['LD_LIBRARY_PATH'] = (
    '/usr/local/cuda-12.1/targets/x86_64-linux/lib:'
    + os.environ.get('LD_LIBRARY_PATH', '')
)
os.environ.setdefault('HF_HOME', '/mnt/extra_storage/kkolpetinou/torch_cache/huggingface')

from nxml_parser import parse_nxml
from extractors import _vocab_lookup
from pipeline_v6 import _EXTRA_BLOCKLIST, _is_negated

BASE = '/home/kkolpetinou/cohort-x-task-1/submission_v3.csv'
OUT = '/home/kkolpetinou/cohort-x-task-1/submission_v7.csv'


def _filter(terms, context):
    out, seen = [], set()
    for t in terms:
        tl = t.lower()
        if tl in _EXTRA_BLOCKLIST or tl in seen:
            continue
        if _is_negated(context, t):
            continue
        out.append(t)
        seen.add(tl)
    return out


def safer_conditions(parsed, v3_cond_str):
    """Return improved conditions string. Falls back to v3 if no high-conf hit."""
    title = (parsed.get('title', '') or '').strip()
    abstract = (parsed.get('abstract_text', '') or '').strip()
    keywords = parsed.get('keywords', []) or []
    kw_text = ' '.join(keywords)

    # Highest confidence: TITLE vocab match
    title_hits = _filter(_vocab_lookup(title), title)
    title_hits.sort(key=lambda t: -len(t))

    # Keywords vocab
    kw_hits = _filter(_vocab_lookup(kw_text), kw_text) if kw_text else []
    kw_hits.sort(key=lambda t: -len(t))

    # Abstract vocab
    abs_hits = _filter(_vocab_lookup(abstract), abstract)
    abs_hits.sort(key=lambda t: -len(t))

    # Parse v3's original
    try:
        v3_list = ast.literal_eval(v3_cond_str) if v3_cond_str.startswith('[') else [v3_cond_str]
        if not isinstance(v3_list, list):
            v3_list = [str(v3_list)]
        v3_list = [str(x).strip() for x in v3_list if x and str(x).strip()]
    except Exception:
        v3_list = []

    # v3 quality check: is it generic junk?
    v3_low = [c.lower() for c in v3_list]
    v3_is_junk = (
        not v3_list
        or v3_list == ['Not Specified']
        or any(j in v3_low for j in ['behavior', 'learning', 'staging',
                                       'observation', 'screening', 'detection'])
    )

    # Combine: title > keywords > abstract, cap at 2
    candidates = []
    for src in [title_hits, kw_hits, abs_hits]:
        for t in src:
            if t not in candidates:
                candidates.append(t)
    candidates = candidates[:2]

    # Decision logic:
    # - If we have a title hit ≥ 10 chars (specific): use our list (highest precision)
    # - If we have any vocab hit AND v3 was junk: use our list
    # - If we have any vocab hit AND v3 list is short (length 1) AND ours is more specific: use ours
    # - Else: keep v3
    if not candidates:
        return v3_cond_str  # nothing better, keep v3

    longest = max(len(c) for c in candidates)
    if longest >= 12 and (title_hits or any(t in candidates for t in title_hits)):
        return str(candidates)
    if v3_is_junk:
        return str(candidates)
    if v3_list and longest > max(len(c) for c in v3_list) + 4:
        return str(candidates)

    return v3_cond_str


def main():
    with open(BASE) as f:
        rows = list(csv.DictReader(f))
    print(f'loaded {len(rows)} from {BASE}')

    changed = 0
    new_rows = []
    for i, r in enumerate(rows, 1):
        pmcid = r['pmcids']
        parsed = parse_nxml(pmcid)
        if parsed is None:
            new_rows.append(r)
            continue
        new_cond = safer_conditions(parsed, r['conditions'])
        if new_cond != r['conditions']:
            changed += 1
            if changed <= 8:
                print(f'  PMC{pmcid}: {r["conditions"][:60]}  →  {new_cond[:60]}')
        r_out = dict(r)
        r_out['conditions'] = new_cond
        new_rows.append(r_out)
        if i % 100 == 0:
            print(f'  [{i}/{len(rows)}] {changed} conditions changed so far', flush=True)

    print(f'\nTotal conditions changed: {changed}/{len(rows)}')

    fieldnames = ['pmcids', 'conditions', 'study_type', 'sex',
                  'minimum_age', 'maximum_age', 'eligibility_criteria']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)
    print(f'wrote → {OUT}')


if __name__ == '__main__':
    main()
