"""Simulate LLM-free age policies against GT on train, scored with the OFFICIAL
number_similarity (Jaccard on extracted numbers). Pick the best policy to apply
to v28 -> v28b. One-off diagnostic, not a pipeline file.
"""
import re, sys, openpyxl
sys.path.insert(0, '.')
from nxml_parser import parse_nxml
from evaluate import number_similarity
from validate_age_detector import detect_upper_bounds, article_text

# lower-bound detector (mirror of upper, for min_age policies)
_RANGE = re.compile(r'\b(\d{1,3})\s*(?:to|[-–—]|and)\s*(\d{1,3})\s*(?:years?|yrs?\b|y\b)', re.I)
_LOWER = re.compile(
    r'(?:over|older than|at least|aged?\s*(?:over|from)?|from|min(?:imum)?\s*age\s*(?:of|:)?\s*|≥|>=?)\s*'
    r'(\d{1,3})\s*(?:years?|yrs?\b|y\b)', re.I)


def detect_lower_bounds(text):
    bounds = set()
    for m in _RANGE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi > lo and lo <= 120:
            bounds.add(lo)
    for m in _LOWER.finditer(text):
        v = int(m.group(1))
        if 0 < v <= 120:
            bounds.add(v)
    return bounds


def load_train():
    wb = openpyxl.load_workbook('Task_1.xlsx'); ws = wb['Train']
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(c).strip() if c else '' for c in rows[0]]
    pi, mi, ma = hdr.index('pmcids'), hdr.index('minimum_age'), hdr.index('maximum_age')
    out = []
    for r in rows[1:]:
        if not r[pi]:
            continue
        out.append({
            'pmcid': str(int(float(r[pi]))),
            'gt_min': str(r[mi]).strip() if r[mi] is not None else '',
            'gt_max': str(r[ma]).strip() if r[ma] is not None else '',
        })
    return out


def main():
    data = load_train()
    # precompute detected bounds per article
    for d in data:
        parsed = parse_nxml(d['pmcid'])
        txt = article_text(parsed) if parsed else ''
        d['ub'] = detect_upper_bounds(txt)
        d['lb'] = detect_lower_bounds(txt)
    n = len(data)

    def score(field, fn):
        return sum(number_similarity(fn(d), d[f'gt_{field}']) for d in data) / n

    # ── max_age policies ──────────────────────────────────────────────────
    print(f'Train rows: {n}')
    print('\n=== maximum_age policies (avg number_similarity) ===')
    pol_max = {
        'always Not Specified':      lambda d: 'Not Specified',
        'range-only (Y of X-Y)':     lambda d: f'{min(d["ub"])} Years' if len(d['ub']) == 1 else 'Not Specified',
        'min detected bound':        lambda d: f'{min(d["ub"])} Years' if d['ub'] else 'Not Specified',
        'max detected bound':        lambda d: f'{max(d["ub"])} Years' if d['ub'] else 'Not Specified',
        'unique bound only':         lambda d: f'{list(d["ub"])[0]} Years' if len(d['ub']) == 1 else 'Not Specified',
    }
    for name, fn in pol_max.items():
        print(f'  {name:28} {score("max", fn):.4f}')

    # ── min_age policies ────────────────────────────────────────────────────
    print('\n=== minimum_age policies (avg number_similarity) ===')
    pol_min = {
        'always 18 Years':           lambda d: '18 Years',
        'always Not Specified':      lambda d: 'Not Specified',
        'lower bound else 18':       lambda d: f'{min(d["lb"])} Years' if d['lb'] else '18 Years',
        'unique lb else 18':         lambda d: f'{list(d["lb"])[0]} Years' if len(d['lb']) == 1 else '18 Years',
    }
    for name, fn in pol_min.items():
        print(f'  {name:28} {score("min", fn):.4f}')

    # ── reference: what do the GT distributions cap us at ──
    print('\n=== reference ===')
    gtmax_ns = sum(1 for d in data if not re.match(r'^\d', d['gt_max']))
    gtmin_18 = sum(1 for d in data if d['gt_min'] == '18 Years')
    print(f'  GT max NotSpecified: {gtmax_ns}/{n} = {gtmax_ns/n:.3f}  (== "always NotSpec" ceiling)')
    print(f'  GT min == 18 Years : {gtmin_18}/{n} = {gtmin_18/n:.3f}  (== "always 18" ceiling)')


if __name__ == '__main__':
    main()
