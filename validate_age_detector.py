"""Validate an explicit-upper-bound detector against GT max_age on the train set.
Goal: decide whether gating max_age on text-supported upper bounds improves accuracy.
NOT a pipeline file — a one-off diagnostic.
"""
import re, sys, openpyxl
sys.path.insert(0, '.')
from nxml_parser import parse_nxml

# Upper-bound cue patterns. Each yields the bounding age number.
_RANGE = re.compile(r'\b(\d{1,3})\s*(?:to|[-–—]|and)\s*(\d{1,3})\s*(?:years?|yrs?\b|y\b|yo\b|years? old)', re.I)
_UPPER = re.compile(
    r'(?:up to|under|younger than|less than|no older than|aged up to|'
    r'max(?:imum)?\s*age\s*(?:of|:)?\s*|≤|<=?)\s*(\d{1,3})\s*(?:years?|yrs?\b|y\b)', re.I)
_UPPER_SUFFIX = re.compile(r'\b(\d{1,3})\s*(?:years?|yrs?)\s*(?:or younger|and younger|or less|and below|or below)', re.I)


def detect_upper_bounds(text):
    """Return set of integer ages that appear as explicit UPPER bounds in text."""
    bounds = set()
    for m in _RANGE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi > lo and hi <= 120:          # plausible upper age of a range
            bounds.add(hi)
    for m in _UPPER.finditer(text):
        v = int(m.group(1))
        if v <= 120:
            bounds.add(v)
    for m in _UPPER_SUFFIX.finditer(text):
        v = int(m.group(1))
        if v <= 120:
            bounds.add(v)
    return bounds


def article_text(parsed):
    return ' '.join([
        parsed.get('title', '') or '',
        parsed.get('abstract_text', '') or '',
        parsed.get('eligibility_text', '') or '',
        parsed.get('methods_text', '') or '',
        parsed.get('all_paragraphs_text', '') or '',
    ])


def main():
    wb = openpyxl.load_workbook('Task_1.xlsx'); ws = wb['Train']
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(c).strip() if c else '' for c in rows[0]]
    pi, ma = hdr.index('pmcids'), hdr.index('maximum_age')

    gt_num = gt_ns = 0
    # confusion: GT-number vs GT-NotSpec  x  detector-fires vs detector-silent
    tp = fp = tn = fn = 0
    exact_hits = 0
    examples_fp = []
    for r in rows[1:]:
        if not r[pi]:
            continue
        pmcid = str(int(float(r[pi])))
        gtmax = str(r[ma]).strip() if r[ma] is not None else ''
        gt_is_num = bool(re.match(r'^\d+\s*[Yy]ears?$', gtmax))
        parsed = parse_nxml(pmcid)
        if parsed is None:
            continue
        bounds = detect_upper_bounds(article_text(parsed))
        fires = len(bounds) > 0
        if gt_is_num:
            gt_num += 1
            gtn = int(re.match(r'^(\d+)', gtmax).group(1))
            if fires:
                tp += 1
                if gtn in bounds:
                    exact_hits += 1
            else:
                fn += 1
        else:
            gt_ns += 1
            if fires:
                fp += 1
                if len(examples_fp) < 8:
                    examples_fp.append((pmcid, sorted(bounds)))
            else:
                tn += 1

    print(f'Train rows scored: {gt_num + gt_ns}  (GT max: {gt_num} numbers, {gt_ns} NotSpecified)')
    print()
    print('DETECTOR vs GT max_age:')
    print(f'  GT=number  & detector fires (TP): {tp:3}   of which exact-number match: {exact_hits}')
    print(f'  GT=number  & detector silent (FN): {fn:3}')
    print(f'  GT=NotSpec & detector fires (FP): {fp:3}   <-- these would WRONGLY keep a max age')
    print(f'  GT=NotSpec & detector silent (TN): {tn:3}')
    print()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    print(f'  precision (fires => really has max): {prec:.1%}')
    print(f'  recall    (real max => fires):       {rec:.1%}')
    print(f'  specificity (NotSpec => silent):     {spec:.1%}')
    print()
    print('  sample false positives (GT=NotSpec but detector found bound):')
    for p, b in examples_fp:
        print(f'    PMC{p}: bounds={b}')


if __name__ == '__main__':
    main()
