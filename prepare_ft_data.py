"""Prepare fine-tuning data: NXML (title+abstract+methods) -> GT eligibility text."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(__file__))
from nxml_parser import parse_nxml

TASK_XLSX = os.path.join(os.path.dirname(__file__), 'Task_1.xlsx')


def read_train_rows():
    """Return the Train sheet of Task_1.xlsx as a list of field dicts."""
    import openpyxl
    ws = openpyxl.load_workbook(TASK_XLSX)['Train']
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c else '' for c in rows[0]]
    result = []
    for row in rows[1:]:
        if not row[0]:
            continue
        d = {header[i]: (str(row[i]).strip() if row[i] is not None else '') for i in range(len(header))}
        d['pmcids'] = str(int(float(d['pmcids'])))
        result.append(d)
    return result


def build_input_text(parsed, max_chars=4000):
    """Build the input prompt for the model."""
    title = (parsed.get('title', '') or '').strip()
    abstract = (parsed.get('abstract_text', '') or '').strip()
    methods = (parsed.get('methods_text', '') or '').strip()
    elig = (parsed.get('eligibility_text', '') or '').strip()

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

    text = '\n\n'.join(parts)
    return text[:max_chars]


def main():
    rows = read_train_rows()
    examples = []
    skipped = 0
    for r in rows:
        pmcid = r['pmcids']
        parsed = parse_nxml(pmcid)
        if parsed is None:
            skipped += 1; continue
        gt_elig = (r.get('eligibility_criteria','') or '').strip()
        if not gt_elig or gt_elig == 'None' or len(gt_elig) < 30:
            skipped += 1; continue
        inp = build_input_text(parsed)
        if len(inp) < 100:
            skipped += 1; continue
        examples.append({
            'pmcid': pmcid,
            'input': inp,
            'output': gt_elig,
        })

    print(f'Built {len(examples)} examples (skipped {skipped})')
    # Stats
    import statistics
    in_lens = [len(e['input']) for e in examples]
    out_lens = [len(e['output']) for e in examples]
    print(f'Input  len: median={statistics.median(in_lens):.0f}, max={max(in_lens)}, p95={sorted(in_lens)[int(len(in_lens)*0.95)]}')
    print(f'Output len: median={statistics.median(out_lens):.0f}, max={max(out_lens)}, p95={sorted(out_lens)[int(len(out_lens)*0.95)]}')

    out_path = os.path.join(os.path.dirname(__file__), 'ft_data.jsonl')
    with open(out_path, 'w') as f:
        for e in examples:
            f.write(json.dumps(e) + '\n')
    print(f'Wrote {out_path}')

    # Train/val split (90/10)
    import random
    random.seed(42)
    random.shuffle(examples)
    n_val = max(20, len(examples) // 10)
    val = examples[:n_val]
    train = examples[n_val:]
    print(f'Split: {len(train)} train / {len(val)} val')

    with open(os.path.join(os.path.dirname(__file__), 'ft_train.jsonl'), 'w') as f:
        for e in train: f.write(json.dumps(e) + '\n')
    with open(os.path.join(os.path.dirname(__file__), 'ft_val.jsonl'), 'w') as f:
        for e in val: f.write(json.dumps(e) + '\n')


if __name__ == '__main__':
    main()
