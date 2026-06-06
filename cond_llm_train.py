"""LLM-RAG conditions extractor + leakage-free train eval.
Uses Mistral-7B (same model as v28) with PubMedBERT-retrieved few-shot examples
from train (self excluded). Scored by official evaluate.biobert_similarity.
Goal: see if LLM-extracted disease names beat the vocab baseline (test LB 0.319).
"""
import os, sys, json, re, ast, openpyxl
sys.path.insert(0, '.')
from nxml_parser import parse_nxml
from evaluate import biobert_similarity
import llm_extractor_rag_v28 as rag   # reuse _get_llm, _retrieve, _parse_json_output

_SYS = ("You are a clinical trial curator. From the article, list ONLY the medical "
        "condition(s) or disease(s) being studied — the diagnoses that define the patient "
        "cohort. Use short canonical names (e.g. 'Hepatocellular Carcinoma', 'Type 2 Diabetes "
        "Mellitus'). Do NOT include imaging methods, devices, measurements, procedures, "
        "biomarkers, or anatomy. Respond with a JSON list of strings only.")


def _article_snippet(parsed):
    title = (parsed.get('title', '') or '').strip()
    abstract = (parsed.get('abstract_text', '') or '').strip()
    kw = ' '.join(parsed.get('keywords', []) or [])
    s = f"Title: {title}\nAbstract: {abstract[:1200]}"
    if kw:
        s += f"\nKeywords: {kw[:300]}"
    return s


def extract_conditions_llm(parsed, exclude_pmcid='', k=4):
    llm = rag._get_llm()
    query = (parsed.get('title', '') or '') + ' ' + (parsed.get('abstract_text', '') or '')[:400]
    similar = rag._retrieve(query, k=k + 2, exclude_pmcid=exclude_pmcid)

    shots = ''
    added = 0
    for ex in similar:
        if added >= k:
            break
        gt_c = ex['gt'].get('conditions', '')
        try:
            lst = ast.literal_eval(gt_c) if gt_c.startswith('[') else None
        except Exception:
            lst = None
        if not lst:
            continue
        # build a compact snippet for the example from its stored article_text
        ex_art = ex.get('article_text', '')[:900]
        shots += f"[INST] {_SYS}\n\nArticle:\n{ex_art} [/INST] {json.dumps(lst, ensure_ascii=False)} </s>"
        added += 1

    prompt = shots + f"[INST] {_SYS}\n\nArticle:\n{_article_snippet(parsed)} [/INST]"
    out = llm(prompt, max_tokens=120, temperature=0.0, stop=['</s>', '[INST]'])
    raw = out['choices'][0]['text'].strip()
    parsed_out = rag._parse_json_output(raw)
    if isinstance(parsed_out, list):
        items = parsed_out
    elif isinstance(parsed_out, dict):
        items = parsed_out.get('conditions', []) or list(parsed_out.values())
    else:
        items = []
    # also try bare json list
    if not items:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                items = ast.literal_eval(m.group())
            except Exception:
                items = []
    items = [str(x).strip() for x in items if str(x).strip()][:6]
    return str(items) if items else str(['Not Specified'])


def load_train():
    wb = openpyxl.load_workbook('Task_1.xlsx'); ws = wb['Train']
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(c).strip() if c else '' for c in rows[0]]
    pi, ci = hdr.index('pmcids'), hdr.index('conditions')
    return [{'pmcid': str(int(float(r[pi]))),
             'gt': str(r[ci]).strip() if r[ci] is not None else ''}
            for r in rows[1:] if r[pi]]


def main():
    n = int(os.environ.get('N', '60'))
    data = load_train()[:n]
    total = 0.0; scored = 0; examples = []
    for d in data:
        parsed = parse_nxml(d['pmcid'])
        if parsed is None:
            continue
        pred = extract_conditions_llm(parsed, exclude_pmcid=d['pmcid'])
        s = biobert_similarity(pred, d['gt'])
        total += s; scored += 1
        examples.append((s, d['pmcid'], pred, d['gt']))
    print(f'\nLLM-RAG conditions on {scored} train rows (self-excluded): mean biobert = {total/scored:.4f}')
    print('\nsample (score | pred | gt):')
    for s, p, pr, gt in examples[:15]:
        print(f'  {s:.2f}  PRED={pr[:50]:50}  GT={gt[:50]}')


if __name__ == '__main__':
    main()
