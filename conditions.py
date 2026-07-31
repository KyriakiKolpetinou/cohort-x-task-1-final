"""
conditions field — Qwen2.5-3B-Instruct LLM-RAG extractor.

For an article, retrieve the k most similar TRAIN articles (PubMedBERT cosine over
train_index.json), use their gold condition lists as few-shot examples, and ask
Qwen2.5-3B-Instruct (Q4 GGUF) to output ONLY the disease/condition names as a
JSON list. The article context is title + abstract + keywords. Article text only;
no external sources.

Public entry point: extract_conditions_llm(parsed, exclude_pmcid='', k=4) -> str
  (returns a Python-list string like "['Hepatocellular Carcinoma']")

Env:
  CONDITIONS_GGUF  path to the instruct GGUF to use (any chat model: Qwen2.5-3B,
                   Llama-3.2-3B, Mistral-7B, ...). Falls back to MISTRAL_GGUF, then
                   to the Qwen2.5-3B model below.
  N_GPU_LAYERS   0 = pure CPU (competition mode); set e.g. 33 to offload to GPU
  LLM_THREADS    CPU threads (default 12)

Model-agnostic: generation goes through llama_cpp create_chat_completion, so each
GGUF's own embedded chat template is applied (no hand-written [INST] tags).
"""
import json, re, os, ast
import numpy as np

MODEL_PATH    = os.environ.get('CONDITIONS_GGUF',
                os.environ.get('MISTRAL_GGUF',
                os.path.join(os.path.dirname(__file__), 'models',
                             'Qwen2.5-3B-Instruct-Q4_K_M.gguf')))
INDEX_FILE    = os.path.join(os.path.dirname(__file__), 'train_index.json')
PUBMEDBERT    = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

_llm = None
_index = None
_embeddings = None
_tokenizer = None
_bert_model = None

_SYS = ("You are a clinical trial curator. From the article, list ONLY the medical "
        "condition(s) or disease(s) being studied — the diagnoses that define the patient "
        "cohort. Use short canonical names (e.g. 'Hepatocellular Carcinoma', 'Type 2 Diabetes "
        "Mellitus'). Do NOT include imaging methods, devices, measurements, procedures, "
        "biomarkers, or anatomy. Respond with a JSON list of strings only.")


# ── model + retrieval index (all lazy-loaded) ────────────────────────────────

def _get_llm():
    global _llm
    if _llm is None:
        os.environ['LD_LIBRARY_PATH'] = (
            '/usr/local/cuda-12.1/targets/x86_64-linux/lib:'
            + os.environ.get('LD_LIBRARY_PATH', '')
        )
        from llama_cpp import Llama
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=int(os.environ.get('N_CTX', '8192')),
            n_threads=int(os.environ.get('LLM_THREADS', '12')),
            n_gpu_layers=int(os.environ.get('N_GPU_LAYERS', '0')),  # 0 = pure CPU
            verbose=False,
        )
    return _llm


def _get_index():
    global _index, _embeddings
    if _index is None:
        with open(INDEX_FILE) as f:
            data = json.load(f)
        _index = data['records']
        _embeddings = np.array(data['embeddings'], dtype=np.float32)
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        _embeddings = _embeddings / np.maximum(norms, 1e-9)   # normalise for cosine
    return _index, _embeddings


def _get_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT)
        _bert_model = AutoModel.from_pretrained(PUBMEDBERT)
        _bert_model.eval()
    return _tokenizer, _bert_model


def _embed_query(text):
    import torch
    tokenizer, model = _get_bert()
    enc = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    emb = emb.cpu().numpy()[0]
    return emb / max(np.linalg.norm(emb), 1e-9)


def _retrieve(query_text, k=3, exclude_pmcid=None):
    index, embeddings = _get_index()
    qemb = _embed_query(query_text)
    ranked = np.argsort(embeddings @ qemb)[::-1]
    results = []
    for i in ranked:
        if exclude_pmcid and index[i]['pmcid'] == str(exclude_pmcid):
            continue
        results.append(index[i])
        if len(results) == k:
            break
    return results


def _parse_json_output(raw):
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    text = re.sub(r'\s*```\s*$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── public API ───────────────────────────────────────────────────────────────

def _article_snippet(parsed):
    title = (parsed.get('title', '') or '').strip()
    abstract = (parsed.get('abstract_text', '') or '').strip()
    kw = ' '.join(parsed.get('keywords', []) or [])
    s = f"Title: {title}\nAbstract: {abstract[:1200]}"
    if kw:
        s += f"\nKeywords: {kw[:300]}"
    # Extra article context for the conditions prompt. Default '' (abstract-only) is
    # the submitted v29f setting. 'methods' appends the methods text (where the cohort
    # is described): it lifted held-out conditions 0.6525 -> 0.6605 and the PUBLIC LB
    # 0.73105 -> 0.73219, but cost private score (0.71063 -> 0.70784) — it fit the
    # public split. 'conclusion'/'both' add the discussion too (measured no better).
    extra = os.environ.get('CONDITIONS_EXTRA', '')
    if extra in ('methods', 'both'):
        m = (parsed.get('methods_text', '') or '').strip()
        if m:
            s += f"\nMethods: {m[:1000]}"
    if extra in ('conclusion', 'both'):
        concl = ''
        for sec in (parsed.get('body_sections', []) or []):
            if re.search(r'conclusion|discussion', str(sec.get('title', '')), re.I):
                concl = (sec.get('text', '') or '').strip(); break
        if concl:
            s += f"\nConclusion: {concl[:800]}"
    return s


def extract_conditions_list(parsed, exclude_pmcid='', k=4, temperature=0.0, seed=None):
    """Return the raw predicted conditions as a Python list (possibly empty).

    This is the ensemble-friendly core: no 'Not Specified' fallback, so the
    aggregation layer can decide what to do with an empty result. Uses whatever
    model MODEL_PATH / _llm currently points at. temperature>0 + varying seed
    gives the diverse samples needed for self-consistency."""
    llm = _get_llm()
    query = (parsed.get('title', '') or '') + ' ' + (parsed.get('abstract_text', '') or '')[:400]
    similar = _retrieve(query, k=k + 2, exclude_pmcid=exclude_pmcid)

    # Few-shot as chat turns: system once, then (user article -> assistant JSON) pairs.
    # llama_cpp applies the GGUF's own chat template, so this works for any instruct model.
    messages = [{'role': 'system', 'content': _SYS}]
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
        ex_art = ex.get('article_text', '')[:900]
        messages.append({'role': 'user', 'content': f"Article:\n{ex_art}"})
        messages.append({'role': 'assistant', 'content': json.dumps(lst, ensure_ascii=False)})
        added += 1

    messages.append({'role': 'user', 'content': f"Article:\n{_article_snippet(parsed)}"})
    kwargs = dict(messages=messages, max_tokens=120, temperature=temperature)
    if seed is not None:
        kwargs['seed'] = seed
    out = llm.create_chat_completion(**kwargs)
    raw = (out['choices'][0]['message']['content'] or '').strip()

    parsed_out = _parse_json_output(raw)
    if isinstance(parsed_out, list):
        items = parsed_out
    elif isinstance(parsed_out, dict):
        items = parsed_out.get('conditions', []) or list(parsed_out.values())
    else:
        items = []
    if not items:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                items = ast.literal_eval(m.group())
            except Exception:
                items = []
    items = [str(x).strip() for x in items if str(x).strip()][:6]
    return items


def extract_conditions_llm(parsed, exclude_pmcid='', k=4):
    """Return the conditions list as a Python-list string (e.g. "['Breast Cancer']")."""
    items = extract_conditions_list(parsed, exclude_pmcid=exclude_pmcid, k=k)
    return str(items) if items else str(['Not Specified'])
