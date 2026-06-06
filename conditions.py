"""
conditions field — Mistral-7B LLM-RAG extractor.

For an article, retrieve the k most similar TRAIN articles (PubMedBERT cosine over
train_index.json), use their gold condition lists as few-shot examples, and ask
Mistral-7B-Instruct-v0.3 (Q4 GGUF) to output ONLY the disease/condition names as a
JSON list. Article text only; no external sources.

Public entry point: extract_conditions_llm(parsed, exclude_pmcid='', k=4) -> str
  (returns a Python-list string like "['Hepatocellular Carcinoma']")

Env:
  MISTRAL_GGUF   path to the Mistral GGUF (default below)
  N_GPU_LAYERS   0 = pure CPU (competition mode); set e.g. 33 to offload to GPU
  LLM_THREADS    CPU threads (default 12)
"""
import json, re, os, ast
import numpy as np

MODEL_PATH    = os.environ.get('MISTRAL_GGUF', '/mnt/extra_storage/kkolpetinou/mistral7b_dl/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf')
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
            n_ctx=16384,
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
    return s


def extract_conditions_llm(parsed, exclude_pmcid='', k=4):
    """Return the conditions list as a Python-list string (e.g. "['Breast Cancer']")."""
    llm = _get_llm()
    query = (parsed.get('title', '') or '') + ' ' + (parsed.get('abstract_text', '') or '')[:400]
    similar = _retrieve(query, k=k + 2, exclude_pmcid=exclude_pmcid)

    shots = ''; added = 0
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
        shots += f"[INST] {_SYS}\n\nArticle:\n{ex_art} [/INST] {json.dumps(lst, ensure_ascii=False)} </s>"
        added += 1

    prompt = shots + f"[INST] {_SYS}\n\nArticle:\n{_article_snippet(parsed)} [/INST]"
    out = llm(prompt, max_tokens=120, temperature=0.0, stop=['</s>', '[INST]'])
    raw = out['choices'][0]['text'].strip()

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
    return str(items) if items else str(['Not Specified'])
