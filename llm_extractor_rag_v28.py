"""
=== PIPELINE PROVENANCE — v28 RAG eligibility/ages extractor ===
Identical to llm_extractor_rag.py (the v3/v17 extractor) EXCEPT the base LLM is
swapped from Mistral-Small-24B-Instruct (Q4, ~14GB) to Mistral-7B-Instruct-v0.3
(Q4, ~4.4GB). Same Mistral-family [INST] prompt template, so this is a true
apples-to-apples model-size test against v17.

Purpose of the swap: v17's 24B model needs ~18-20GB RAM and FAILS the competition
hardware rule ("standard PC, i7, >=16GB RAM, no GPU"). Mistral-7B Q4 fits 16GB.

WHAT THIS MODULE DOES (unchanged from the original):
  - For each article: retrieve top-K most similar TRAIN articles by PubMedBERT
    (CPU) cosine similarity from train_index.json, use them as few-shot examples.
  - Prompt Mistral in [INST]...[/INST] format, parse JSON output for
    eligibility_criteria + minimum_age + maximum_age, then post-process (clean
    hedging, expand abbreviations, regex age fallbacks).

CPU NOTE: _get_llm() reads N_GPU_LAYERS from the env (default 0 = pure CPU, the
competition-compliant mode). Set N_GPU_LAYERS=33 to offload to a GPU for speed
during dev; the model OUTPUT is identical either way.

DEPENDENCIES: llama_cpp, transformers (PubMedBERT, CPU), train_index.json,
  the Mistral-7B GGUF at MODEL_PATH below.
USED BY: pipeline_v28.py
"""
import json, re, os, sys
import numpy as np

# v28: Mistral-7B-Instruct-v0.3 Q4 (16GB-RAM compliant) replaces the 24B used by v17.
MODEL_PATH = os.environ.get('MISTRAL_GGUF', '/mnt/extra_storage/kkolpetinou/mistral7b_dl/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf')
INDEX_FILE = os.path.join(os.path.dirname(__file__), 'train_index.json')
BIOBERT_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

_llm = None
_index = None
_embeddings = None
_tokenizer = None
_bert_model = None


def _get_llm():
    global _llm
    if _llm is None:
        os.environ['LD_LIBRARY_PATH'] = (
            '/usr/local/cuda-12.1/targets/x86_64-linux/lib:'
            + os.environ.get('LD_LIBRARY_PATH', '')
        )
        from llama_cpp import Llama
        # Default 0 = pure CPU (competition-compliant). Override with N_GPU_LAYERS for dev speed.
        n_gpu = int(os.environ.get('N_GPU_LAYERS', '0'))
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=16384,
            n_threads=int(os.environ.get('LLM_THREADS', '12')),
            n_gpu_layers=n_gpu,
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
        # Normalise for cosine similarity
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        _embeddings = _embeddings / np.maximum(norms, 1e-9)
    return _index, _embeddings


def _get_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
        _bert_model = AutoModel.from_pretrained(BIOBERT_MODEL)
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
    emb = emb / max(np.linalg.norm(emb), 1e-9)
    return emb


def _retrieve(query_text, k=3, exclude_pmcid=None, study_type=None):
    index, embeddings = _get_index()
    qemb = _embed_query(query_text)
    scores = embeddings @ qemb
    ranked = np.argsort(scores)[::-1]
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


def _is_looping(raw):
    words = raw.split()
    if len(words) < 20:
        return False
    from collections import Counter
    top_count = Counter(words).most_common(1)[0][1]
    return top_count / len(words) > 0.15


def _normalise_age(val, default):
    val = str(val).strip() if val else ''
    if not val:
        return default
    if re.match(r'^\d+\s+[Yy]ears?$', val):
        n = int(re.match(r'^(\d+)', val).group(1))
        return val if n > 0 else default
    # Extract first number but skip 0 (artifact of "0-80 years" ranges)
    m = re.search(r'(\d+)', val)
    if m:
        n = int(m.group(1))
        return f"{n} Years" if n > 0 else default
    return default


_HEDGING_LINE = re.compile(
    r'^\*\s*(not explicitly stated.*|not explicitly provided.*|not stated.*|'
    r'not provided.*|not mentioned.*|none explicitly stated.*|none stated.*|'
    r'no explicit.*|no eligibility.*|no inclusion.*|not available.*|'
    r'not described.*|not reported.*)$',
    re.IGNORECASE
)


def _clean_eligibility(text):
    """Remove hedging bullet points, collapse empty sections, fallback to Not Specified."""
    if not text or not text.strip():
        return 'Not Specified'
    lines = text.split('\n')
    cleaned = [l for l in lines if not _HEDGING_LINE.match(l.strip())]
    # Remove section headers left with no content
    result = re.sub(
        r'((?:Inclusion|Exclusion) Criteria:\s*\n+)(?=(?:(?:Inclusion|Exclusion) Criteria:)|\s*$)',
        '', '\n'.join(cleaned), flags=re.I
    ).strip()
    return result if result else 'Not Specified'


_abbrev_map = None

def _build_abbrev_map():
    """Build abbreviation→full-term map from train index GT eligibility + article text.
    Mines patterns: 'Full Term (ABBR)' and 'ABBR (Full Term)' common in medical writing.
    """
    global _abbrev_map
    if _abbrev_map is not None:
        return _abbrev_map

    # Curated high-value abbreviations that appear in clinical trial language
    base = {
        r'\bCF\b': 'Cystic Fibrosis',
        r'\bHCC\b': 'Hepatocellular Carcinoma',
        r'\bNSCLC\b': 'Non-Small Cell Lung Cancer',
        r'\bSCLC\b': 'Small Cell Lung Cancer',
        r'\bRA\b': 'Rheumatoid Arthritis',
        r'\bMS\b': 'Multiple Sclerosis',
        r'\bT2DM\b': 'Type 2 Diabetes Mellitus',
        r'\bT1DM\b': 'Type 1 Diabetes Mellitus',
        r'\bDM\b': 'Diabetes Mellitus',
        r'\bCOPD\b': 'Chronic Obstructive Pulmonary Disease',
        r'\bCKD\b': 'Chronic Kidney Disease',
        r'\bCRC\b': 'Colorectal Cancer',
        r'\bBC\b': 'Breast Cancer',
        r'\bPC\b': 'Prostate Cancer',
        r'\bAML\b': 'Acute Myeloid Leukemia',
        r'\bCLL\b': 'Chronic Lymphocytic Leukemia',
        r'\bNHL\b': 'Non-Hodgkin Lymphoma',
        r'\bMM\b': 'Multiple Myeloma',
        r'\bIBD\b': 'Inflammatory Bowel Disease',
        r'\bUC\b': 'Ulcerative Colitis',
        r'\bCD\b': 'Crohn\'s Disease',
        r'\bSLE\b': 'Systemic Lupus Erythematosus',
        r'\bAS\b': 'Ankylosing Spondylitis',
        r'\bPsA\b': 'Psoriatic Arthritis',
        r'\bCAD\b': 'Coronary Artery Disease',
        r'\bHF\b': 'Heart Failure',
        r'\bAF\b': 'Atrial Fibrillation',
        r'\bMI\b': 'Myocardial Infarction',
        r'\bCVD\b': 'Cardiovascular Disease',
        r'\bHTN\b': 'Hypertension',
        r'\bOA\b': 'Osteoarthritis',
        r'\bOSA\b': 'Obstructive Sleep Apnea',
        r'\bIPF\b': 'Idiopathic Pulmonary Fibrosis',
        r'\bPAH\b': 'Pulmonary Arterial Hypertension',
        r'\bNPC\b': 'Nasopharyngeal Carcinoma',
        r'\bGBM\b': 'Glioblastoma',
        r'\bGIST\b': 'Gastrointestinal Stromal Tumor',
        r'\bNET\b': 'Neuroendocrine Tumor',
        r'\bPTC\b': 'Papillary Thyroid Carcinoma',
        r'\bAD\b': 'Alzheimer\'s Disease',
        r'\bPD\b': 'Parkinson\'s Disease',
        r'\bALS\b': 'Amyotrophic Lateral Sclerosis',
        r'\bTBI\b': 'Traumatic Brain Injury',
        r'\bSCI\b': 'Spinal Cord Injury',
        r'\bHIV\b': 'HIV Infection',
        r'\bHBV\b': 'Hepatitis B Virus Infection',
        r'\bHCV\b': 'Hepatitis C Virus Infection',
        r'\bTB\b': 'Tuberculosis',
        r'\bBMI\b': 'Body Mass Index',
        r'\bECOG\b': 'ECOG Performance Status',
        r'\bPS\b': 'Performance Status',
        r'\bFEV1\b': 'Forced Expiratory Volume in 1 Second',
        r'\bFVC\b': 'Forced Vital Capacity',
        r'\bEF\b': 'Ejection Fraction',
        r'\bLVEF\b': 'Left Ventricular Ejection Fraction',
        r'\beGFR\b': 'Estimated Glomerular Filtration Rate',
        r'\bGFR\b': 'Glomerular Filtration Rate',
        r'\bALT\b': 'Alanine Aminotransferase',
        r'\bAST\b': 'Aspartate Aminotransferase',
        r'\bCreatinine\b': 'Creatinine',
    }

    # Also mine from train index: "Full Term (ABBR)" patterns in article text
    try:
        index, _ = _get_index()
        abbrev_pattern = re.compile(
            r'([A-Z][a-z][A-Za-z\s]{3,40}?)\s*\(([A-Z]{2,6})\)',
        )
        for rec in index:
            for text in [rec.get('article_text', ''), rec.get('gt', {}).get('eligibility_criteria', '')]:
                for m in abbrev_pattern.finditer(text):
                    full, abbr = m.group(1).strip(), m.group(2)
                    if 3 <= len(abbr) <= 6 and len(full) > len(abbr):
                        pattern = r'\b' + re.escape(abbr) + r'\b'
                        if pattern not in base:
                            base[pattern] = full
    except Exception:
        pass

    _abbrev_map = base
    return _abbrev_map


def _expand_abbreviations(text):
    """Expand medical abbreviations in generated eligibility text."""
    if not text:
        return text
    abbrevs = _build_abbrev_map()
    for pattern, full in abbrevs.items():
        # Only expand when abbreviation stands alone (not already inside a longer word)
        text = re.sub(pattern, full, text)
    return text


_ELIGIBILITY_KEYWORDS = re.compile(
    r'\b(inclusion criteria|exclusion criteria|eligible|eligibility|'
    r'inclusion criterion|exclusion criterion|'
    r'were included|were excluded|were enrolled|were recruited|'
    r'patients? (with|who|aged?|presenting|diagnosed|having|undergoing)|'
    r'participants? (with|who|aged?|must|were)|'
    r'age\s*[≥><=]+\s*\d|≥\s*\d+\s*years?|>\s*\d+\s*years?|'
    r'signed (informed )?consent|written consent|'
    r'contraindication|prior (treatment|surgery|diagnosis)|'
    r'no (history|evidence|prior)|without (prior|history)|'
    r'criteria (for|of) (inclusion|exclusion|enrollment)|'
    r'enrolled if|included if|excluded if)',
    re.IGNORECASE
)

_REVIEW_LITERATURE_PATTERN = re.compile(
    r'\b(PubMed|Scopus|Web of Science|EMBASE|MEDLINE|Cochrane|literature search|'
    r'systematic review|meta.analysis|search strategy|database search|'
    r'peer.reviewed|conference proceedings|published studies|included articles?|'
    r'articles? included|studies? included|search terms?|keywords?.*search|'
    r'PICOS|PICO framework|study design.*eligib|eligib.*study design|'
    r'prospective.*cohort studies.*described|clinical trials.*prospective.*retrospective cohort)\b',
    re.IGNORECASE
)


def _is_review_literature_criteria(text):
    """Returns True if the eligibility text describes study-selection for a review, not patient criteria."""
    return bool(_REVIEW_LITERATURE_PATTERN.search(text[:600]))


def _parse_criteria_items(raw):
    """Split a block of text into individual criterion strings."""
    lines = [l.strip() for l in raw.split('\n')]
    non_empty = [l for l in lines if l]

    # Detect if content is explicitly bulleted/numbered
    has_markers = sum(
        1 for l in non_empty
        if re.match(r'^(?:\d+[\.\)\-:]|\([a-zA-Z0-9]\)|[a-zA-Z][\.\)]|\*|\-|•)\s*', l)
    )
    explicitly_listed = has_markers >= max(1, len(non_empty) // 3)

    items = []
    current = ''
    for line in lines:
        if not line:
            if current:
                items.append(current.strip())
                current = ''
            continue
        if explicitly_listed and re.match(r'^(?:\d+[\.\)\-:]|\([a-zA-Z0-9]\)|[a-zA-Z][\.\)]|\*|\-|•)\s*', line):
            if current:
                items.append(current.strip())
            line = re.sub(r'^(?:\d+[\.\)\-:]\s*|\([a-zA-Z0-9]\)\s*|[a-zA-Z][\.\)]\s*|\*\s*|\-\s*|•\s*)', '', line).strip()
            current = line
        elif not explicitly_listed:
            # Prose: treat each non-empty line as its own item
            if current:
                items.append(current.strip())
            current = line
        else:
            current = (current + ' ' + line).strip() if current else line
    if current:
        items.append(current.strip())

    # For prose, also try splitting long items on sentence boundaries
    final = []
    for item in items:
        if len(item) > 250:
            # Split on "; " or ". " followed by a capital letter or ≥/>
            parts = re.split(r'(?<=\.)\s+(?=[A-Z≥>])|;\s+', item)
            final.extend(p.strip() for p in parts if len(p.strip()) > 12)
        elif len(item) > 12:
            final.append(item)
    return final[:12]  # cap at 12 items per section


def _extract_verbatim_eligibility(d):
    """
    If the NXML has an explicit eligibility/inclusion/exclusion section,
    format it directly as eligibility criteria without calling the LLM.
    Returns dict with eligibility_criteria/minimum_age/maximum_age, or None.
    """
    raw = (d.get('eligibility_text', '') or '').strip()
    if not raw or len(raw) < 150:
        return None

    # Skip review-article literature search criteria
    if _is_review_literature_criteria(raw):
        return None

    # Pre-process: strip citation brackets, expand period-less sentence boundaries
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', raw)   # remove [1], [1,2] citations
    text = re.sub(r'\[(?:[–\-,\d\s]+)\]', '', text)   # remove [1-5] style citations
    # Split sentences that run together: "sentence.NextSentence" → "sentence.\nNextSentence"
    text = re.sub(r'\.([A-Z][a-z])', r'.\n\1', text)
    # Split numbered items that run together: "1.item2.item" → "1.item\n2.item"
    text = re.sub(r'(?<=\w)(\d+\.)(?=[A-Z])', r'\n\1', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    incl_re = re.compile(r'(?:^|\n)\s*inclusion\s*criteria[:\.]?\s*\n?', re.IGNORECASE)
    excl_re = re.compile(r'(?:^|\n)\s*exclusion\s*criteria[:\.]?\s*\n?', re.IGNORECASE)

    incl_m = incl_re.search(text)
    excl_m = excl_re.search(text)

    if incl_m or excl_m:
        if incl_m and excl_m:
            if incl_m.start() < excl_m.start():
                incl_raw = text[incl_m.end():excl_m.start()].strip()
                excl_raw = text[excl_m.end():].strip()
            else:
                excl_raw = text[excl_m.end():incl_m.start()].strip()
                incl_raw = text[incl_m.end():].strip()
        elif incl_m:
            incl_raw = text[incl_m.end():].strip()
            excl_raw = ''
        else:
            # Only exclusion header — text before it is inclusion criteria
            incl_raw = text[:excl_m.start()].strip()
            excl_raw = text[excl_m.end():].strip()
    else:
        # No headers — treat entire text as inclusion criteria
        incl_raw = text
        excl_raw = ''

    incl_items = _parse_criteria_items(incl_raw)
    excl_items = _parse_criteria_items(excl_raw)

    # If parsing produced no items, fall back to sentence splitting
    if not incl_items and incl_raw:
        sentences = re.split(r'(?<=[.;])\s+', incl_raw)
        incl_items = [s.strip() for s in sentences if len(s.strip()) > 20][:8]

    if not incl_items:
        return None

    parts = []
    if incl_items:
        parts.append('Inclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in incl_items))
    if excl_items:
        parts.append('Exclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in excl_items))

    eligibility = '\n\n'.join(parts)
    eligibility = _clean_eligibility(eligibility)
    eligibility = _expand_abbreviations(eligibility)

    if not eligibility or eligibility == 'Not Specified':
        return None

    # Extract ages from eligibility section only (not full article — avoids demographic age ranges)
    elig_only = {'eligibility_text': raw, 'methods_text': '', 'abstract_text': ''}
    min_age, max_age = _extract_age_from_article_text(elig_only)

    if not min_age:
        min_age = _infer_min_age_from_text(eligibility, raw)
        if min_age == 'Not Specified':
            min_age = '18 Years'
    if not max_age:
        max_age = 'Not Specified'

    return {
        'minimum_age': min_age,
        'maximum_age': max_age,
        'eligibility_criteria': eligibility,
    }


def _extract_verbatim_candidates(d):
    """Find sentences from the article that look like eligibility criteria."""
    import re as _re
    candidates = []
    sources = [
        d.get('eligibility_text', '') or '',
        d.get('methods_text', '') or '',
        d.get('abstract_text', '') or '',
    ]
    seen = set()
    for source in sources:
        # Split into sentences
        sentences = _re.split(r'(?<=[.;])\s+|\n+', source)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
            if sent in seen:
                continue
            if _ELIGIBILITY_KEYWORDS.search(sent):
                candidates.append(sent)
                seen.add(sent)
    return candidates[:20]  # cap at 20 candidates


def _build_article_text(d, verbatim_candidates=None, conditions=None, study_type=None):
    parts = []
    title = d.get('title', '').strip()
    if title:
        parts.append(f"Title: {title}")
    abstract = d.get('abstract_text', '').strip()
    if abstract:
        parts.append(f"Abstract:\n{abstract[:2000]}")
    eligib = d.get('eligibility_text', '').strip()
    if eligib:
        parts.append(f"Eligibility section:\n{eligib[:3000]}")
    methods = d.get('methods_text', '').strip()
    if methods:
        parts.append(f"Methods/Patients section:\n{methods[:8000]}")
    if not eligib and not methods:
        body = d.get('full_body_text', '').strip()
        if body:
            parts.append(f"Article body:\n{body[:8000]}")
    if verbatim_candidates:
        cand_text = '\n'.join(f'- {s}' for s in verbatim_candidates)
        parts.append(
            f"Key sentences extracted verbatim from article "
            f"(use these phrases directly in your output):\n{cand_text}"
        )
    return '\n\n'.join(parts)


_SYSTEM_PROMPT = """\
You are a clinical trial data extractor. Your task is to write the eligibility criteria for a study exactly as they would appear in a ClinicalTrials.gov entry, as curated by a medical practitioner.

Use the language and style of a medical practitioner writing for ClinicalTrials.gov:
- Each criterion is a short, precise bullet point starting with "* "
- Use clinical/protocol language: "Diagnosis of X", "History of X", "Age ≥ Y years", "Ability to provide written informed consent", "No prior X", "Patients with X"
- ALWAYS write medical conditions and diseases in full — never use abbreviations (write "Cystic Fibrosis" not "CF", "Hepatocellular Carcinoma" not "HCC", "Non-Small Cell Lung Cancer" not "NSCLC", "Rheumatoid Arthritis" not "RA", "Multiple Sclerosis" not "MS")
- Extract the ACTUAL criteria used in THIS study from the article — not generic criteria for the condition. The criteria may be clinical (diagnosis, age, labs) OR technical/procedural (imaging quality, timing windows, data completeness, institutional criteria)
- If criteria are implicit in the methods, convert them to protocol-style bullets

Participants may be patients, healthy volunteers, clinicians, radiologists, or other healthcare workers.

ALWAYS produce meaningful eligibility criteria. Never write "not explicitly stated".

Return ONLY a JSON object:
- "minimum_age": minimum participant age as "X Years" ONLY if explicitly stated. If not stated, return exactly "".
- "maximum_age": maximum participant age as "X Years" ONLY if explicitly stated. If not stated, return exactly "".
- "eligibility_criteria": formatted exactly as:
  Inclusion Criteria:

  * criterion 1
  * criterion 2

  Exclusion Criteria:

  * criterion 1

  Always produce at least 2-3 inclusion criteria."""


def _extract_age_from_article_text(d):
    """
    Extract min/max age from NXML text via regex.
    Returns (min_age_str, max_age_str) where each is "N Years" or ''.

    Strategy: search eligibility_text first (strict eligibility patterns only),
    then fall back to methods/abstract with the same strict patterns.
    Does NOT use range patterns on methods/abstract to avoid grabbing
    patient demographic age ranges (e.g. "patients aged 45-70 years").
    """
    elig_text = d.get('eligibility_text', '') or ''
    methods_text = d.get('methods_text', '') or ''
    abstract_text = d.get('abstract_text', '') or ''

    min_age = ''
    max_age = ''

    # Strict eligibility-context min age patterns
    min_patterns = [
        r'age\s*[≥>]\s*=?\s*(\d+)\s*years?',
        r'(\d+)\s*years?\s+(?:of\s+age\s+)?(?:or\s+)?(?:older|above|over)',
        r'at\s+least\s+(\d+)\s*years?',
        r'minimum\s+age\s+(?:of\s+)?(\d+)',
        r'aged?\s+(\d+)\s+(?:years?\s+)?(?:or\s+)?(?:older|above|over)',
        r'≥\s*(\d+)\s*years?',
        r'>\s*(\d+)\s*years?',
    ]
    # Strict eligibility-context max age patterns — explicit upper bound language only
    max_patterns = [
        r'age\s*[≤<]\s*=?\s*(\d+)\s*years?',
        r'(\d+)\s*years?\s+(?:of\s+age\s+)?(?:or\s+)?(?:younger|below|under)',
        r'maximum\s+age\s+(?:of\s+)?(\d+)',
        r'≤\s*(\d+)\s*years?',
        r'no\s+(?:more\s+)?(?:older\s+)?than\s+(\d+)\s*years?',
        r'not\s+(?:more\s+than|older\s+than|exceed(?:ing)?)\s+(\d+)\s*years?',
        r'age\s+limit\s+(?:of\s+)?(\d+)',
        r'aged?\s+(?:between\s+\d+\s+and\s+)?(\d+)\s*years?\s+(?:or\s+)?(?:younger|below|under)',
        r'up\s+to\s+(?:and\s+including\s+)?(\d+)\s*years?\s+(?:of\s+age)?',
    ]

    def _search_patterns(text, patterns):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 120:
                    return f"{n} Years"
        return ''

    # Search eligibility section first, then methods, then abstract
    for source in [elig_text, methods_text, abstract_text]:
        if source and not min_age:
            min_age = _search_patterns(source, min_patterns)
        if source and not max_age:
            max_age = _search_patterns(source, max_patterns)
        if min_age and max_age:
            break

    # Range pattern only on eligibility_text — avoids demographic ranges in methods/abstract
    if elig_text and (not min_age or not max_age):
        range_pat = re.search(
            r'(?:age(?:d)?\s+)?(?:between\s+)?(\d+)\s*[-–]\s*(\d+)\s*years?',
            elig_text, re.IGNORECASE
        )
        if range_pat:
            lo, hi = int(range_pat.group(1)), int(range_pat.group(2))
            if 1 <= lo <= 120 and 1 <= hi <= 120 and lo < hi:
                if not min_age:
                    min_age = f"{lo} Years"
                if not max_age:
                    max_age = f"{hi} Years"

    return min_age, max_age


def _infer_min_age_from_text(eligibility_text, article_text):
    """If LLM didn't extract a min age but text clearly describes adults, return '18 Years'."""
    combined = (eligibility_text + ' ' + article_text).lower()
    adult_patterns = [
        r'\badults?\b',
        r'age\s*[≥>=]+\s*18',
        r'18\s+years?\s+or\s+older',
        r'≥\s*18\s+years?',
        r'at\s+least\s+18\s+years?',
        r'minimum\s+age\s+of\s+18',
        r'aged?\s+18\s+(years?\s+)?or\s+(over|above|older)',
    ]
    for pat in adult_patterns:
        if re.search(pat, combined):
            return '18 Years'
    return 'Not Specified'


def extract_eligibility_with_llm(d, conditions=None, study_type=None):
    """RAG-augmented eligibility extraction using Mistral-24B with few-shot retrieval."""
    llm = _get_llm()

    query_text = d.get('title', '') + ' ' + d.get('abstract_text', '')[:400]
    similar = _retrieve(query_text, k=5, exclude_pmcid=d.get('pmcid', ''), study_type=study_type)

    verbatim = _extract_verbatim_candidates(d)
    article_text = _build_article_text(d, verbatim_candidates=verbatim if verbatim else None,
                                       conditions=conditions, study_type=study_type)

    # Build Mistral [INST] prompt with retrieved examples
    shots = ''
    shots_added = 0
    for ex in similar:
        if shots_added >= 3:
            break
        gt = ex['gt']
        gt_elig = gt.get('eligibility_criteria', '').strip()
        gt_min = gt.get('minimum_age', '').strip()
        gt_max = gt.get('maximum_age', '').strip()
        # Only use examples with substantive eligibility text
        if not gt_elig or gt_elig == 'None' or len(gt_elig) < 80:
            continue
        ex_out = json.dumps({
            'minimum_age': gt_min if gt_min and gt_min != 'None' else '',
            'maximum_age': gt_max if gt_max and gt_max != 'None' else '',
            'eligibility_criteria': gt_elig,
        }, ensure_ascii=False)
        shots += f"[INST] {_SYSTEM_PROMPT}\n\nArticle:\n{ex['article_text']} [/INST] {ex_out} </s>"
        shots_added += 1

    prompt = shots + f"[INST] {_SYSTEM_PROMPT}\n\nArticle:\n{article_text} [/INST]"

    out = llm(prompt, max_tokens=2500, temperature=0.0, stop=['</s>', '[INST]'])
    raw = out['choices'][0]['text'].strip()

    if _is_looping(raw):
        return _defaults()

    result = _parse_json_output(raw)

    min_age = _normalise_age(result.get('minimum_age', ''), default='')
    max_age = _normalise_age(result.get('maximum_age', ''), default='')

    raw_elig = result.get('eligibility_criteria', '')
    if isinstance(raw_elig, dict):
        parts = []
        incl = raw_elig.get('Inclusion Criteria', raw_elig.get('inclusion_criteria', []))
        excl = raw_elig.get('Exclusion Criteria', raw_elig.get('exclusion_criteria', []))
        if incl:
            items = incl if isinstance(incl, list) else [str(incl)]
            parts.append('Inclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in items if i))
        if excl:
            items = excl if isinstance(excl, list) else [str(excl)]
            parts.append('Exclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in items if i))
        eligibility = '\n\n'.join(parts) if parts else str(raw_elig).strip()
    else:
        eligibility = str(raw_elig).strip()

    eligibility = _clean_eligibility(eligibility)
    eligibility = _expand_abbreviations(eligibility)

    # Post-process ages: try regex on article text, then apply fallbacks
    regex_min, regex_max = _extract_age_from_article_text(d)

    if not min_age:
        min_age = regex_min or _infer_min_age_from_text(eligibility, article_text)
        if not min_age or min_age == 'Not Specified':
            min_age = '18 Years'

    if not max_age:
        max_age = regex_max or 'Not Specified'

    return {
        'minimum_age': min_age,
        'maximum_age': max_age,
        'eligibility_criteria': eligibility,
    }


def _defaults():
    return {
        'minimum_age': 'Not Specified',
        'maximum_age': 'Not Specified',
        'eligibility_criteria': 'Not Specified',
    }
