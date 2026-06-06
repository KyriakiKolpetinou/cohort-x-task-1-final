"""
study_type and sex fields.

extract_study_type(parsed) -> 'INTERVENTIONAL' | 'OBSERVATIONAL'
    Fine-tuned PubMedBERT classifier (models/study_type_classifier/). Falls back to a
    runtime TF-IDF+LR model (trained from Task_1.xlsx) and then a keyword rule if the
    classifier weights are absent.
extract_sex(parsed) -> 'MALE' | 'FEMALE' | 'ALL'
    High-precision disease/anatomy keyword rules; defaults to ALL when ambiguous.

Both take the dict returned by nxml_parser.parse_nxml(). Article text only.
"""
import re
import os

# ── study_type: fine-tuned PubMedBERT ────────────────────────────────────────

_bert_tokenizer = None
_bert_model = None
_BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'study_type_classifier')

# TF-IDF fallback (used if the PubMedBERT model is not available)
_study_type_clf = None
_study_type_vec = None


def _load_bert_classifier():
    global _bert_tokenizer, _bert_model
    if not os.path.exists(_BERT_MODEL_DIR):
        return False
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _bert_tokenizer = AutoTokenizer.from_pretrained(_BERT_MODEL_DIR)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(_BERT_MODEL_DIR)
        _bert_model.eval()
        return True
    except Exception:
        return False


def _train_study_type_classifier():
    """Fallback only: train a TF-IDF+LR model from Task_1.xlsx at run time."""
    global _study_type_clf, _study_type_vec
    import openpyxl
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    xlsx = os.path.join(os.path.dirname(__file__), 'Task_1.xlsx')
    if not os.path.exists(xlsx):
        return

    wb = openpyxl.load_workbook(xlsx)
    ws = wb['Train']
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c else '' for c in rows[0]]
    pmcid_col = header.index('pmcids')
    type_col  = header.index('study_type')

    texts, labels = [], []
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from nxml_parser import parse_nxml
    for row in rows[1:]:
        if not row[pmcid_col] or not row[type_col]:
            continue
        label = str(row[type_col]).strip().upper()
        if label not in ('INTERVENTIONAL', 'OBSERVATIONAL'):
            continue
        pmcid = str(int(float(row[pmcid_col])))
        d = parse_nxml(pmcid)
        if d is None:
            continue
        full = (d.get('full_body_text', '') or d.get('abstract_text', '') or '')
        text = (d.get('title', '') + ' ' + full).lower()
        text = re.sub(r'\s+', ' ', text)[:4000]
        texts.append(text)
        labels.append(label)

    if len(texts) < 10:
        return

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf.fit(X, labels)
    _study_type_vec = vec
    _study_type_clf = clf


def extract_study_type(d):
    """Return 'INTERVENTIONAL' or 'OBSERVATIONAL'.
    Priority: fine-tuned PubMedBERT -> TF-IDF+LR -> keyword rule-based.
    """
    global _bert_tokenizer, _bert_model, _study_type_clf, _study_type_vec

    if _bert_model is None:
        _load_bert_classifier()

    if _bert_model is not None:
        import torch
        full = (d.get('full_body_text', '') or d.get('abstract_text', '') or '')
        text = (d.get('title', '') + ' ' + full).lower()
        text = re.sub(r'\s+', ' ', text).strip()[:4000]
        inputs = _bert_tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            logits = _bert_model(**inputs).logits
        return _bert_model.config.id2label[logits.argmax(dim=-1).item()]

    if _study_type_clf is None:
        _train_study_type_classifier()
    if _study_type_clf is not None:
        full = (d.get('full_body_text', '') or d.get('abstract_text', '') or '')
        text = (d.get('title', '') + ' ' + full).lower()
        text = re.sub(r'\s+', ' ', text)[:4000]
        return _study_type_clf.predict(_study_type_vec.transform([text]))[0]

    return _extract_study_type_rule_based(d)


# keyword scoring used only by the rule-based fallback
_STUDY_TYPE_SCORES = [
    (r'\brandomized\b', +4), (r'\brandomised\b', +4), (r'\brandomization\b', +3),
    (r'\brandomisation\b', +3), (r'\bdouble.blind\b', +4), (r'\bplacebo.controlled\b', +4),
    (r'\bplacebo\b', +3), (r'\bcontrolled trial\b', +4), (r'\bclinical trial\b', +2),
    (r'\bphase (?:i|ii|iii|iv)\b', +3), (r'\bphase [1-4]\b', +3), (r'\btreatment arm\b', +3),
    (r'\bintervention arm\b', +3), (r'\bsingle.blind\b', +2), (r'\bcrossover\b', +2),
    (r'\binterventional\b', +2), (r'\brct\b', +2), (r'\bprotocol\b', +1),
    (r'\btreatment group\b', +1),
    (r'\bretrospective\b', -4), (r'\bobservational study\b', -4), (r'\bobservational design\b', -4),
    (r'\bcohort study\b', -3), (r'\bcase.control\b', -3), (r'\bcross.sectional\b', -2),
    (r'\bmeta.analysis\b', -3), (r'\bsystematic review\b', -2), (r'\bregistry.based\b', -3),
    (r'\bepidemiological study\b', -2), (r'\bprospective cohort\b', -2), (r'\bcase series\b', -2),
    (r'\bobservational\b', -1), (r'\bcohort\b', -1), (r'\bregistry\b', -1),
]
_STUDY_TYPE_PATTERNS = [(re.compile(p, re.IGNORECASE), s) for p, s in _STUDY_TYPE_SCORES]
_NCT_PAT = re.compile(r'\bNCT\d{8}\b')
_OBS_ABSTRACT = re.compile(
    r'\b(?:retrospective|observational\s+study|observational\s+cohort|'
    r'cross.sectional\s+study|case.control\s+study|registry.based|'
    r'prospective\s+observational|prospective\s+cohort\s+study)\b', re.IGNORECASE)
_INTERV_ABSTRACT = re.compile(
    r'\b(?:randomized?\s+controlled\s+trial|randomised?\s+controlled\s+trial|'
    r'double.blind(?:ed)?|placebo.controlled|phase\s+(?:i|ii|iii|iv)\s+(?:study|trial)|'
    r'single.blind(?:ed)?|open.label\s+(?:trial|study))\b', re.IGNORECASE)


def _extract_study_type_rule_based(d):
    abstract = d.get('abstract_text', '')
    title = d.get('title', '')
    text = title + ' ' + abstract
    score = sum(s for pat, s in _STUDY_TYPE_PATTERNS if pat.search(text))
    if _OBS_ABSTRACT.search(abstract):
        score -= 3
    if _INTERV_ABSTRACT.search(abstract):
        score += 3
    if _NCT_PAT.search(d.get('all_paragraphs_text', '') + ' ' + abstract):
        score += 1
    return 'OBSERVATIONAL' if score < 0 else 'INTERVENTIONAL'   # default INTERVENTIONAL (majority)


# ── sex ──────────────────────────────────────────────────────────────────────

_MALE_PATTERNS = [
    r'\bprostate\s+cancer\b', r'\bprostate\s+carcinoma\b', r'\bprostatectomy\b',
    r'\bprostate.specific\s+antigen\b', r'\bpsa\s+level\b',
    r'\btesticular\s+cancer\b', r'\btesticular\s+torsion\b',
    r'\bpenile\b', r'\berectile\s+dysfunction\b',
    r'\bmale[s]?\s+only\b', r'\bmen\s+only\b', r'\bonly\s+men\b', r'\bonly\s+male\b',
    r'\ball\s+(?:patients\s+)?were\s+male\b',
]
_FEMALE_PATTERNS = [
    r'\bbreast\s+cancer\s+patient', r'\bbreast\s+cancer\s+(?:cohort|study|trial|population|subject)',
    r'\bHER2.positive\b', r'\bHER2\+\b', r'\btrastuzumab\b', r'\bpertuzumab\b',
    r'\bovarian\s+cancer\b', r'\bovarian\s+carcinoma\b',
    r'\bcervical\s+cancer\b', r'\bcervical\s+carcinoma\b',
    r'\buterine\s+cancer\b', r'\bendometrial\s+cancer\b', r'\bendometrial\s+carcinoma\b',
    r'\bvulvar\s+cancer\b', r'\bvaginal\s+cancer\b',
    r'\bgynecologic(?:al)?\s+(?:cancer|malignancy|oncology)\b',
    r'\bgynaecologic(?:al)?\s+(?:cancer|malignancy|oncology)\b',
    r'\bendometriosis\b', r'\bplacenta[l]?\b', r'\bplacental\s+(?:function|imaging|MRI)\b',
    r'\bmaternal.fetal\b', r'\bpregnancy\s+(?:MRI|imaging|complications)\b',
    r'\bnipple\s+discharge\b', r'\bmammograph\b',
    r'\bpelvic\s+floor\s+(?:disorder|dysfunction|defect)\b',
    r'\bfemale[s]?\s+only\b', r'\bwomen\s+only\b', r'\bonly\s+women\b', r'\bonly\s+female\b',
    r'\ball\s+(?:patients\s+)?were\s+female\b',
]
_MALE_PAT   = [re.compile(p, re.IGNORECASE) for p in _MALE_PATTERNS]
_FEMALE_PAT = [re.compile(p, re.IGNORECASE) for p in _FEMALE_PATTERNS]


def _combined_text(d, sections):
    return ' '.join(d.get(s, '') for s in sections if d.get(s, ''))


def extract_sex(d):
    """Return 'MALE', 'FEMALE', or 'ALL'."""
    text = _combined_text(d, ('title', 'abstract_text', 'methods_text', 'eligibility_text'))
    male_hits   = sum(1 for p in _MALE_PAT   if p.search(text))
    female_hits = sum(1 for p in _FEMALE_PAT if p.search(text))
    if male_hits > 0 and female_hits == 0:
        return 'MALE'
    if female_hits > 0 and male_hits == 0:
        return 'FEMALE'
    return 'ALL'   # both or neither -> default
