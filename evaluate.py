"""
Evaluation using the actual CohortX metrics:
- Number Similarity (Jaccard on extracted numbers) for ages
- BioBERT cosine similarity for conditions and study_type
- FM3S for eligibility_criteria
"""
import re
import sys
import os
import json
import numpy as np

# ── Number Similarity ────────────────────────────────────────────────────────

def extract_numbers(text):
    if not text or text in ('None', 'Not Specified', ''):
        return set()
    nums = set()
    # Digit-based
    for m in re.findall(r'\d+(?:\.\d+)?', str(text)):
        nums.add(float(m))
    # Word-based
    try:
        import spacy
        from word2number import w2n
        nlp = _get_spacy()
        doc = nlp(str(text))
        for token in doc:
            try:
                nums.add(float(w2n.word_to_num(token.text)))
            except Exception:
                pass
    except Exception:
        pass
    return nums


_spacy_nlp = None
def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load('en_core_web_sm')
    return _spacy_nlp


def number_similarity(pred, gt):
    p = extract_numbers(pred)
    g = extract_numbers(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    union = len(p | g)
    return inter / union if union > 0 else 0.0


# ── BioBERT Semantic Similarity ──────────────────────────────────────────────

_biobert_model = None
_biobert_tokenizer = None

def _get_biobert():
    global _biobert_model, _biobert_tokenizer
    if _biobert_model is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
        _biobert_tokenizer = AutoTokenizer.from_pretrained(name)
        _biobert_model = AutoModel.from_pretrained(name)
        _biobert_model.eval()
    return _biobert_tokenizer, _biobert_model


def biobert_similarity(pred, gt):
    if not pred or not gt:
        return 0.0
    import torch
    tokenizer, model = _get_biobert()
    texts = [str(pred)[:512], str(gt)[:512]]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    embs = embs.cpu().numpy()
    a, b = embs[0], embs[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── FM3S for Eligibility ─────────────────────────────────────────────────────

def _get_nouns(doc):
    nouns = []
    i = 0
    tokens = list(doc)
    while i < len(tokens):
        if tokens[i].pos_ in ('NOUN', 'PROPN'):
            compound = tokens[i].text
            j = i + 1
            while j < len(tokens) and tokens[j].pos_ in ('NOUN', 'PROPN', 'ADJ'):
                compound += ' ' + tokens[j].text
                j += 1
            nouns.append(compound.lower())
            i = j
        else:
            i += 1
    return nouns


def _get_verbs(doc):
    stop_verbs = {'be', 'have', 'say', 'do', 'will', 'would', 'could', 'should', 'may', 'might', 'shall'}
    verbs = []
    for token in doc:
        if token.pos_ == 'VERB' and token.lemma_.lower() not in stop_verbs:
            verbs.append((token.lemma_.lower(), token.tag_))
    return verbs


def _lin_sim(w1, w2):
    try:
        from nltk.corpus import wordnet as wn
        from nltk.corpus import wordnet_ic
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        syns1 = wn.synsets(w1)
        syns2 = wn.synsets(w2)
        best = 0.0
        for s1 in syns1[:3]:
            for s2 in syns2[:3]:
                try:
                    sim = s1.lin_similarity(s2, brown_ic)
                    if sim and sim > best:
                        best = sim
                except Exception:
                    pass
        return best
    except Exception:
        # Fallback: exact match
        return 1.0 if w1 == w2 else 0.0


def _noun_score(nouns1, nouns2):
    if not nouns1 or not nouns2:
        return 0.0
    scores = []
    for n in nouns1:
        best = max((_lin_sim(n.split()[0], m.split()[0]) for m in nouns2), default=0.0)
        scores.append(best)
    return sum(scores) / len(scores)


def _verb_score(verbs1, verbs2):
    if not verbs1 or not verbs2:
        return 0.0
    scores = []
    for v, tag in verbs1:
        same_tense = [vv for vv, ttag in verbs2 if ttag == tag]
        if not same_tense:
            scores.append(0.0)
            continue
        best = max((_lin_sim(v, vv) for vv in same_tense), default=0.0)
        scores.append(best)
    return sum(scores) / len(scores)


def _cwo_score(text1, text2):
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and', 'or', 'with', 'for', 'at', 'by'}
    def content_words(t):
        return [w.lower() for w in re.findall(r'\b\w+\b', t) if w.lower() not in stop and len(w) > 2]
    def bigrams(words):
        return list(zip(words, words[1:]))

    w1 = content_words(text1)
    w2 = content_words(text2)
    if not w1 or not w2:
        return 0.0

    common = set(w1) & set(w2)
    if not common:
        return 0.0

    # Unigram position agreement
    pos1 = {w: i/len(w1) for i, w in enumerate(w1)}
    pos2 = {w: i/len(w2) for i, w in enumerate(w2)}
    uni_agree = sum(1 for w in common if abs(pos1.get(w, 0) - pos2.get(w, 0)) < 0.1)
    uni_score = uni_agree / len(common)

    # Bigram order agreement
    bg1 = set(bigrams(w1))
    bg2 = set(bigrams(w2))
    common_bg = bg1 & bg2
    bi_score = len(common_bg) / len(bg1) if bg1 else 0.0

    return 0.5 * uni_score + 0.5 * bi_score


def fm3s(pred, gt):
    if not pred or not gt or pred in ('Not Specified', 'None') or gt in ('Not Specified', 'None'):
        return 0.0
    nlp = _get_spacy()
    doc1 = nlp(str(pred)[:5000])
    doc2 = nlp(str(gt)[:5000])

    nouns1, nouns2 = _get_nouns(doc1), _get_nouns(doc2)
    verbs1, verbs2 = _get_verbs(doc1), _get_verbs(doc2)

    ns = _noun_score(nouns1, nouns2)
    vs = _verb_score(verbs1, verbs2)
    cwo = _cwo_score(str(pred), str(gt))

    # FM3S aggregation
    score = (ns + vs + cwo) / 3.0
    return min(max(score, 0.0), 1.0)


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_row(pred, gt):
    scores = {}
    scores['conditions'] = biobert_similarity(pred.get('conditions', ''), gt.get('conditions', ''))
    scores['study_type'] = biobert_similarity(pred.get('study_type', ''), gt.get('study_type', ''))
    scores['sex'] = 1.0 if pred.get('sex', '').strip() == gt.get('sex', '').strip() else 0.0
    scores['minimum_age'] = number_similarity(pred.get('minimum_age', ''), gt.get('minimum_age', ''))
    scores['maximum_age'] = number_similarity(pred.get('maximum_age', ''), gt.get('maximum_age', ''))
    scores['eligibility_criteria'] = fm3s(pred.get('eligibility_criteria', ''), gt.get('eligibility_criteria', ''))
    return scores


if __name__ == '__main__':
    # Quick test
    pred = {'eligibility_criteria': 'Inclusion Criteria:\n\n* Adults 18 years or older\n* Confirmed HCC diagnosis\n\nExclusion Criteria:\n\n* Prior chemotherapy',
            'minimum_age': '18 Years', 'maximum_age': 'Not Specified',
            'conditions': "['Hepatocellular Carcinoma']", 'study_type': 'INTERVENTIONAL', 'sex': 'ALL'}
    gt = {'eligibility_criteria': 'Inclusion Criteria:\n\n* Age >= 18 years\n* Histologically confirmed HCC\n\nExclusion Criteria:\n\n* Previous antitumor treatment',
          'minimum_age': '18 Years', 'maximum_age': 'Not Specified',
          'conditions': "['Hepatocellular Carcinoma (HCC)']", 'study_type': 'INTERVENTIONAL', 'sex': 'ALL'}
    scores = evaluate_row(pred, gt)
    for k, v in scores.items():
        print(f'{k}: {v:.3f}')
    print(f'mean: {sum(scores.values())/len(scores):.3f}')
