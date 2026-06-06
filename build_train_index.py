"""
Build a retrieval index from train articles for RAG-based extraction.
Embeds title+abstract of each train article using PubMedBERT.
Saves embeddings + GT labels to train_index.json.
"""
import json, os, sys, re
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nxml_parser import parse_nxml

TASK_XLSX = os.path.join(os.path.dirname(__file__), 'Task_1.xlsx')
INDEX_FILE = os.path.join(os.path.dirname(__file__), 'train_index.json')
MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)


def embed_texts(texts, tokenizer, model, batch_size=16):
    import torch
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
        all_embeddings.append(emb.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f'  Embedded {min(i+batch_size, len(texts))}/{len(texts)}', flush=True)
    return np.vstack(all_embeddings)


def main():
    import openpyxl
    from transformers import AutoTokenizer, AutoModel

    print('Loading PubMedBERT...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print('Reading train data...')
    wb = openpyxl.load_workbook(TASK_XLSX)
    ws = wb['Train']
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c else '' for c in rows[0]]

    records = []
    for row in rows[1:]:
        if not row[0]: continue
        d = {header[i]: (str(row[i]).strip() if row[i] is not None else '') for i in range(len(header))}
        pmcid = str(int(float(d['pmcids'])))
        parsed = parse_nxml(pmcid)
        if parsed is None:
            continue
        title = parsed.get('title', '').strip()
        abstract = parsed.get('abstract_text', '').strip()
        methods = parsed.get('methods_text', '').strip()
        eligib = parsed.get('eligibility_text', '').strip()

        # Article text for embedding (title + abstract)
        embed_text = f"{title} {abstract[:800]}".strip()

        # Article text for the prompt example — richer text for better few-shot demonstrations
        parts = [f"Title: {title}"]
        if abstract:
            parts.append(f"Abstract:\n{abstract[:1500]}")
        if eligib:
            parts.append(f"Eligibility section:\n{eligib[:2000]}")
        if methods:
            parts.append(f"Methods/Patients section:\n{methods[:2500]}")
        article_text = '\n\n'.join(parts)

        records.append({
            'pmcid': pmcid,
            'embed_text': embed_text,
            'article_text': article_text,
            'gt': {
                'eligibility_criteria': d.get('eligibility_criteria', ''),
                'minimum_age': d.get('minimum_age', ''),
                'maximum_age': d.get('maximum_age', ''),
                'conditions': d.get('conditions', ''),
                'study_type': d.get('study_type', ''),
                'sex': d.get('sex', ''),
            }
        })

    print(f'Loaded {len(records)} train articles with parsed NXML')

    texts = [r['embed_text'] for r in records]
    print('Embedding...')
    embeddings = embed_texts(texts, tokenizer, model)

    # Save index
    index = {
        'records': records,
        'embeddings': embeddings.tolist(),
    }
    with open(INDEX_FILE, 'w') as f:
        json.dump(index, f)
    print(f'Saved index to {INDEX_FILE} ({len(records)} entries)')


if __name__ == '__main__':
    main()
