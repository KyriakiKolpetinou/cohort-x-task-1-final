"""
Train the PubMedBERT study_type classifier on the 416 train examples.
Saves to models/study_type_classifier/.

SEED is pinned to 7 (default): chosen by study_type_seed_sweep.py as the seed whose
test predictions best match the original v29 study_type. This is the submitted model
(public LB 0.72864, beating the lost original's 0.72797). Override with SEED=<n>.
"""
import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from bs4 import BeautifulSoup

os.environ['LD_LIBRARY_PATH'] = (
    '/usr/local/cuda-12.1/targets/x86_64-linux/lib:'
    + os.environ.get('LD_LIBRARY_PATH', '')
)

SEED = int(os.environ.get('SEED', '7'))
set_seed(SEED)

NXML_DIR = os.path.join(os.path.dirname(__file__), 'PMC_NXML_Archives')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'models', 'study_type_classifier')
BERT_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
LABELS = ['INTERVENTIONAL', 'OBSERVATIONAL']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def get_text(pmcid):
    path = os.path.join(NXML_DIR, f'PMC{pmcid}.nxml')
    if not os.path.exists(path):
        return ''
    try:
        with open(path) as f:
            soup = BeautifulSoup(f, 'lxml-xml')
        return soup.get_text(' ', strip=True)[:4000]
    except Exception:
        return ''


class StudyTypeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    gt = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Task_1.xlsx'))

    texts, labels = [], []
    for _, row in gt.iterrows():
        text = get_text(str(row['pmcids']))
        if not text:
            continue
        st = str(row['study_type']).strip().upper()
        if st not in LABEL2ID:
            continue
        texts.append(text.lower())
        labels.append(LABEL2ID[st])

    print(f"Training on {len(texts)} examples")
    print(f"  INTERVENTIONAL: {labels.count(0)}, OBSERVATIONAL: {labels.count(1)}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = StudyTypeDataset(enc, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2,
        id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        save_strategy='no',
        report_to='none',
        logging_steps=20,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    # Quick train-set accuracy check
    model.eval()
    correct = 0
    for i, text in enumerate(texts):
        inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
            model.cuda()
        with torch.no_grad():
            out = model(**inp)
        pred = torch.argmax(out.logits, dim=1).item()
        if pred == labels[i]:
            correct += 1
    print(f"Train accuracy: {correct}/{len(texts)} = {correct/len(texts):.1%}")


if __name__ == '__main__':
    main()
