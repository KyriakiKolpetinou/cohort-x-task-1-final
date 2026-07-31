"""RAFT step 3: Fine-tune v13 BART on best-of-N outputs (raft_train_best.jsonl).

Starts from v13 BART checkpoint (not from facebook/bart-base) — we're refining.
"""
import os, sys, argparse
import torch

from transformers import (
    BartTokenizerFast, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

START_FROM = os.environ.get('START_FROM', os.path.join(HERE, 'models', 'bart_eligibility_v1', 'final'))   # v13
OUT_DIR = os.environ.get('OUT_DIR', os.path.join(HERE, 'models', 'bart_raft_v17_final'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)  # lower LR for refinement
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_target_len', type=int, default=384)
    args = parser.parse_args()

    print(f'Loading v13 BART from {START_FROM}...', flush=True)
    tokenizer = BartTokenizerFast.from_pretrained(START_FROM)
    model = BartForConditionalGeneration.from_pretrained(START_FROM)

    base = os.path.dirname(__file__)
    ds = load_dataset('json', data_files={
        'train': os.path.join(base, 'raft_train_best.jsonl'),  # best-of-N outputs
        'val':   os.path.join(base, 'ft_val.jsonl'),
    })
    print(f'  train: {len(ds["train"])}, val: {len(ds["val"])}')

    PREFIX = "Extract eligibility criteria: "
    def preprocess(ex):
        inputs = [PREFIX + t for t in ex['input']]
        model_inputs = tokenizer(inputs, max_length=args.max_input_len, truncation=True, padding=False)
        labels = tokenizer(text_target=ex['output'], max_length=args.max_target_len, truncation=True, padding=False)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        fp16=True,
        report_to='none',
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=tokenized['train'], eval_dataset=tokenized['val'],
        tokenizer=tokenizer, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f'\nRAFT fine-tuning v13 BART on best-of-N outputs for {args.epochs} epochs...', flush=True)
    trainer.train()
    trainer.save_model(os.path.join(OUT_DIR, 'final'))
    tokenizer.save_pretrained(os.path.join(OUT_DIR, 'final'))
    print(f'Saved → {OUT_DIR}/final')


if __name__ == '__main__':
    main()
