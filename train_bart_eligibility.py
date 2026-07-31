"""Fine-tune BART-base to map article (title+abstract+methods) -> GT eligibility text."""
import os, sys, json, argparse
import torch

from transformers import (
    BartTokenizerFast, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'facebook/bart-base'
OUT_DIR = os.environ.get('OUT_DIR', os.path.join(HERE, 'models', 'bart_eligibility_v1'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_target_len', type=int, default=384)
    parser.add_argument('--quick', action='store_true', help='2 epochs, for sanity test')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2

    print(f'Loading {MODEL_NAME}...', flush=True)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    print('Loading data...', flush=True)
    base = os.path.dirname(__file__)
    ds = load_dataset('json', data_files={
        'train': os.path.join(base, 'ft_train.jsonl'),
        'val':   os.path.join(base, 'ft_val.jsonl'),
    })
    print(f'  train: {len(ds["train"])}, val: {len(ds["val"])}')

    PREFIX = "Extract eligibility criteria: "

    def preprocess(ex):
        inputs = [PREFIX + t for t in ex['input']]
        model_inputs = tokenizer(inputs, max_length=args.max_input_len,
                                  truncation=True, padding=False)
        labels = tokenizer(text_target=ex['output'],
                            max_length=args.max_target_len,
                            truncation=True, padding=False)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)
    print(f'  tokenized columns: {tokenized["train"].column_names}')

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
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['val'],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f'\nTraining for up to {args.epochs} epochs...', flush=True)
    trainer.train()

    print(f'\nSaving best model to {OUT_DIR}/final...', flush=True)
    trainer.save_model(os.path.join(OUT_DIR, 'final'))
    tokenizer.save_pretrained(os.path.join(OUT_DIR, 'final'))

    # Quick eval: generate on 3 val examples
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print('\n=== Sample generations on val ===')
    for i in range(min(3, len(ds['val']))):
        ex = ds['val'][i]
        inp = PREFIX + ex['input']
        enc = tokenizer(inp, max_length=args.max_input_len, truncation=True,
                        return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_length=args.max_target_len,
                                 num_beams=4, no_repeat_ngram_size=3)
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n--- val PMC{ex['pmcid']} ---")
        print(f"PRED: {gen[:300]}")
        print(f"GT:   {ex['output'][:300]}")


if __name__ == '__main__':
    main()
