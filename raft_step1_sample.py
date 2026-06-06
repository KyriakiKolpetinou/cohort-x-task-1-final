"""RAFT step 1: sample N=8 diverse candidates per train example from v13 BART.

Output: raft_candidates.jsonl with one row per (train_example × candidate).
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('HF_HOME', '/mnt/extra_storage/kkolpetinou/torch_cache/huggingface')

from transformers import BartTokenizerFast, BartForConditionalGeneration

MODEL_DIR = '/mnt/extra_storage/kkolpetinou/bart_eligibility_v1/final'  # v13
PREFIX = 'Extract eligibility criteria: '
OUT_PATH = '/home/kkolpetinou/cohort-x-task-1/raft_candidates.jsonl'


def load_train_examples():
    base = os.path.dirname(__file__)
    examples = []
    with open(os.path.join(base, 'ft_train.jsonl')) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--max_length', type=int, default=384)
    args = parser.parse_args()

    print(f'Loading v13 BART from {MODEL_DIR}...', flush=True)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).eval().to('cuda')

    examples = load_train_examples()
    print(f'Sampling {args.n_samples} candidates per example for {len(examples)} train examples', flush=True)

    out_f = open(OUT_PATH, 'w')
    t0 = time.time()

    for i, ex in enumerate(examples, 1):
        inp = PREFIX + ex['input']
        enc = tokenizer(inp, max_length=1024, truncation=True, return_tensors='pt').to('cuda')

        # Generate N samples with pure sampling (num_beams=1 to allow num_return_sequences > 1)
        with torch.no_grad():
            samples_ids = model.generate(
                **enc,
                max_length=args.max_length,
                do_sample=True,
                num_beams=1,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.n_samples,
                no_repeat_ngram_size=3,
            )
        candidates = [tokenizer.decode(s, skip_special_tokens=True) for s in samples_ids]

        # Also one greedy "default" output
        with torch.no_grad():
            greedy_ids = model.generate(
                **enc, max_length=args.max_length, num_beams=4,
                no_repeat_ngram_size=3, early_stopping=True,
            )
        greedy = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

        row = {
            'pmcid': ex['pmcid'],
            'input': ex['input'],
            'gt': ex['output'],
            'greedy': greedy,
            'candidates': candidates,
        }
        out_f.write(json.dumps(row) + '\n')
        out_f.flush()

        if i % 20 == 0 or i == len(examples):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1)
            eta = (len(examples) - i) / max(rate, 0.001)
            print(f'  [{i}/{len(examples)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s', flush=True)

    out_f.close()
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
