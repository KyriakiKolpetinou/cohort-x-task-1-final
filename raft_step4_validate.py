"""RAFT Phase 4: Validate v17 RAFT model vs v13 BART on val + train subsets."""
import os, sys, re, argparse, time
import torch
sys.path.insert(0, os.path.dirname(__file__))

from transformers import BartTokenizerFast, BartForConditionalGeneration
from nxml_parser import parse_nxml
from prepare_ft_data import build_input_text, read_train_rows
from evaluate import fm3s

HERE = os.path.dirname(os.path.abspath(__file__))

V13 = os.environ.get('V13_DIR', os.path.join(HERE, 'models', 'bart_eligibility_v1', 'final'))
V17 = os.environ.get('BART_DIR', os.path.join(HERE, 'models', 'bart_raft_v17_final', 'final'))
PREFIX = 'Extract eligibility criteria: '


def jacc(a, b):
    A = set(re.findall(r'[a-z0-9]+', re.sub(r'\s+', ' ', str(a).lower())))
    B = set(re.findall(r'[a-z0-9]+', re.sub(r'\s+', ' ', str(b).lower())))
    return len(A & B) / len(A | B) if (A | B) else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()

    print('Loading v13...', flush=True)
    tok_v13 = BartTokenizerFast.from_pretrained(V13)
    m_v13 = BartForConditionalGeneration.from_pretrained(V13).eval().to('cuda')

    print('Loading v17 (RAFT)...', flush=True)
    tok_v17 = BartTokenizerFast.from_pretrained(V17)
    m_v17 = BartForConditionalGeneration.from_pretrained(V17).eval().to('cuda')

    def gen(model, tokenizer, parsed):
        inp = PREFIX + build_input_text(parsed)
        enc = tokenizer(inp, max_length=1024, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**enc, max_length=384, num_beams=4,
                                  no_repeat_ngram_size=3, early_stopping=True)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    train = read_train_rows()[:args.n]
    sums = {'v13': {'fm': 0, 'jc': 0, 'len': 0},
            'v17': {'fm': 0, 'jc': 0, 'len': 0}}
    n_done = 0
    v17_wins = 0; v13_wins = 0
    for i, gt in enumerate(train, 1):
        parsed = parse_nxml(gt['pmcids'])
        if parsed is None: continue
        n_done += 1
        gt_e = gt.get('eligibility_criteria', '') or ''

        v13_out = gen(m_v13, tok_v13, parsed)
        v17_out = gen(m_v17, tok_v17, parsed)

        f_v13 = fm3s(v13_out, gt_e); f_v17 = fm3s(v17_out, gt_e)
        sums['v13']['fm'] += f_v13; sums['v13']['jc'] += jacc(v13_out, gt_e); sums['v13']['len'] += len(v13_out)
        sums['v17']['fm'] += f_v17; sums['v17']['jc'] += jacc(v17_out, gt_e); sums['v17']['len'] += len(v17_out)
        if f_v17 > f_v13 + 0.01: v17_wins += 1
        elif f_v13 > f_v17 + 0.01: v13_wins += 1
        print(f'[{i:3d}/{args.n}] PMC{gt["pmcids"]:>8s}  v13 f{f_v13:.2f}/l{len(v13_out):>3d}  '
              f'v17 f{f_v17:.2f}/l{len(v17_out):>3d}  Δ={f_v17-f_v13:+.3f}', flush=True)

    print(f'\n=== Avg over {n_done} train ===')
    for k in ['v13', 'v17']:
        s = sums[k]
        print(f'  {k:<5s} fm3s={s["fm"]/n_done:.4f}  jacc={s["jc"]/n_done:.4f}  avglen={s["len"]/n_done:.0f}')
    print(f'\nWin counts (>+0.01 fm3s):  v17={v17_wins}  v13={v13_wins}  tied={n_done-v17_wins-v13_wins}')


if __name__ == '__main__':
    main()
