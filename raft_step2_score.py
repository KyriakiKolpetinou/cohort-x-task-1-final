"""RAFT step 2: Score all candidates with FM3S vs GT. Build best-of-N training set."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import fm3s

HERE = os.path.dirname(os.path.abspath(__file__))

CAND_PATH = os.path.join(HERE, 'raft_candidates.jsonl')
TRAIN_OUT = os.path.join(HERE, 'raft_train_best.jsonl')
STATS_OUT = os.path.join(HERE, 'raft_stats.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_top', type=int, default=1, help='How many top candidates to keep per example')
    args = parser.parse_args()

    rows = []
    with open(CAND_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f'Loaded {len(rows)} candidate groups', flush=True)

    best_rows = []
    stats = {
        'greedy_fm3s_sum': 0.0,
        'best_fm3s_sum': 0.0,
        'mean_fm3s_sum': 0.0,
        'n': 0,
        'beat_greedy': 0,  # how often best-of-N beats greedy
    }

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        gt = row['gt']
        greedy = row['greedy']
        candidates = row['candidates']

        # Score each
        cand_scores = [fm3s(c, gt) for c in candidates]
        greedy_score = fm3s(greedy, gt)

        # Find best among candidates
        best_idx = max(range(len(cand_scores)), key=lambda j: cand_scores[j])
        best_score = cand_scores[best_idx]
        best_cand = candidates[best_idx]

        # If greedy is better, use that
        use_greedy = greedy_score >= best_score
        chosen = greedy if use_greedy else best_cand
        chosen_score = max(greedy_score, best_score)

        stats['greedy_fm3s_sum'] += greedy_score
        stats['best_fm3s_sum'] += chosen_score
        stats['mean_fm3s_sum'] += sum(cand_scores) / len(cand_scores)
        stats['n'] += 1
        if best_score > greedy_score:
            stats['beat_greedy'] += 1

        # Output the chosen as training target
        best_rows.append({
            'pmcid': row['pmcid'],
            'input': row['input'],
            'output': chosen,
            'fm3s': chosen_score,
            'greedy_fm3s': greedy_score,
            'mean_cand_fm3s': sum(cand_scores) / len(cand_scores),
            'used_greedy': use_greedy,
        })

        if i % 20 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(f'  [{i}/{len(rows)}] {elapsed:.0f}s, mean_greedy={stats["greedy_fm3s_sum"]/stats["n"]:.3f}, '
                  f'mean_best={stats["best_fm3s_sum"]/stats["n"]:.3f}', flush=True)

    n = stats['n']
    print(f'\n=== RAFT Step 2 Summary ===')
    print(f'Examples: {n}')
    print(f'Mean greedy FM3S (vs GT): {stats["greedy_fm3s_sum"]/n:.4f}')
    print(f'Mean candidate FM3S:      {stats["mean_fm3s_sum"]/n:.4f}')
    print(f'Mean BEST-of-N FM3S:      {stats["best_fm3s_sum"]/n:.4f}')
    print(f'Sampled beats greedy in {stats["beat_greedy"]}/{n} examples ({100*stats["beat_greedy"]/n:.1f}%)')

    with open(TRAIN_OUT, 'w') as f:
        for r in best_rows:
            f.write(json.dumps({'pmcid': r['pmcid'], 'input': r['input'], 'output': r['output']}) + '\n')
    with open(STATS_OUT, 'w') as f:
        json.dump({k: v / n if k.endswith('_sum') else v for k, v in stats.items()}, f, indent=2)
    print(f'\nWrote training data → {TRAIN_OUT}')
    print(f'Wrote stats → {STATS_OUT}')


if __name__ == '__main__':
    main()
