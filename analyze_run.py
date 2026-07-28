"""
Analyze a training run: coverage trend, best episode, hidden norm, etc.
Usage: python analyze_run.py --tag v6_gae_fix
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from run_manager import find_run

def analyze(run_dir):
    data_path = os.path.join(run_dir, "plots", "training_data.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        return False

    d = np.load(data_path)
    cov = d['coverage']
    rew = d['reward']
    length = d['length']
    lc = d['loss_c']
    la = d['loss_a']
    hn = d['hidden_norm']
    n_eps = len(cov)

    # Where was best?
    best_idx = cov.argmax()
    best_val = cov.max()
    early_best = best_idx < n_eps * 0.1  # best in first 10%

    # Coverage trend
    first_q = cov[:n_eps//4].mean()
    last_q = cov[3*n_eps//4:].mean()
    trend = "UP" if last_q > first_q * 1.05 else ("DOWN" if last_q < first_q * 0.95 else "FLAT")

    # Hidden norm
    hn_first = hn[:10].mean()
    hn_last = hn[-10:].mean()
    hn_growing = hn_last > hn_first * 1.2

    # Completion bonus
    max_steps = length.max()
    early_stops = (length < max_steps).sum()

    # Reward
    rew_first = rew[:10].mean()
    rew_last = rew[-10:].mean()

    # Critic loss
    lc_first = lc[:10].mean()
    lc_last = lc[-10:].mean()
    lc_std_first = lc[:10].std()
    lc_std_last = lc[-10:].std()

    print(f"Run: {os.path.basename(run_dir)}")
    print(f"Episodes: {n_eps}")
    print(f"{'='*55}")
    print(f"{'Metric':<30} {'Value':>10}  {'Status':>10}")
    print(f"{'-'*55}")
    print(f"{'Best Coverage':<30} {best_val:>9.2%}  {'!! EARLY !!' if early_best else 'ok'}")
    print(f"{'Best at episode':<30} {best_idx+1:>10}")
    print(f"{'Coverage trend':<30} {first_q:>9.2%} → {last_q:.2%}  {trend}")
    print(f"{'Hidden norm':<30} {hn_first:>9.1f} → {hn_last:.1f}  {'OK GROWING' if hn_growing else '!! FLAT !!'}")
    print(f"{'Completion bonus':<30} {early_stops:>9} /{n_eps}  {'OK' if early_stops > n_eps*0.05 else '!! RARE !!'}")
    print(f"{'Reward':<30} {rew_first:>9.0f} → {rew_last:.0f}  {'UP' if rew_last > rew_first else 'DOWN'}")
    print(f"{'Critic loss':<30} {lc_first:>9.1f} → {lc_last:.1f} (σ:{lc_std_first:.1f}→{lc_std_last:.1f})")
    print(f"{'='*55}")

    # Verdict
    issues = []
    if early_best:
        issues.append("BEST_TOO_EARLY (best in first 10% = random luck)")
    if trend == "DOWN":
        issues.append("COVERAGE_DECLINING (policy collapse)")
    if not hn_growing:
        issues.append("HIDDEN_NOT_GROWING (GRU not learning)")
    if early_stops < n_eps * 0.05:
        issues.append("COMPLETION_RARE (threshold too high or unreachable)")
    if lc_std_last > lc_std_first * 3:
        issues.append("CRITIC_DIVERGING (increasing loss variance)")

    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  !! {i}")
        return False
    else:
        print("ALL CHECKS PASSED - training looks healthy")
        return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--run", type=str, default=None)
    args = p.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
    if not run_dir:
        print(f"Run not found: tag={args.tag}, run={args.run}")
        sys.exit(1)
    analyze(run_dir)

if __name__ == "__main__":
    main()
