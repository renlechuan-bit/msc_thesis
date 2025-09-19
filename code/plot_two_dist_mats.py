import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

def plot_two_dist_mats_aligned(tsv1, tsv2, save_as, x_label=None, y_label=None):
    df1 = pd.read_table(tsv1, index_col=0)
    df2 = pd.read_table(tsv2, index_col=0)

    
    common = sorted(set(df1.index) & set(df1.columns) & set(df2.index) & set(df2.columns))
    if len(common) < 2:
        raise ValueError("Common samples < 2, cannot compare.")
    df1 = df1.loc[common, common]
    df2 = df2.loc[common, common]

    xs, ys = [], []
    for s1, s2 in combinations(common, 2):
        xs.append(df1.loc[s1, s2])
        ys.append(df2.loc[s1, s2])

    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size == 0:
        raise ValueError("Valid points count is 0.")

    corr = np.corrcoef(xs, ys)[0, 1]

    plt.figure(figsize=(6,5))
    plt.scatter(xs, ys, s=2, alpha=0.4)
    #if xs.size >= 2:
    #    k, b = np.polyfit(xs, ys, 1)
    #    xx = np.array([xs.min(), xs.max()])
    #    plt.plot(xx, k*xx + b, linewidth=1)
    plt.title(f"Correlation coefficient: {corr:.3f}")
    plt.xlabel(x_label or Path(tsv1).stem)
    plt.ylabel(y_label or Path(tsv2).stem)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
    return corr

dist_dir = Path("/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/dist_matrices")
l1_path  = "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/L1_extend_out_unifrac.tsv"
out_dir  = Path("/data/gpfs/projects/punim2504/msc_thesis/plots/L1--dist_mat")
out_dir.mkdir(parents=True, exist_ok=True)

summary = []
l1_name = Path(l1_path).name
for tsv in sorted(dist_dir.glob("*.tsv")):
    if tsv.name == l1_name:
        continue
    out_png = out_dir / f"L1_vs_{tsv.stem}.png"
    try:
        r = plot_two_dist_mats_aligned(l1_path, str(tsv), str(out_png), x_label="L1-UniFrac")
        print(f"[OK] {tsv.stem}: r={r:.3f}")
        summary.append((tsv.stem, r, str(out_png)))
    except Exception as e:
        print(f"[FAIL] {tsv.name}: {e}")
