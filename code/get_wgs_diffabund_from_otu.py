# Created by Lechuan 
"""
Compute pairwise diff-abundance plots from an OTU table + Newick tree.
- Rows of OTU = features/species names (must match tree tips)
- Columns of OTU = samples
- Tree tips = same species names as OTU row names
"""

import os
import argparse
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# =========================
# Tree parsing (with fallback)
# =========================
def parse_tree_file(tree_path):
    """
    Return: Tint, lint, nodes_in_order, nodes_to_index
    - Tint[i] = j means j is parent of i (indices refer to nodes_in_order)
    - lint[(i, j)] = branch length of edge i->j
    - nodes_in_order: bottom-up (leaves to root)
    - nodes_to_index: node label -> index
    """
    # Try DendroPy first
    try:
        from dendropy import Tree, datamodel
        with open(tree_path, "r") as f:
            newick = f.read().strip()
        dtree = Tree.get(
            data=newick,
            schema="newick",
            suppress_internal_node_taxa=False,
            store_tree_weights=True
        )

        # Ensure every internal node has a label
        i = 0
        for node in dtree.nodes():
            if node.taxon is None:
                node.taxon = datamodel.taxonmodel.Taxon(label=f"temp{i}")
                i += 1

        full_nodes_in_order = list(dtree.levelorder_node_iter())
        full_nodes_in_order.reverse()  # bottom-up

        nodes_in_order = [n.taxon.label for n in full_nodes_in_order]
        nodes_to_index = {lab: idx for idx, lab in enumerate(nodes_in_order)}

        Tint, lint = {}, {}
        for idx, node in enumerate(full_nodes_in_order):
            parent = node.parent_node
            if parent is not None:
                j_idx = nodes_to_index[parent.taxon.label]
                Tint[idx] = j_idx
                w = node.edge.length if node.edge.length is not None else 1.0
                lint[(idx, j_idx)] = float(w)

        return Tint, lint, nodes_in_order, nodes_to_index

    except Exception:
        # Fallback: Biopython
        from Bio import Phylo

        tree = Phylo.read(tree_path, "newick")

        # Give labels to unlabeled internal nodes
        counter = 0
        for clade in tree.find_clades(order="level"):
            if clade.name is None:
                clade.name = f"temp{counter}"
                counter += 1

        # Build bottom-up order: do level order then reverse so children appear before parent
        full_nodes = list(tree.find_clades(order="level"))
        full_nodes.reverse()

        nodes_in_order = [c.name for c in full_nodes]
        nodes_to_index = {lab: idx for idx, lab in enumerate(nodes_in_order)}

        # Build parent map first
        parent_map = {}
        for parent in tree.find_clades():
            for child in parent.clades:
                parent_map[child] = parent

        Tint, lint = {}, {}
        # Edge lengths: child.branch_length, default to 1.0 if None
        for idx, node in enumerate(full_nodes):
            parent = parent_map.get(node)
            if parent is not None:
                j_idx = nodes_to_index[parent.name]
                Tint[idx] = j_idx
                w = node.branch_length if node.branch_length is not None else 1.0
                lint[(idx, j_idx)] = float(w)

        return Tint, lint, nodes_in_order, nodes_to_index



def L2UniFrac_weighted(Tint, lint, nodes_in_order, P, Q, include_tmp_diffab=True):
	'''
	(Z, diffab) = L2Unifrac_weighted(Tint, lint, nodes_in_order, P, Q)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
	and two probability vectors P and Q (typically P = envs_prob_dict[samples[i]], Q = envs_prob_dict[samples[j]]).
	Returns the weighted Unifrac distance Z and the differential abundance. The differential abundance vector diffab 
	is a dictionary with tuple keys using elements of Tint and values diffab[(i, j)] equal to the signed difference 
	of abundance between the two samples restricted to the sub-tree defined by nodes_in_order(i) and weighted by the 
	edge length lint[(i,j)].
	'''
	num_nodes = len(nodes_in_order)
	Z = 0
	diffab = dict()
	partial_sums = [float(e1) - float(e2) for (e1, e2) in zip(P, Q)]
	for i in range(num_nodes - 1):
		val = partial_sums[i]
		partial_sums[Tint[i]] += val
		if val != 0 and (include_tmp_diffab or nodes_in_order[i][0] != 't'):
			diffab[(i, Tint[i])] = lint[i, Tint[i]]*val # Captures diffab
		Z += lint[i, Tint[i]]*(val**2)
	Z = np.sqrt(Z)
	return (Z, diffab)




# =========================
# IO helpers
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise diff-abundance plots from an OTU table + Newick tree (no .profile files)."
    )
    p.add_argument("-i", "--otu_file", required=True,
                   help="OTU table (TSV/CSV). Rows=species/features, Cols=samples.")
    p.add_argument("-t", "--tree_file", required=True,
                   help="Newick tree file; tip labels must match OTU row names.")
    p.add_argument("-m", "--meta_file", required=True,
                   help="Metadata file (TSV/CSV) with id_col and phenotype column.")
    p.add_argument("--id_col", default="library_id",
                   help="Sample ID column in metadata. Default: library_id")
    p.add_argument("-p", "--phenotype", default="IHO_Sea_fmt",
                   help="Grouping column in metadata. Default: IHO_Sea_fmt")
    p.add_argument("-s", "--save_dir", required=True,
                   help="Directory to save output PNGs.")
    p.add_argument("--prefix", default="diffab",
                   help="Filename prefix. Default: diffab")
    p.add_argument("--thresh", type=float, default=5e-4,
                   help="Plot threshold on |diffab|. Default: 5e-4")
    p.add_argument("--max_disp", type=int, default=0,
                   help="If >0, show only top-K positives and top-K negatives. Default: 0 (all above thresh)")
    p.add_argument("--sep_meta", default=None,
                   help="Explicit metadata separator (e.g. ',' or '\\t'); auto if omitted.")
    p.add_argument("--sep_otu", default=None,
                   help="Explicit OTU separator (e.g. ',' or '\\t'); auto if omitted.")
    p.add_argument("--compare", nargs=2, metavar=("GROUP_A", "GROUP_B"),
                   help="Only compare these two phenotype labels (A vs B). "
                        "Example: --compare 'North Atlantic Ocean' 'South Pacific Ocean'")
    return p.parse_args()


def read_table_auto(path, explicit_sep=None, index_col=0):
    if explicit_sep is not None:
        return pd.read_csv(path, sep=explicit_sep, index_col=index_col)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, index_col=index_col)
    return pd.read_csv(path, sep="\t", index_col=index_col)


def load_otu_vectors(otu_path, nodes_to_index, sep_override=None, normalize=True):
    """
    Read OTU (rows=features, cols=samples) and project to vectors aligned to nodes_in_order indices:
    - Only leaf (tip) positions get values; internal nodes = 0
    - Optionally column-normalize to relative abundances
    Return: dict sample_id -> np.ndarray(len(nodes))
    """
    df = read_table_auto(otu_path, explicit_sep=sep_override, index_col=0).fillna(0)
    # guard: ensure string row names
    df.index = df.index.map(str)

    if normalize:
        colsum = df.sum(axis=0)
        colsum[colsum == 0] = 1.0
        df = df.div(colsum, axis=1)

    n = len(nodes_to_index)
    sample_vecs = {}
    miss = 0

    for sample in df.columns:
        vec = np.zeros(n, dtype=float)
        col = df[sample]
        nz = col[col != 0]
        for feature, val in nz.items():
            if feature in nodes_to_index:
                vec[nodes_to_index[feature]] = float(val)
            else:
                miss += 1
        sample_vecs[str(sample)] = vec

    if miss > 0:
        print(f"[warn] {miss} feature entries not found in tree tips (ignored).")
    return sample_vecs


def get_rep_vectors(sample_vecs, meta_path, id_col, phenotype_col, sep_override=None):
    """
    Group by phenotype and average sample vectors element-wise to get representative vectors.
    Return: dict phenotype -> vector
    """
    meta = read_table_auto(meta_path, explicit_sep=sep_override, index_col=None)
    meta[id_col] = meta[id_col].astype(str)
    present = meta[id_col].isin(sample_vecs.keys())
    meta = meta[present].copy()
    if meta.empty:
        raise ValueError("No overlapping samples between metadata and OTU table.")

    rep = {}
    for label, sub in meta.groupby(phenotype_col, dropna=False):
        sids = sub[id_col].tolist()
        mats = np.stack([sample_vecs[sid] for sid in sids], axis=0)
        rep[str(label) if pd.notna(label) else "unknown"] = mats.mean(axis=0)
    return rep


# =========================
# Plotting (simple stem plot)
# =========================
def plot_diffab_simple(nodes_in_order, diffab, P_label, Q_label, index_to_label,
                       thresh=5e-4, maxDisp=0, leaves=None, only_species=False):
    
    n = len(nodes_in_order)
    y = np.zeros(n)
    for (i, _j), v in diffab.items():
        y[i] = v

    if leaves is None:
        leaves = set(range(n))

    
    cand_pos = [i for i in range(n) if (i in leaves) and y[i] >  thresh]
    cand_neg = [i for i in range(n) if (i in leaves) and y[i] < -thresh]

    if only_species:
        cand_pos = [i for i in cand_pos if "s__" in str(index_to_label.get(i, ""))]
        cand_neg = [i for i in cand_neg if "s__" in str(index_to_label.get(i, ""))]

    if maxDisp:
        cand_pos = cand_pos[:maxDisp]
        cand_neg = cand_neg[:maxDisp]

    sel = [i for i in range(n) if (i in leaves) and abs(y[i]) > thresh]

    if not sel:
        return None

    xs_new = list(range(len(sel)))
    ys_new = [y[i] for i in sel]

    labels = [str(index_to_label.get(i, i)) for i in sel]


    
    xs_pos_new = [x for x, i in zip(xs_new, sel) if y[i] > 0]
    ys_pos_new = [y[i] for i in sel if y[i] > 0]
    xs_neg_new = [x for x, i in zip(xs_new, sel) if y[i] < 0]
    ys_neg_new = [y[i] for i in sel if y[i] < 0]

    fig, ax = plt.subplots(figsize=(18, 6))  

    
    ax.axhline(0, color="black", linewidth=1)

    
    if xs_pos_new:
        ax.stem(xs_pos_new, ys_pos_new,
                linefmt="b-", markerfmt="bo", basefmt=" ")

    
    if xs_neg_new:
        ax.stem(xs_neg_new, ys_neg_new,
                linefmt="r-", markerfmt="ro", basefmt=" ")

    ax.set_ylabel("Differential Abundance")
    ax.set_title(f"{P_label} vs {Q_label}")

    ax.set_xticks(xs_new)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

    for i, lab in enumerate(ax.get_xticklabels()):
        if i % 2 == 1:
            lab.set_y(lab.get_position()[1] - 0.03)

    plt.tight_layout()
    return fig




def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Parse tree
    Tint, lint, nodes_in_order, nodes_to_index = parse_tree_file(args.tree_file)
    index_to_label = {v: k for k, v in nodes_to_index.items()}

    parent_set = set(Tint.values())
    leaves = {i for i in range(len(nodes_in_order)) if i not in parent_set}

    # 2) OTU -> sample vectors
    sample_vecs = load_otu_vectors(args.otu_file, nodes_to_index, sep_override=args.sep_otu, normalize=True)

    # 3) Representative vectors per phenotype
    rep_dict = get_rep_vectors(sample_vecs, args.meta_file, args.id_col, args.phenotype, sep_override=args.sep_meta)
    groups = list(rep_dict.keys())
    if len(groups) < 2:
        raise ValueError("Need at least 2 phenotype groups to compare.")
    
    if args.compare:
        g1, g2 = args.compare[0], args.compare[1]
        missing = [g for g in (g1, g2) if g not in rep_dict]
        if missing:
            avail = ", ".join(sorted(rep_dict.keys()))
            raise ValueError(f"Group(s) not found in metadata representatives: {missing}. "
                             f"Available groups: {avail}")
        compare_pairs = [(g1, g2)]
    else:
        compare_pairs = list(it.combinations(groups, 2))

    # 4) Pairwise comparisons -> L2UniFrac -> plot
    for g1, g2 in compare_pairs:
        P, Q = rep_dict[g1], rep_dict[g2]
        Z, diffab = L2UniFrac_weighted(Tint, lint, nodes_in_order, P, Q, include_tmp_diffab=False)

    
        fname = f"{args.prefix}_{sanitize_filename(g1)}_{sanitize_filename(g2)}_diffabund.png"
        out = os.path.join(args.save_dir, fname)

    
        fig = plot_diffab_simple(nodes_in_order, diffab, g1, g2, index_to_label,
                                 thresh=args.thresh, maxDisp=args.max_disp,
                                 leaves=leaves, only_species=False)

    
        if fig is None:
            print(f"[skip] {g1} vs {g2}: no points after filters (thresh={args.thresh})")
            continue

    
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[ok] saved: {out} (distance Z={Z:.6g})")



if __name__ == "__main__":
    main()
