import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler



def split_on_capitals(s, length=3):
    return ' '.join(re.findall(r'[A-Z][a-z]*', s)) if len(s) > length else s


def old_plot_colored_markers(sample_metadata, color_category="depth", cmap="tab20c", jitter=0.05, legend=True, title=None):
    # Create figure and projection
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(20 * 4, 5 * 4))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([-180, 180, -90, 90], crs=projection)

    # Add Natural Earth data
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, color="white")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Create a continuous color map
    cmap = plt.get_cmap(cmap)
    color_dict = {category: cmap(i) for i, category in enumerate(sample_metadata[color_category].unique())}

    for ix, row in sample_metadata.iterrows():
        x, y = row["longitude"], row["latitude"]
        
        # Add jitter to x and y coordinates
        if jitter:
            x += np.random.uniform(-jitter, jitter)
            y += np.random.uniform(-jitter, jitter)
        color = color_dict[row[color_category]]
        line, = ax.plot(x, y, marker="o", markersize=10, color=color, zorder=3, markeredgecolor="k", alpha=0.75)
    
    if legend:
        patches = {str(label): matplotlib.patches.Patch(color=color, label=label) for label, color in color_dict.items()}
        try:
            patches = {f"{k} (N = {sample_metadata[color_category].value_counts().loc[int(k)]})": v for k, v in sorted(patches.items())}
        except ValueError:
            patches = {f"{k} (N = {sample_metadata[color_category].value_counts().loc[k]})": v for k, v in sorted(patches.items())}
        ax.legend(patches.values(), patches.keys(), loc="best")
    

    ax.set_title(color_category if title is None else title, fontdict={"fontsize": 20})

    return ax


def plot_colored_markers(sample_metadata, color_category, cmap="tab20c", jitter=0.05, legend=True, title=None, palette=None, drop_duplicate_coords=True):
    # Create figure and projection
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(20 * 4, 5 * 4))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([-180, 180, -90, 90], crs=projection)

    # Add Natural Earth data
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, color="white")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Create a continuous color map
    cmap = plt.get_cmap(cmap)
    color_dict = {category: cmap(i) for i, category in enumerate(sample_metadata[color_category].unique())}
    if palette is not None:
        color_dict = palette
        color_dict = {k: v for k, v in color_dict.items() if k in sample_metadata[color_category].unique()}

    if drop_duplicate_coords:
        sample_metadata_ = sample_metadata.drop_duplicates(subset="coords")
    else:
        sample_metadata_ = sample_metadata

    for ix, row in sample_metadata_.iterrows():
        x, y = row["longitude"], row["latitude"]
        
        # Add jitter to x and y coordinates
        if jitter:
            x += np.random.uniform(-jitter, jitter)
            y += np.random.uniform(-jitter, jitter)
        color = color_dict[row[color_category]]
        line, = ax.plot(x, y, marker="o", markersize=10, color=color, zorder=3, markeredgecolor="k", alpha=0.75)
    
    if legend:
        patches = {str(label): matplotlib.patches.Patch(color=color, label=label) for label, color in color_dict.items()}
        try:
            patches = {f"{k} (N = {sample_metadata[color_category].value_counts().loc[int(k)]})": v for k, v in sorted(patches.items())}
        except (KeyError, ValueError):
            patches = {f"{k} (N = {sample_metadata[color_category].value_counts().loc[k]})": v for k, v in sorted(patches.items())}
        ax.legend(patches.values(), patches.keys(), loc="best")
    

    ax.set_title(color_category if title is None else title, fontdict={"fontsize": 20})

    return ax

get_cmap_ix_tab20c = lambda ix: matplotlib.colors.rgb2hex(plt.get_cmap("tab20c").colors[ix])
get_cmap_ix = lambda ix, cmap: matplotlib.colors.rgb2hex(plt.get_cmap(cmap).colors[ix])
palettes = {
        "k_10": {
    16: {
        "description": "Baltic Sea",
        "label": "BALT",
        "category": "BALT",
        "counts": 51,
        "color": get_cmap_ix(10, "tab20"),
        "marker": "X"
    },
    11: {
        "description": "Pacific Equatorial divergence/countercurrent",
        "label": "TRHI",
        "category": "TROP",
        "counts": 54,
        "color": get_cmap_ix(8, "tab20"),
        "marker": "s"
    },
    5: {
        "description": "Subtropical oceanic gyres",
        "label": "TGYR",
        "category": "TROP",
        "counts": 161,
        "color": get_cmap_ix(1, "tab20"),
        "marker": "D"
    },
    9: {
        "description": "Broad Tropical",
        "label": "TRLO",
        "category": "TROP",
        "counts": 818,
        "color": get_cmap_ix(0, "tab20"),
        "marker": "o"
    },
    0: {
        "description": "Antarctic polar",
        "label": "APLR",
        "category": "POLR",
        "counts": 30,
        "color": get_cmap_ix(15, "tab20"),
        "marker": "v"
    },
    14: {
        "description": "Arctic/Boreal polar",
        "label": "BPLR",
        "category": "POLR",
        "counts": 42,
        "color": get_cmap_ix(14, "tab20"),
        "marker": "^"
    },
    10: {
        "description": "Coastal temperate/Upwelling areas",
        "label": "CTEM",
        "category": "TEMP",
        "counts": 139,
        "color": get_cmap_ix(4, "tab20"),
        "marker": "P"
    },
    2: {
        "description": "Subantarctic-like",
        "label": "OTEM",
        "category": "TEMP",
        "counts": 43,
        "color": get_cmap_ix(6, "tab20"),
        "marker": "<"
    },
    3: {
        "description": "North Atlantic Drift-like",
        "label": "STEM",
        "category": "TEMP",
        "counts": 34,
        "color": get_cmap_ix(7, "tab20"),
        "marker": ">"
    },
    7: {
        "description": "Mediterranean-like",
        "label": "MTEM",
        "category": "TEMP",
        "counts": 159,
        "color": get_cmap_ix(17, "tab20"),
        "marker": "p"
    },
}
,
        "k_8_adjusted": {
            16:{"description": "Baltic Sea", "label": "BALT", "category": "BALT", "counts": 51, "color": get_cmap_ix_tab20c(16)},
            11:{"description": "Pacific Equatorial Divergence/Countercurrent", "label": "PEQD", "category": "TROP", "counts": 54, "color": get_cmap_ix_tab20c(2) },
            5:{"description": "Tropical Convergence", "label": "TCON", "category": "TROP", "counts": 161, "color": get_cmap_ix_tab20c(7) },
            9:{"description": "Broad Tropical", "label": "TROP", "category": "TROP", "counts": 818, "color": get_cmap_ix_tab20c(0) },
            0:{"description": "Antarctic Polar", "label": "APLR", "category": "POLR", "counts": 30, "color": get_cmap_ix_tab20c(13) },
            14:{"description": "Arctic Polar", "label": "BPLR", "category": "POLR", "counts": 42, "color": get_cmap_ix_tab20c(12) },
            10:{"description": "Upwelling Areas", "label": "UPWL", "category": "TEMP", "counts": 139, "color": get_cmap_ix_tab20c(8) },
            7:{"description": "Temperate", "label": "TEMP", "category": "TEMP", "counts": 159, "color": get_cmap_ix_tab20c(4) }
    },   
        "k_8_richter_upwl_balt_polar": {
            16:{"description": "Baltic Sea", "label": "BALT", "category": "BALT", "counts": 51, "color": get_cmap_ix(10, "tab20")},
            11:{"description": "Pacific Equatorial Divergence/Countercurrent", "label": "PEQD", "category": "TROP", "counts": 54, "color": get_cmap_ix(8, "tab20") },
            5:{"description": "Tropical Convergence", "label": "TCON", "category": "TROP", "counts": 161, "color": get_cmap_ix(1, "tab20") },
            9:{"description": "Broad Tropical", "label": "TROP", "category": "TROP", "counts": 818, "color": get_cmap_ix(0, "tab20") },
            0:{"description": "Antarctic Polar", "label": "APLR", "category": "POLR", "counts": 30, "color": get_cmap_ix(15, "tab20") },
            14:{"description": "Arctic Polar", "label": "BPLR", "category": "POLR", "counts": 42, "color": get_cmap_ix(14, "tab20") },
            10:{"description": "Upwelling Areas", "label": "UPWL", "category": "TEMP", "counts": 139, "color": get_cmap_ix(4, "tab20") },
            7:{"description": "Mediterranean", "label": "MEDI", "category": "TEMP", "counts": 82, "color": get_cmap_ix(6, "tab20") }
    }
}


# source: [[round(i / 255, 3) for i in c] for c in cartocolors.qualitative.get_map("Prism_9").colors[1:]]
old_default_palette = {"1": [0.114, 0.412, 0.588],  # Blue
                   "2": [0.22, 0.651, 0.647],   # Teal
                   "3": [0.059, 0.522, 0.329],  # Green
                   "4": [0.58, 0.204, 0.431],   # Purple
                   "5": [0.451, 0.686, 0.282],  # Lime
                   "6": [0.929, 0.678, 0.031],  # Yellow
                   "7": [0.882, 0.486, 0.02],   # Orange
                   "8": [0.8, 0.314, 0.243],}   # Red


# -----------------------------
# Command-line entry point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot global samples from a CSV on a map.")
    parser.add_argument("-i", "--csv", required=True, help="Path to your metadata CSV")
    parser.add_argument("-c", "--color", default="Prov",
                        help="Column name to color by (e.g., Prov, depth_cat, biogeographical_province)")
    parser.add_argument("-p", "--palette", default="k_8_adjusted",
                        help="Palette name in `palettes` to use (k_8_adjusted, k_8_richter_upwl_balt_polar, k_10)")
    parser.add_argument("--no-palette", action="store_true",
                        help="Ignore preset palettes and use a matplotlib cmap instead")
    parser.add_argument("--jitter", type=float, default=0.05, help="Coordinate jitter amount")
    parser.add_argument("--drop-dup", action="store_true", help="Drop duplicate coordinates")
    args = parser.parse_args()

    # 1) Read CSV
    df = pd.read_csv(args.csv)
    
    # 2) Clean coordinates & build coords tuple
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # Drop rows missing lat/lon or out of range
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    df["coords"] = list(zip(df["latitude"], df["longitude"]))

    # Also drop rows missing the color column
    if args.color not in df.columns:
        raise ValueError(f"Color column '{args.color}' not found in CSV.")
    df = df.dropna(subset=[args.color])

    # 3) Decide palette usage
    palette = None
    if not args.no_palette:
        try:
            pal = palettes[args.palette]  # your predefined palettes
            vals = pd.unique(df[args.color].dropna())
            # Detect numeric vs label
            is_all_numeric = pd.to_numeric(pd.Series(vals), errors="coerce").notna().all()
            if is_all_numeric:
                # Column holds numeric province codes: use {code: color}
                df[args.color] = pd.to_numeric(df[args.color], errors="coerce").astype("Int64")
                palette = {k: v["color"] for k, v in pal.items()}
                df = df[df[args.color].isin(palette.keys())].copy()
            else:
                # Column holds labels like APLR/BPLR: use {label: color}
                df[args.color] = df[args.color].astype(str).str.strip()
                palette = {v["label"]: v["color"] for v in pal.values()}
                unknown = sorted(set(vals) - set(palette.keys()))
                if unknown:
                    print("These categories were not found in the selected palette and will be ignored:", unknown)
                df = df[df[args.color].isin(palette.keys())].copy()
        except Exception as e:
            print(f"Could not use the preset palette '{args.palette}'. Falling back to automatic cmap. Reason: {e}")
            palette = None

    # 4) Plot
    kwargs = dict(
        sample_metadata=df,
        color_category=args.color,
        jitter=args.jitter,
        legend=True,
        title=f"Map colored by {args.color}",
        drop_duplicate_coords=args.drop_dup,
    )
    if palette is not None:
        kwargs["palette"] = palette
    else:
        kwargs["cmap"] = "tab20c"

    ax = plot_colored_markers(**kwargs)

    # 5) Save & show
    out_path = f"map_by_{args.color}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    plt.show()