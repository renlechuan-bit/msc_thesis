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


IHO_Sea_fmt_palette = {
    'Southern Ocean': '#0022ff',
    'South Pacific Ocean': '#000000',
    'South Atlantic Ocean': '#000000',
    'Indian Ocean': '#000000',
    'Coral Sea': '#000000',
    'North Pacific Ocean': '#000000',
    'North Atlantic Ocean': '#000000',
    'Bay of Bengal': '#000000',
    'Arabian Sea': '#000000',
    'Red Sea': '#000000',
    'Mediterranean Sea': '#000000',
    'Celtic Sea': '#000000',
    'North Sea': '#000000',
    'Baltic Sea': '#000000',
    'Arctic Ocean': '#000000'
 }


def plot_colored_markers(sample_metadata, color_category, cmap="tab20c", jitter=0.05, legend=True, title=None, drop_duplicate_coords=True, custom_palette=None):
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

    if custom_palette is None:
        # Create a continuous color map
        cmap = plt.get_cmap(cmap)
        color_dict = {category: cmap(i) for i, category in enumerate(sample_metadata[color_category].unique())}
    else:
        color_dict = custom_palette

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


# -----------------------------
# Command-line entry point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot global samples from a CSV on a map.")
    parser.add_argument("-i", "--csv", required=True, help="Path to your metadata CSV")
    parser.add_argument("-c", "--color", default="IHO_Sea_fmt",
                        help="Column name to color by (e.g., Prov, depth_cat, biogeographical_province)")
    parser.add_argument("--no-palette", action="store_true",
                        help="Ignore preset palettes and use a matplotlib cmap instead")
    parser.add_argument("--jitter", type=float, default=0.05, help="Coordinate jitter amount")
    parser.add_argument("--cmap", type=str, default="tab20c")
    parser.add_argument("--drop-dup", action="store_true", help="Drop duplicate coordinates")
    parser.add_argument("-o", "--outdir", help="Path to output dir", default=None)
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

    # 4) Plot
    kwargs = dict(
        sample_metadata=df,
        cmap=args.cmap,
        color_category=args.color,
        jitter=args.jitter,
        legend=True,
        title=f"Map colored by {args.color}",
        drop_duplicate_coords=args.drop_dup,
        # custom_palette=IHO_Sea_fmt_palette
    )

    ax = plot_colored_markers(**kwargs)

    # 5) Save & show
    if args.outdir is None:
        out_path = f"map_by_{args.color}.png"
    else:
        out_path = f"{args.outdir}/map_by_{args.color}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    plt.show()