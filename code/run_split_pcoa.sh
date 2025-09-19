#!/bin/bash
#SBATCH --job-name="run_pcoa"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --account=punim2504
#SBATCH --partition=sapphire
#SBATCH --mail-user=lechuanr@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh" ]; then
        . "/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh"
    else
        export PATH="/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate L2UniFrac
 
BASE_DIR="/data/gpfs/projects/punim2504/msc_thesis/code/L2_UniFrac_wei/L2-UniFrac-Paper"
cd "$BASE_DIR" || { echo "❌ Cannot access directory $BASE_DIR"; exit 1; }
export PYTHONPATH="$BASE_DIR/L2-UniFrac:$BASE_DIR/L2-UniFrac/src:$BASE_DIR/scripts:$BASE_DIR/src:$PYTHONPATH"


INPUT_DIR="/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2"
OUTPUT_DIR="/data/gpfs/projects/punim2504/msc_thesis/plots/split_2Dpcoa"
TREE_FILE="/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk"   
META_DIR="/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe"
mkdir -p "$OUTPUT_DIR"


for tsv in "${INPUT_DIR}"/pairwise_*.tsv; do
    group=$(basename "$tsv" .tsv)
    group_name="${group#pairwise_}"

    echo "=== Processing group: $group ==="

    # Corresponding meta file
    meta_file="${META_DIR}/${group_name}_and_representative_meta.tsv"
    out_png="${OUTPUT_DIR}/${group_name}_and_representative_pcoa.png"


    # Draw PCoA plot
    python "${BASE_DIR}/scripts/plot_df.py" \
        -t 2Dpcoa \
        -f "$tsv" \
        -m "$meta_file" \
        -env environment \
        -s "$out_png" \
        -cmap coolwarm
done