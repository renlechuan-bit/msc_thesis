#SBATCH --job-name="your-job-name"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --account=punim2504
#SBATCH --partition=cascade,interactive,sapphire


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

# insert your 'conda activate' command here, for example:
# conda activate plot_map

# insert your code here, for example:
# python plot_map.py -i /home/lechuanr/punim2504/metadata_and_rank_tables_2025-06-11/metadata_1446.csv -c Prov -p k_8_adjusted --drop-dup
