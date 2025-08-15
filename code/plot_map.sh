# Notes for running command
# These are just notes -- run manually from command line
sinteractive -p cascade,interactive,sapphire --time=12:00:00 --cpus-per-task=1 --mem=16G --account=punim2504
conda create -n plot_map -c conda-forge python=3.10 -y
conda activate plot_map
conda install -c conda-forge cartopy matplotlib pandas numpy scikit-learn shapely pyproj proj geos -y
python plot_map.py -i /home/lechuanr/punim2504/metadata_and_rank_tables_2025-06-11/metadata_1446.csv -c Prov -p k_8_adjusted --drop-dup