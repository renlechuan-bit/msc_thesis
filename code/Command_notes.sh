ssh spartan-login2.hpc.unimelb.edu.au

sinteractive -p cascade,interactive,sapphire --time=12:00:00 --cpus-per-task=1 --mem=16G --account=punim2504
conda activate L2UniFrac

python scripts/get_16s_L2UniFrac_average.py -i mydata/ps_rel.biom -m mydata/metadata_forL2.txt -k SRA -v IHO_Sea_fmt -o mydata/representative_otu_by_IHO_Sea_fmt.tsv 


python scripts/get_pairwise_L2_unifrac.py -i mydata/otu_table_L2_extended.tsv -t mydata/tree_L2.nwk -o mydata/L2_extend_out_unifrac.tsv
python scripts/get_pairwise_L2_unifrac.py -i mydata/otu_table_L2_extended.tsv -t mydata/tree_L2.nwk -o mydata/L1_extend_out_unifrac.tsv -L1 1
python scripts/compare_L1_L2.py -f1 mydata/L1_extend_out_unifrac.tsv -f2 mydata/L2_extend_out_unifrac.tsv -s mydata/L1_L2_plot.png

python scripts/_split_dataframe.py \
    -f /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/otu/otu_extended.biom \
    -m /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/metadata_forL2.txt \
    -e IHO_Sea_fmt \
    -o /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe \
    -t /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk



# Test one plot
python "/data/gpfs/projects/punim2504/msc_thesis/code/L2_UniFrac_wei/L2-UniFrac-Paper/scripts/plot_df.py" \
    -t 2Dpcoa \
    -f "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2/pairwise_Arabian Sea.tsv" \
    -m "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe/Arabian Sea_and_representative_meta.tsv" \
    -env environment \
    -s "/data/gpfs/projects/punim2504/msc_thesis/plots/Arabian Sea_and_representative_pcoa.png" \
    -cmap coolwarm


sbatch run_split_pairwise.sh
sbatch run_split_pcoa.sh


python get_wgs_diffabund_from_otu.py \
  -i /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/otu/otu_table_L2_extended.tsv \
  -t /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk \
  -m /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/metadata_forL2.txt \
  --id_col sample_name \
  -p IHO_Sea_fmt \
  -s /data/gpfs/projects/punim2504/msc_thesis/plots/diffab_plots \
  --prefix IHO \
  --thresh 5e-4 \
  --max_disp 50
    


python scripts/_split_dataframe.py \
    -f /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/otu/otu_extended.biom \
    -m /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/metadata_IHO.txt \
    -e IHO_Sea_comb \
    -o /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe_supplementary \
    -t /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk \
    --env_filter "North Atlantic Ocean" "South Pacific Ocean"


# North Atlantic Ocean
python scripts/get_pairwise_L2_unifrac.py \
    -i "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe_supplementary/North Atlantic Ocean_and_representative.tsv" \
    -t "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk" \
    -o "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2_supp/pairwise_North Atlantic Ocean.tsv"


python scripts/get_pairwise_L2_unifrac.py \
    -i "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe_supplementary/South Pacific Ocean_and_representative.tsv" \
    -t "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk" \
    -o "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2_supp/pairwise_South Pacific Ocean.tsv"

python scripts/plot_df.py \
    -t 2Dpcoa \
    -f "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2_supp/pairwise_North Atlantic Ocean.tsv" \
    -m "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe_supplementary/North Atlantic Ocean_and_representative_meta.tsv" \
    -env environment \
    -s "/data/gpfs/projects/punim2504/msc_thesis/plots/plots_supp/North Atlantic Ocean_and_representative_pcoa.png" \
    -cmap coolwarm

python scripts/plot_df.py \
    -t 2Dpcoa \
    -f "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_L2_supp/pairwise_South Pacific Ocean.tsv" \
    -m "/data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/split_dataframe_supplementary/South Pacific Ocean_and_representative_meta.tsv" \
    -env environment \
    -s "/data/gpfs/projects/punim2504/msc_thesis/plots/plots_supp/South Pacific Ocean_and_representative_pcoa.png" \
    -cmap coolwarm



python get_wgs_diffabund_from_otu.py \
  -i /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/otu/otu_table_L2_extended.tsv \
  -t /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/tree_L2.nwk \
  -m /data/gpfs/projects/punim2504/msc_thesis/data/unifrac_L2_data/metadata_IHO.txt \
  --id_col sample_name \
  -p IHO_Sea_comb \
  -s /data/gpfs/projects/punim2504/msc_thesis/plots/diffab_plots \
  --prefix diffab \
  --thresh 5e-4 \
  --max_disp 50 \
  --compare "North Pacific Ocean" "South Pacific Ocean"
