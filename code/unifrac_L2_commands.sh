ssh spartan-login2.hpc.unimelb.edu.au

sinteractive -p cascade,interactive,sapphire --time=12:00:00 --cpus-per-task=1 --mem=16G --account=punim2504
conda activate L2UniFrac
python scripts/get_16s_L2UniFrac_average.py -i mydata/ps_rel.biom -m mydata/metadata_forL2.txt -k SRA -v IHO_Sea_fmt -o mydata/representative_otu_by_IHO_Sea_fmt.tsv 
python scripts/get_pairwise_L2_unifrac.py -i mydata/otu_table_L2_extended.tsv -t mydata/tree_L2.nwk -o mydata/L2_extend_out_unifrac.tsv
python scripts/get_pairwise_L2_unifrac.py -i mydata/otu_table_L2_extended.tsv -t mydata/tree_L2.nwk -o mydata/L1_extend_out_unifrac.tsv -L1 1
python scripts/compare_L1_L2.py -f1 mydata/L1_extend_out_unifrac.tsv -f2 mydata/L2_extend_out_unifrac.tsv -s mydata/L1_L2_plot.png

python scripts/_split_dataframe.py -f mydata/otu_table_L2_extended.tsv -m mydata/metadata_forL2.txt -e IHO_Sea_fmt -o mydata

bash run_pairwise_and_pcoa.sh
