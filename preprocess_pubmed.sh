#!/bin/bash

#SBATCH --ntasks=1          ### How many CPU cores do you need?
#SBATCH --mem=50G           ### How much RAM memory do you need?
#SBATCH -p short            ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1        ### How many GPUs do you need?
#SBATCH -t 2-00:00:00       ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/xwan/bt_data/logs/out_%j.log
#SBATCH -e /trinity/home/xwan/bt_data/logs/error_%j.log
#SBATCH --job-name=pmcxy	    ### Name your job so you can distinguish between jobs


module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

source "/trinity/home/xwan/bt_data/.venv/bin/activate"

#######################################
# PubMed OA
#######################################

# 1. prepare files
# 1.1 search term
# python src/pubmed_search.py \
# --query-file pubmed/pubmed_query.csv \
# --start-year 2023 \
# --end-year 2024 \
# --output tmp_data \
# --thread 2

# # 1.2 filter with OA dataset
# python src/pubmed_download.py filter \
# --input /Users/xinyi/Documents/GitHub/bt_data/tmp_data/all.csv \
# --output /Users/xinyi/Documents/GitHub/bt_data/tmp_data/oa_file_list_onlykeywords.csv


# # 1.3 download dataset
# python src/pubmed_download.py download \
# --input tmp_data/oa_file_list_onlykeywords.csv \
# --output data/pubmed/ \
# --extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp .xml,.nxml \
# --thread 1

# 2. quality control
# 2.1 globbing files
# python src/glob_files.py \
# --input /data/archive/xwan/bt_pmd/pub \
# --output /data/archive/xwan/bt_pmd/glob/images.pkl \
# --field images \
# --style slash_to_underscore \
# --extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

# 2.2 image sanity check
# taskset -c 0,1,2,3,4,5,6,7 python src/image_sanity_check.py \
# --input /data/archive/xwan/bt_pmd/glob/images.pkl \
# --output /data/archive/xwan/bt_pmd/glob/images.uncorrupted.pkl \
# --field images

# featurize images
############## DIVIDE ##############
# python src/divide.py \
# --input /data/archive/xwan/bt_pmd/glob/images.uncorrupted.pkl \
# --output /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.pkl \
# --field images \
# --num 10

# for i in {0..9}
# do
#     python src/featurize.py \
#     --input /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided."$i".pkl \
#     --output /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided."$i".featurized.pt \
#     --device cuda;
# done


# python src/merge_files.py \
# --input \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.0.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.1.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.2.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.3.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.4.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.5.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.6.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.7.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.8.featurized.pt \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.divided.9.featurized.pt \
# --field efficientnet_feature \
# --output \
# /data/archive/xwan/bt_pmd/glob/images.uncorrupted.unordered.featurized.pt
############## DIVIDE ##############

# cluster images
# python src/cluster.py \
# --input /data/archive/xwan/bt_pmd/glob/images.uncorrupted.pkl \
# --featurized-file /data/archive/xwan/bt_pmd/glob/images.uncorrupted.unordered.featurized.pt \
# --output /data/archive/xwan/bt_pmd/glob/images.uncorrupted.clustering.efficientnet.pca \
# --pca \
# --feature-to-use efficientnet \
# -n1 20 -n2 20


# python src/filter.py \
# --input /data/archive/xwan/bt_pmd/glob/images.uncorrupted.pkl \
# --label-file /data/archive/xwan/bt_pmd/glob/images.uncorrupted.clustering.efficientnet.pca/kmeans_label_lower.csv \
# --exclude-label 00 01 03 05 07 08 09 11 12 13 15 16 17 19 \
# --output /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt1.pkl

# python src/cluster.py \
# --input /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt1.pkl \
# --featurized-file /data/archive/xwan/bt_pmd/glob/images.uncorrupted.unordered.featurized.pt \
# --output /data/archive/xwan/bt_pmd/glob/images.uncorrupted.bt1.clustering.efficientnet.pca \
# --pca \
# --feature-to-use efficientnet \
# -n1 20 -n2 20

# python src/filter.py \
# --input /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt1.pkl \
# --label-file /data/archive/xwan/bt_pmd/glob/images.uncorrupted.bt1.clustering.efficientnet.pca/kmeans_label_lower.csv \
# --exclude-label 01 05 11 12 13 00_06 00_07 00_13 00_15 03_00 03_01 03_03 03_07 03_10 03_14 03_17 03_18 03_19 \
# 06_00 06_01 06_02 06_03 06_04 06_05 06_07 06_08 06_09 06_10 06_11 06_12 06_13 06_14 06_16 06_17 06_18 06_19 10_05 10_13 10_19 \
# 15_18 15_16 15_05 \
# 17_01 17_03 17_05 17_09 17_11 17_14 17_16 \
# 18_08 18_13 \
# 19_13 19_14 19_17 \
# --output /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.pkl


# globbing xml files
# python src/glob_files.py \
# --input /data/archive/xwan/bt_pmd/pub/ \
# --output /data/archive/xwan/bt_pmd/glob/xml.pkl \
# --field xml \
# --style slash_to_underscore \
# --extension .xml,.nxml;

# image sanity check again
# taskset -c 0,1,2,3,4,5,6,7 python src/image_sanity_check.py \
# --input /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.pkl \
# --output /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.doublecheck.pkl \
# --field images \
# --relative-path

# # match image and text
# python src/pubmed_match.py \
# --image /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.pkl \
# --xml /data/archive/xwan/bt_pmd/glob/xml.pkl \
# --output /data/archive/xwan/bt_pmd/glob/xml.matched.pkl

# taskset -c 0,1,2,3,4,5,6,7 python src/image_sanity_check.py \
# --input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
# --output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.doublecheck_.pkl \
# --field images \
# --relative-path

# finalize
# cp /data/archive/xwan/bt_pmd/glob/xml.matched.pkl /data/archive/xwan/bt_pmd/glob/final_text.pkl;

# python src/save_as_path.py \
# --input /data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.pkl \
# --field images \
# --output /data/archive/xwan/bt_pmd/glob/final_image

