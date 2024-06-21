#!/bin/bash

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
# --input data/pubmed \
# --output data/pubmed/glob/images.pkl \
# --field images \
# --style slash_to_underscore \
# --extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

# 2.2 image sanity check
# taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
python src/image_sanity_check.py \
--input data/pubmed/glob/images.pkl \
--output data/pubmed/glob/images.uncorrupted.pkl \
--field images
