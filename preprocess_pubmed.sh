#!/bin/bash

#######################################
# PubMed OA
#######################################

# 1. prepare files
# 1.1 search term
python src/pubmed_search.py \
--query-file pubmed/pubmed_query.csv \
--start-year 1990 \
--end-year 2023 \
--output /data/archive/xwan/bonetumor/pubmed/search_csv \
--thread 1
