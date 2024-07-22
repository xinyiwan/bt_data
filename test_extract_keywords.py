import argparse
import glob
import os, sys
import shutil
from collections import OrderedDict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tqdm




path_input = Path('/data/archive/xwan/bt_pmd/pubmed/final_text.pkl')
with open(path_input, "rb") as f:
    pic_text = pickle.load(f)

pic_text['caption_t'] = pic_text.caption_final.str[0]
count = pic_text.caption_t.str.split(expand=True).stack().value_counts()
count.to_csv("/trinity/home/xwan/bt_data/pubmed/test_text_count.csv")


for idx in pic_text.index:
    cap = pic_text.loc[idx, 'caption_final'][0]
    cap_n = list(set(cap.split(" ")))
    cap_n = " ".join(cap_n)
    pic_text.loc[idx, 'cap_non_repeat'] = cap_n

count_n = pic_text.cap_non_repeat.str.split(expand=True).stack().value_counts()
count_n.to_csv("/trinity/home/xwan/bt_data/pubmed/test_text_count_paper_wise.csv")

