import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import os, sys
from pathlib import Path
import pickle
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import tqdm

sys.path.insert(1, '/trinity/home/xwan/bt_data/src')
from MONET.utils.plotting import stack_images

# initilize configures
with initialize(version_base=None, config_path="configs", job_name="test_app"):
    cfg = compose(config_name="eval")
    cfg.model.net.device='cpu'
    cfg.ckpt_path = '/trinity/home/xwan/bt_data/logs/train/runs/2024-07-03_17-08-32/checkpoints/epoch_002.ckpt'
    cfg.datamodule.split_seed = 42
    cfg.paths.output_dir = '/trinity/home/xwan/bt_data/logs/eval/runs/test'

# initialize datamodule
dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
dm.setup()
loader_val = dm.val_dataloader()

sample_id_list = np.random.RandomState(42).choice(
    dm.data_train.metadata_all.index,
    size=100,
    replace=False,
)
image_list = []
text_list = []
for sample_id in tqdm.tqdm(sample_id_list):
    # print("check", dm.data_train.generate_prompt_token)
    data = dm.data_train.getitem(dm.data_train.sample_id_to_idx(sample_id))
    image_list.append(data["image"])
    text_list.append(data["prompt"])

stack_images(image_list=image_list, text_list=text_list, path="/trinity/home/xwan/bt_data/test/imgs/pubmet_sets.jpg")

# # load text.pkl
# path_input = Path('/data/archive/xwan/bt_pmd/pubmed/final_text.pkl')
# with open(path_input, "rb") as f:
#     text_df = pickle.load(f)

# # turn the list caption into string
# text_df['caption_t'] = text_df.caption_final.str[0]

# # define keywords
# kw = '|'.join(['tumor', 'benign', 'malignant'])

# # find captions with keywords
# text_df['kw'] = text_df['caption_t'].str.contains(kw, case=False, regex=True)
