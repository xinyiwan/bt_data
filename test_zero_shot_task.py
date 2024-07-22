import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from functools import partial
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import sys, os
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import Logger
from typing import List, Tuple
from collections import OrderedDict
from pathlib import Path
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split



sys.path.insert(1, '/trinity/home/xwan/bt_data/src')
from src import utils

sys.path.insert(1, '/trinity/home/xwan/bt_data/src/MONET')
import MONET
from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_pkl
from MONET.utils.text_processing import (
    generate_prompt_token_from_caption,
    generate_prompt_token_from_concept,
)
from MONET.utils.loader import custom_collate



with initialize(version_base=None, config_path="configs", job_name="test_app"):
    cfg = compose(config_name="eval")
    cfg.model.net.device='cpu'
    cfg.ckpt_path = '/trinity/home/xwan/bt_data/logs/train/runs/2024-07-03_17-08-32/checkpoints/epoch_002.ckpt'
    cfg.datamodule.split_seed = 42
    cfg.paths.output_dir = '/trinity/home/xwan/bt_data/logs/eval/runs/test'
    cfg.datamodule.batch_size_test = 1


# print(OmegaConf.to_yaml(cfg))
print(cfg.model)
model: LightningModule = hydra.utils.instantiate(cfg.model)
datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
# datamodule.setup()



data_dir = '/data/archive/xwan/bt_pmd/pubmed'
n_px: int = 224
norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
split_seed = 42

def setup_pubmed_predict_data_loader(data_dir, n_px, norm_mean, norm_std, split_seed):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[os.path.splitext(k)[0]] = str(
            Path(data_dir) / "final_image" / v
        )  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_pickle(Path(data_dir) / "final_text.pkl")

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    # set train dataset
    data_predict = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, ["article_id", "href"]],
        integrity_level="weak",
    )
    # add text data
    data_predict.text_data = text_df.loc[val_idx]

    data_predict.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="caption_final",
        use_random=False,
    )

    predict_loader = DataLoader(
            dataset=data_predict,            
            batch_size=10,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            shuffle=True,
            collate_fn=custom_collate,
        )
    
    return data_predict, predict_loader


predict, predict_loader = setup_pubmed_predict_data_loader(data_dir, n_px, norm_mean, norm_std, split_seed)
# give back loss function of all
a = trainer.predict(model=model, dataloaders=predict_loader, ckpt_path=cfg.ckpt_path, return_predictions=True)

print(a)
# import MONET
# from models.contrastive_module import ContrastiveLitModule
# import MONET.models.components.image_text_encoder.ImageTextEncoder

# net = ContrastiveLitModule()
# model = LightningTransformer.load_from_checkpoint(PATH)



