import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import pandas as pd
from typing import List, Tuple
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import AutoModel
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
import hydra
from functools import partial
from omegaconf import DictConfig
from src import eval, utils
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger
from MONET.utils.metrics import skincon_calcualte_auc_all
from collections import OrderedDict
import sys, os
from pathlib import Path
from MONET.utils.io import load_pkl
from MONET.utils.text_processing import (
    generate_prompt_token_from_caption,
    generate_prompt_token_from_concept,
)
from sklearn.model_selection import train_test_split
from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.loader import custom_collate


log = utils.get_pylogger(__name__)

# @utils.task_wrapper
def zero_shot_task(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoints on the prompts

    Args:
        cfg (DictConfig): Configuration composed by Hydra. 

    Returns:
        Dict: Dict with probs according to the given prompts.
    """
    cfg.ckpt_path = cfg.ckpt_path if cfg.ckpt_path is not None else None

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # model.load_from_checkpoint(cfg.ckpt_path)
    # print(model.net.summary)

    data_dir = cfg.datamodule.data_dir
    n_px: int = 224
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    split_seed = cfg.datamodule.split_seed
    batch_size = cfg.datamodule.batch_size_test
    num_workers = cfg.datamodule.num_workers
    predict, predict_loader = setup_pubmed_predict_data_loader(data_dir, n_px, norm_mean, norm_std, split_seed, batch_size, num_workers)

    res = trainer.predict(model=model, dataloaders=predict_loader, ckpt_path=cfg.ckpt_path)
    
    return res 
    


def setup_pubmed_predict_data_loader(data_dir, n_px, norm_mean, norm_std, split_seed, batch_size, num_workers):
    data_dir = Path(data_dir) / "pubmed"
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
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
            shuffle=True,
            collate_fn=custom_collate,
        )
    
    return data_predict, predict_loader

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    res = zero_shot_task(cfg)
    print(res[0].shape)
    

if __name__ == "__main__":
    main()
