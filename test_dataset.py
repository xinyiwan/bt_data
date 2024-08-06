import io
import os, sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn

sys.path.insert(1, '/trinity/home/xwan/bt_data/src')
from src import utils

sys.path.insert(1, '/trinity/home/xwan/bt_data/src/MONET')
from MONET.utils.loader import custom_collate
from MONET.datamodules.components.base_dataset import BaseDataset, convert_image_to_rgb


base_dataset = BaseDataset(
    image_path_or_binary_dict={
        1:"/data/archive/xwan/bt_pmd/test/test_img1.jpg",
        2:"/data/archive/xwan/bt_pmd/test/test_img2.jpg",
    },
    n_px=224,
    norm_mean=(0.48145466, 0.4578275, 0.40821073),
    norm_std=(0.26862954, 0.26130258, 0.27577711),
    augment=False,
    metadata_all=pd.DataFrame(index=[1, 2]),
)

print(base_dataset[0])
print(base_dataset[1])
loader = torch.utils.data.DataLoader(
    base_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate
)
print(next(iter(loader)))
print(next(iter(loader)))
