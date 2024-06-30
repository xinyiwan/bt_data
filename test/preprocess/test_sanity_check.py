import sys, os
from PIL import Image
import io
import torchvision.transforms as T
from torch import nn
import numpy

sys.path.insert(1, '/Users/xinyi/Documents/GitHub/bt_data/src')
from image_sanity_check import sanity_check_image, convert_image_to_rgb
from utils.io import load_hdf5, load_pkl, save_to_hdf5, save_to_pkl


n_px=224,
norm_mean=(0.48145466, 0.4578275, 0.40821073),
norm_std=(0.26862954, 0.26130258, 0.27577711)
transforms = T.Compose(
        [
            T.RandomResizedCrop(
                size=n_px,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomVerticalFlip(p=0.5),
            T.RandomHorizontalFlip(p=0.3),
            # T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)]), p=0.5),
            convert_image_to_rgb,
            T.RandomApply(
                nn.ModuleList(
                    [
                        T.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.1,
                            hue=0.0,
                        )
                    ]
                ),
                p=1.0,
            ),
            # convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ]
    )
path_input = '/Users/xinyi/Documents/GitHub/bt_data/data/pubmed/glob/images.pkl'
field = 'images'
data_dict = load_pkl(path_input, field, verbose=True)
# success_key_list, failure_key_list = sanity_check_image(data_dict)
l = list(data_dict.keys())

image = Image.open(data_dict[l[0]])
image = transforms(image)
