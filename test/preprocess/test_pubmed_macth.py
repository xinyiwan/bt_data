import os, sys
from pathlib import Path

import tqdm

sys.path.insert(1, '/trinity/home/xwan/bt_data/src')
from utils.io import load_pkl

path_image = Path('/data/archive/xwan/bt_pmd/glob/filtered/images.uncorrupted.bt2.doublecheck.pkl')
path_xml = Path('/data/archive/xwan/bt_pmd/glob/xml.pkl')
path_output = Path('/data/archive/xwan/bt_pmd/glob/xml.matched.pkl')

image_dict = load_pkl(path_input=path_image, field="images", verbose=True)
xml_dict = load_pkl(path_input=path_xml, field="xml", verbose=True)

image_article_key = {"_".join(key.split("_")[:6]) for key in image_dict.keys()}
xml_article_key = {"_".join(key.split("_")[:6]) for key in xml_dict.keys()}
# assert image_article_key.issubset(xml_article_key), "Image article not in xml article"


# for t in image_article_key:
#     if t in xml_article_key:
#         pass
#     else:
#         print("exp")
#         print(t)


image_key_list = [os.path.splitext(key)[0] for key in image_dict.keys()]
import collections
dpc = [item for item, count in collections.Counter(image_key_list).items() if count > 1]

del image_dict['pmc_oa_package_0f_f6_PMC9986859_cureus-0015-00000034618-i05.gif']
image_key_list = [os.path.splitext(key)[0] for key in image_dict.keys()]
print(len(set(image_key_list)) == len(image_dict.keys()))
