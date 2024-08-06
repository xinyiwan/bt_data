import sys
import random
from collections import OrderedDict

import nltk
import numpy as np
import pandas as pd

import clip


sys.path.insert(1, '/trinity/home/xwan/bt_data/src')
from MONET.utils.static import concept_to_prompt
from MONET.utils.text_processing import str_to_token


caption_str = "This image is a MRI."
a, b = str_to_token(caption_str, use_random=True)
print(a)
print(b)
