import sys
import pandas as pd
sys.path.insert(1, '/Users/xinyi/Documents/GitHub/bt_data/src')
from pubmed_download import download_and_extract_article

hostname="ftp.ncbi.nlm.nih.gov"
local_dir = "data/pubmed/"
include_extension_list = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif', '.bmp', '.xml', '.nxml']


path_input = '/Users/xinyi/Documents/GitHub/bt_data/tmp_data/oa_file_list_onlykeywords.csv'
oa_file_list = pd.read_csv(path_input, index_col=0)
file_list = ("pub/pmc/" + oa_file_list["File"]).tolist()

file_list_split = [
                file_list[i * (16) : min((i + 1) * (16), len(file_list))]
                for i in range(len(file_list) // 16 + 1)
            ]

key = 'pub/pmc/oa_package/2e/64/PMC11126675.tar.gz'
print(len(file_list_split))

for i in range(len(file_list_split)):
    if key in file_list_split[i]:
        print('yes')
        print(file_list_split[i])
        break

# find i = 1
# print(file_list_split[i])
download_and_extract_article(
    file_list=file_list_split[i],
    hostname=hostname,
    local_dir=local_dir,
    include_extension_list=include_extension_list,
    blocksize=33554,
    use_pbar=False
)

