import sys, os
import pandas as pd
import shutil
from pathlib import Path

sys.path.insert(1, '/Users/xinyi/Documents/GitHub/bt_data/src')
from pubmed_download import extract_tar, check_and_setup_ftp_connection

hostname="ftp.ncbi.nlm.nih.gov"
include_extension_list = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif', '.bmp', '.xml', '.nxml']
local_dir = 'data/pubmed'
local_dir = Path(local_dir)

# file = 'pub/pmc/oa_package/2e/64/PMC11126675.tar.gz'
file = 'pub/pmc/oa_package/a5/45/PMC10472599.tar.gz'

local_path = local_dir / file
os.makedirs(os.path.dirname(local_path), exist_ok=True)
blocksize = 33554

ftp = check_and_setup_ftp_connection(hostname)


for _ in range(5):
    ftp = check_and_setup_ftp_connection(hostname, ftp=ftp)
    try:
        with open(str(local_path) + ".download", "w+b") as f:
            res = ftp.retrbinary("RETR %s" % file, f.write, blocksize=blocksize)
    except BaseException as e:
        print("error", e)
        print(local_path)
        if os.path.exists(str(local_path) + ".download"):
            os.remove(str(local_path) + ".download")
    else:
        if res.startswith("226 Transfer complete"):
            os.rename(str(local_path) + ".download", local_path)
            break
        else:
            print(f"Downloaded of file {file} is not complete.")
            os.remove(str(local_path) + ".download")
    # except KeyboardInterrupt as e:
    #     os.remove(str(local_path) + ".download")
    #     raise e

# extract files
try:
    extract_tar(
        local_path,
        output_dir=os.path.dirname(local_path),
        include_extension_list=include_extension_list,
    )

except BaseException as e:
    shutil.rmtree(str(local_path).replace(".tar.gz", ""))
    os.remove(local_path)
    # remove tar file
else:
    os.remove(local_path)
