import sys
import pandas as pd
sys.path.insert(1, '/Users/xinyi/Documents/GitHub/bt_data/src')
from pubmed_download import check_and_setup_ftp_connection, download_and_extract_article

hostname="ftp.ncbi.nlm.nih.gov"
ftp = check_and_setup_ftp_connection(hostname)
ftp = check_and_setup_ftp_connection(hostname, ftp=ftp)
local_path = 'data/pubmed/pub/pmc/oa_package/2e/64/PMC11126675.tar.gz'
blocksize = 33554
file = 'pub/pmc/oa_package/ee/ca/PMC10680002.tar.gz'
with open(str(local_path) + ".download", "w+b") as f:
    res = ftp.retrbinary("RETR %s" % file, f.write, blocksize=blocksize)
