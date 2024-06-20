import sys
sys.path.insert(1, '/Users/xinyi/Documents/GitHub/bt_data/src')
from pubmed_search import PubMedDownloader 


download_path = '/Users/xinyi/Documents/GitHub/bt_data/tmp_data'
pubmed_downloader = PubMedDownloader(driver="chrome", download_path=download_path)

pubmed_downloader.search("bone tumour", 2023)
