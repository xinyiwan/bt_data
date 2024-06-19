import requests
import wget
import zipfile
import os
import ssl


# ssl._create_default_https_context = ssl._create_unverified_context
# # get the latest chrome driver version number
# url = 'https://chromedriver.storage.googleapis.com/LATEST_RELEASE'
# response = requests.get(url)
# version_number = response.text

# build the donwload url
# download_url = "https://storage.googleapis.com/chrome-for-testing-public/126.0.6478.61/linux64/chrome-linux64.zip"
download_url = "https://storage.googleapis.com/chrome-for-testing-public/126.0.6478.61/linux64/chromedriver-linux64.zip"

# download the zip file using the url built above
latest_driver_zip = wget.download(download_url,'chromedriver')

# extract the zip file
with zipfile.ZipFile(latest_driver_zip, 'r') as zip_ref:
    zip_ref.extractall() # you can specify the destination folder path here
# delete the zip file downloaded above
os.remove(latest_driver_zip)
