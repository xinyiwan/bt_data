from selenium import webdriver
from chromedriver_py import binary_path # this will get you the path variable
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--no-sandbox')
# options.binary_location = '/trinity/home/xwan/bt_data/chrome-linux64/chrome'


print(binary_path)
svc = webdriver.ChromeService(executable_path=binary_path)
driver = webdriver.Chrome(options = options, service=svc)

# path = '/trinity/home/xwan/bt_data/chromedriver-linux64/chromedriver'
# path = '/trinity/home/xwan/bt_data/chrome-linux64/chrome'


# service = Service(executable_path=path)

# driver = webdriver.Chromed(options=chrome_options, service=service)
# driver.get("https://google.com")
