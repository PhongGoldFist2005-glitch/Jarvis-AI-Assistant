import requests
from bs4 import BeautifulSoup
import re

url = "https://dienthoaigiakho.vn/tin-cong-nghe/top-100-game-hay-nhat-moi-thoi-dai/"
request = requests.get(url= url)
text = request.text
# print(text)

soup = BeautifulSoup(text, "html.parser")

# Lấy tất cả thẻ a
a_checker = soup.find_all("strong")
for item in a_checker:
    result = re.split(r'[><]', str(item))
    # print(result[2].strip())

with open(r"P:\Program Files\Python313\AI_assistance\Model\craw.txt", "a", encoding= "utf-8") as f:
    for item in a_checker:
        result = re.split(r'[><]', str(item))
        f.write(f"{result[2].strip()}\n")