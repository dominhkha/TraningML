import requests
import config
from bs4 import BeautifulSoup
from collections import Counter
import json
if __name__=="__main__":
    url = "https://vnexpress.net/phap-luat"
    content = requests.get(url)
    soup = BeautifulSoup(content.content,"html.parser")
    a_list = soup.find_all("a")
    content = ""
    for i in a_list:
        if i.has_attr('title'):
            a = i['title'].strip().lower()+u""
            content+=a.strip()

    f = open("stopword_vietnamese.txt",encoding='utf-8')
    stop_words = f.readlines()
    stop_words = [(i+u"").strip() for i in stop_words]
    refactor = ""
    refactor = [i if i not in stop_words else "no" for i in content.split(" ")]
    count = Counter(refactor)
    print(count.most_common(20))










