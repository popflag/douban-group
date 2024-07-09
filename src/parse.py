import json
import logging
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup


def parse_list(html) -> list[dict[str, Any]]:
    """
    解析帖子列表
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select('table[class="olt"] tr[class=""]')
    posts = []
    for row in rows:
        link = row.select_one('td[class="title"] a')
        time_text = row.select_one('td[class="time"]').get_text()
        if re.match(r"\d{2}-\d{2} \d{2}:\d{2}", time_text):
            time_text = str(datetime.now().year) + "-" + time_text
        elif re.match(r"\d{4}-\d{2}-\d{2}", time_text):
            # 跨年的日期，豆瓣显示的是 年-月-日
            time_text += " 00:00"
        r_count = row.select_one('td[class="r-count"]').get_text()
        author = row.select_one("td:nth-child(2) a")
        posts.append(
            {
                "title": link["title"],
                "url": link["href"],
                "reply_count": int(r_count) if r_count else 0,
                "time": datetime.strptime(time_text, "%Y-%m-%d %H:%M"),
                "author": {"name": author.get_text(), "url": author["href"]},
            }
        )
    return posts


def parse_detail(html: str) -> dict[str, Any]:
    """
    解析帖子详情
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.h1.get_text(strip=True)
    content = soup.find("div", class_="topic-richtext").get_text("\n", True)
    author = soup.select_one("#topic-content h3 a")
    create_time = soup.select_one("#topic-content h3 span.create-time").get_text()
    rent = extract_rent(title + "\n" + content)
    return {
        "title": title,
        "rent": rent,
        "create_time": datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S"),
        "content": content,
        "author": {"name": author.get_text(), "url": author["href"]},
    }


def extract_rent(text: str) -> int:
    """
    从文本中提取租金
    """
    # 连续3-5位数字及前后2个字符的内容
    it = re.finditer(r"(.[\s\D])(\d{4})([\s\D].?)", text)
    for match in it:
        # 过滤干扰项
        if re.match(r"押金|补贴", match.group(1)):
            continue
        if re.match(r"米|年", match.group(3)):
            continue
        return int(match.group(2))
    logging.warning("租金提取失败\n%s", text)
    return 0
