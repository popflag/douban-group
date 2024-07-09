import json
import os

import dotenv
from openai import OpenAI
from pydantic import BaseModel

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BACKEND_URL = os.getenv('OPENAI_BACKEND_URL')

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BACKEND_URL)


class HouseSchema(BaseModel):
    houseType: str
    hostPrice: int
    cashPledge: str
    publicTransport: str
    detailedAddress: str
    requirements: str


SYSTEM_PROMPT = f'''## 工作内容
你是一个租房机器人，帮助用户从自然语言的文本中提取出所需的关键信息，必须使用JSON格式返回，不包含注释。

## 参考输出格式
### json schema
{HouseSchema.model_json_schema()}

### example
{{
    houseType: "公寓", // 房源类型: [公寓 | 小区 | 城中村 | ...]
    hostPrice: 2700, // 价格
    cashPledge: "押二付一", // 押金方式：[押二付一 | 押一付一 | ...]
    publicTransport: "地铁：距离水贝站仅八分钟路程", // 最近的公共交通距离
    detailedAddress: ""， // 具体的地址，去除正文中为了SEO所插入的关联关键词
    requirements: "不能养宠物", // 基本要求: [限女生 | 不能养宠物 | ...]
}}'''


def extract_json(text: str) -> str:
    """
    从大模型的返回中提取JSON
    """
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_content = text[json_start:json_end].replace("\\_", "_")
        return json_content
    except Exception as e:
        return f"Error extracting JSON: {e}"


def parse_detail_with_ai(context: str) -> dict:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]
    )

    message = completion.choices[0].message.content
    house_parse = HouseSchema.model_validate_json(extract_json(message))
    json_item = house_parse.model_dump_json()
    return json.loads(json_item)


if __name__ == '__main__':
    context = '''
龙华6号线阳台山东 小区单间短租2000 
    
个人转租短租8-10月份，后续根据个人时间安排可再租，适用时间自由的，个人补贴300房租2000包网费（原价199），物业费126，房子位于恒大时尚慧谷，小区里有公交车首末站两站直达地铁站，随时可以看房，帖子没删就是还在

电梯8楼，电费0.7，目前小区不收水费，押1.5付1

密码锁，厨房大配套电磁炉、油烟机、消毒碗柜，收纳多

大单间+朝东南光线特别好，两层窗帘拉了窗帘遮光效果好

配置：220L容量大冰箱+海尔洗衣机带烘干+茶几+小米投影仪+空调+沙发+转折梳妆台+真皮大床1.5米+油烟机+宜家大衣柜，卫浴干湿分离+坐厕

没删就是在随时看房#短租[话题]# #个人转租[话题]# ##转租[话题]# #龙华租房[话题]#'''
    print(parse_detail_with_ai(context))
