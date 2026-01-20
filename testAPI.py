from openai import OpenAI

client = OpenAI(
base_url='https://aigc.efunds.com.cn/v1',
api_key='sk-RQir7DACSOuWlgzd2722A716945a473b82316c1d3e69B047',
) 
# # test basic llm api

# resp = client.chat.completions.create(
#     model='EFundGPT-air',
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你好"},
#     ],
#     temperature=1.0,
#     stream=False,
#     extra_headers={
#         "Efunds-User-Name": "SX-{wuyanjie}",
#         "Efunds-Acc-Token": "SX-{wuyanjie}",
#         "Efunds-Source": "2025-SX",
#     },
# ) 
# print(resp.choices[0].message.content)

# test vlm api
import base64
 
with open("cat.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
 
resp = client.chat.completions.create(
    model="EFundGPT-vl-air",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述图片"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            }
        ]
    }],
    extra_headers={
    "Efunds-User-Name": "SX-{wuyanjie}",
    "Efunds-Acc-Token": "SX-{wuyanjie}",
    "Efunds-Source": "2025-SX",
    },
)
print(resp.choices[0].message.content)