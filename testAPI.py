from openai import OpenAI

client = OpenAI(
base_url='https://aigc.efunds.com.cn/v1',
api_key='sk-RQir7DACSOuWlgzd2722A716945a473b82316c1d3e69B047',
) 
resp = client.chat.completions.create(
    model='EFundGPT-air',
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ],
    temperature=1.0,
    stream=False,
    extra_headers={
        "Efunds-User-Name": "SX-{wuyanjie}",
        "Efunds-Acc-Token": "SX-{wuyanjie}",
        "Efunds-Source": "2025-SX",
    },
) 
print(resp.choices[0].message.content)