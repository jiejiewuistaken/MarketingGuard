# test_response_format.py
import json
import os
import traceback
 
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

 
MODELS = [m.strip() for m in os.getenv("TEST_MODELS", "gpt-4o-mini,EFundGPT-vl-air").split(",")]
 
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
 
extra_headers = {}
raw_headers = os.getenv("OPENAI_EXTRA_HEADERS", "").strip()
if raw_headers:
    try:
        extra_headers = json.loads(raw_headers)
    except json.JSONDecodeError:
        print("Invalid OPENAI_EXTRA_HEADERS JSON. Ignoring.")
 
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "TestSchema",
        "schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["ok"],
            "additionalProperties": False,
        },
    },
}
 
messages = [{"role": "user", "content": "返回一个JSON：{ok: true, note: 'ok'}"}]
 
def run_call(label, fn):
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as e:
        print(f"[FAIL] {label} -> {type(e).__name__}: {str(e)[:180]}")
 
for model in MODELS:
    print("\n======", model, "======")
 
    run_call(
        "chat.completions + json_schema",
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=schema,
            extra_headers=extra_headers or None,
        ),
    )
 
    run_call(
        "chat.completions + json_object",
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_headers=extra_headers or None,
        ),
    )
 
    if hasattr(client, "responses") and hasattr(client.responses, "create"):
        run_call(
            "responses.create + json_schema",
            lambda: client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "输出JSON: {ok: true, note: 'ok'}"}],
                    }
                ],
                response_format=schema,
                extra_headers=extra_headers or None,
            ),
        )
    else:
        print("[SKIP] responses.create not available in SDK")