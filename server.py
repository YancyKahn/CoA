import uvicorn
from fastapi import FastAPI, StreamingResponse
import json
import io
import argparse

app = FastAPI()

@app.get("/stream_json")
def stream_json_response():
    # 这里是一个生成器函数，生成一些 JSON 对象
    def generate():
        for i in range(3):
            data = {"index": i, "message": f"Message {i}"}
            yield json.dumps(data) + "\n"
    
    # 使用 StreamingResponse 将生成器函数的输出流式发送到客户端
    return StreamingResponse(io.TextIOWrapper(io.BytesIO("".join(generate()).encode())), media_type="application/json")


if __name__ == "__main__":
    pass
