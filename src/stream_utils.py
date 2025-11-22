"""
流式响应处理工具模块
提供用于假流式响应（Fake Streaming）的通用辅助函数
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, AsyncGenerator, Optional


async def monitor_task_with_heartbeat(
    task: asyncio.Task, heartbeat_data: Dict[str, Any], interval: float = 3.0
) -> AsyncGenerator[bytes, None]:
    """
    监控异步任务，并在等待期间发送SSE心跳数据。

    Args:
        task: 要监控的异步任务
        heartbeat_data: 心跳包数据（字典）
        interval: 心跳间隔（秒）

    Yields:
        SSE格式的心跳数据字符串（bytes）
    """
    try:
        while not task.done():
            await asyncio.sleep(interval)
            if not task.done():
                yield f"data: {json.dumps(heartbeat_data)}\n\n".encode()
    except asyncio.CancelledError:
        # 如果生成器被取消（例如客户端断开连接），则取消底层任务
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise


def decode_response_body(response: Any) -> str:
    """
    通用的响应体解码函数，处理bytes或str类型的body/content
    """
    if hasattr(response, "body"):
        return (
            response.body.decode()
            if isinstance(response.body, bytes)
            else str(response.body)
        )
    elif hasattr(response, "content"):
        return (
            response.content.decode()
            if isinstance(response.content, bytes)
            else str(response.content)
        )
    else:
        return str(response)


def create_openai_chunk(
    delta: Dict[str, Any],
    model: str = "gcli2api-streaming",
    finish_reason: Optional[str] = None,
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """创建OpenAI格式的流式chunk"""
    if not response_id:
        response_id = str(uuid.uuid4())

    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "system_fingerprint": "gcli2api-fake-stream",
    }
    return chunk


def create_openai_error_chunk(
    message: str, model: str = "gcli2api-streaming", response_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建OpenAI格式的错误chunk"""
    return create_openai_chunk(
        delta={"role": "assistant", "content": message},
        model=model,
        finish_reason="stop",
        response_id=response_id,
    )


def create_gemini_error_chunk(message: str, code: int = 500) -> Dict[str, Any]:
    """创建Gemini格式的错误chunk"""
    return {
        "error": {
            "message": message,
            "type": "api_error",
            "code": code,
        }
    }
