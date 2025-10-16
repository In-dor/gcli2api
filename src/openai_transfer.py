"""
OpenAI Transfer Module - Handles conversion between OpenAI and Gemini API formats
被openai-router调用，负责OpenAI格式与Gemini格式的双向转换
"""

import json
import re
from logging import INFO, info
import time
import uuid
from typing import Dict, Any, Union, List, Optional

from config import (
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    get_thinking_budget,
    is_search_model,
    should_include_thoughts,
)
from log import log
from .models import ChatCompletionRequest, OpenAIChatMessage, OpenAIDelta


async def openai_request_to_gemini_payload(
    openai_request: ChatCompletionRequest,
) -> Dict[str, Any]:
    """
    将OpenAI聊天完成请求转换为完整的Gemini API payload格式。
    此函数经过重构，以正确处理系统消息、工具调用和多模态内容。

    Args:
        openai_request: OpenAI格式请求对象

    Returns:
        完整的Gemini API payload，包含model和request字段
    """
    # 1. 分离系统消息和用户/助手消息
    system_instructions_parts = []
    non_system_messages = []
    for msg in openai_request.messages:
        if msg.role == "system":
            content = msg.content
            if isinstance(content, str):
                system_instructions_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        system_instructions_parts.append(part.get("text", ""))
        else:
            non_system_messages.append(msg)

    # 2. 预处理：构建 tool_call_id 到函数名的映射
    tool_call_map = {}
    for message in non_system_messages:
        if message.role == "assistant" and message.tool_calls:
            for tc in message.tool_calls:
                if tc.get("id") and tc.get("function", {}).get("name"):
                    tool_call_map[tc["id"]] = tc["function"]["name"]

    # 3. 转换消息内容
    contents = []
    for message in non_system_messages:
        role = "model" if message.role == "assistant" else message.role

        # a. 处理工具响应 (tool role)
        if message.role == "tool":
            func_name = tool_call_map.get(message.tool_call_id)
            if not func_name:
                log.warning(
                    f"找不到 tool_call_id '{message.tool_call_id}' 对应的函数名，已跳过此工具响应。"
                )
                continue

            try:
                response_content = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                response_content = {"content": str(message.content)}

            contents.append(
                {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": func_name,
                                "response": response_content,
                            }
                        }
                    ],
                }
            )
            continue

        # b. 处理助手发起的工具调用 (assistant role with tool_calls)
        if message.role == "assistant" and message.tool_calls:
            parts = []
            for tc in message.tool_calls:
                function_details = tc.get("function", {})
                try:
                    args = json.loads(function_details.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                parts.append(
                    {
                        "functionCall": {
                            "name": function_details.get("name"),
                            "args": args,
                        }
                    }
                )
            if parts:
                contents.append({"role": "model", "parts": parts})
            # 如果助手消息只有工具调用，没有文本内容，则处理完后继续下一个消息
            if not message.content:
                continue

        # c. 处理常规文本和图片内容
        gemini_parts = []
        msg_content = message.content
        if isinstance(msg_content, str):
            # New logic to handle mixed text and image content from assistant history
            if message.role == "assistant" and "![image](data:" in msg_content:
                # Use regex to find all markdown images and surrounding text
                pattern = r"!\[image\]\((data:[^)]+)\)"

                last_end = 0
                for match in re.finditer(pattern, msg_content):
                    start, end = match.span()

                    # Add preceding text if any
                    preceding_text = msg_content[last_end:start].strip()
                    if preceding_text:
                        gemini_parts.append({"text": preceding_text})

                    # Add the image part
                    data_uri = match.group(1)
                    try:
                        header, base64_data = data_uri.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        gemini_parts.append(
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data,
                                }
                            }
                        )
                    except (ValueError, IndexError):
                        # If parsing fails, add the raw markdown as text
                        gemini_parts.append({"text": msg_content[start:end]})

                    last_end = end

                # Add any remaining text after the last image
                remaining_text = msg_content[last_end:].strip()
                if remaining_text:
                    gemini_parts.append({"text": remaining_text})
            else:
                # Original logic for simple text content
                if msg_content:
                    gemini_parts.append({"text": msg_content})
        elif isinstance(msg_content, list):
            for part in msg_content:
                if part.get("type") == "text" and part.get("text"):
                    gemini_parts.append({"text": part.get("text")})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if image_url and image_url.startswith("data:"):
                        try:
                            header, base64_data = image_url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            gemini_parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": base64_data,
                                    }
                                }
                            )
                        except (ValueError, IndexError):
                            log.warning(f"无法解析图片数据URI: {image_url[:50]}...")

        if gemini_parts:
            # 合并连续的同角色消息
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(gemini_parts)
            else:
                contents.append({"role": role, "parts": gemini_parts})

    # 4. 构建生成配置 (generationConfig)
    generation_config = {}
    if openai_request.temperature is not None:
        generation_config["temperature"] = openai_request.temperature
    if openai_request.top_p is not None:
        generation_config["topP"] = openai_request.top_p
    if openai_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = openai_request.max_tokens
    if openai_request.stop is not None:
        stop_seq = (
            [openai_request.stop]
            if isinstance(openai_request.stop, str)
            else openai_request.stop
        )
        generation_config["stopSequences"] = stop_seq
    if openai_request.n is not None:
        generation_config["candidateCount"] = openai_request.n
    if (
        openai_request.response_format
        and openai_request.response_format.get("type") == "json_object"
    ):
        generation_config["responseMimeType"] = "application/json"

    # 5. 构建最终请求体
    request_data = {
        "contents": contents,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
    }
    if generation_config:
        request_data["generationConfig"] = generation_config

    if system_instructions_parts:
        request_data["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_instructions_parts)}]
        }

    if openai_request.tools:
        request_data["tools"] = openai_request.tools

    if openai_request.tool_choice:
        tool_choice = openai_request.tool_choice
        if isinstance(tool_choice, str) and tool_choice in ["none", "auto"]:
            request_data["toolConfig"] = {
                "functionCallingConfig": {"mode": tool_choice.upper()}
            }
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            func_name = tool_choice["function"].get("name")
            if func_name:
                request_data["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [func_name],
                    }
                }

    if is_search_model(openai_request.model):
        request_data["tools"] = [{"googleSearch": {}}]

    # 决定是否包含思维链以及其预算
    custom_thinking_setting = getattr(openai_request, "thinking_budget", None)

    include_thoughts_flag = False
    final_thinking_budget = None

    if custom_thinking_setting is not None:
        if isinstance(custom_thinking_setting, bool):
            if custom_thinking_setting is True:
                # 用户传入 "thinking_budget": true
                include_thoughts_flag = True
                # 使用模型名称定义的默认预算
                final_thinking_budget = get_thinking_budget(openai_request.model)
            # 如果是 false，则 include_thoughts_flag 保持 False，禁用思维链
        elif isinstance(custom_thinking_setting, int):
            if custom_thinking_setting > 0:
                # 用户传入了具体的预算值
                include_thoughts_flag = True
                final_thinking_budget = custom_thinking_setting
            # 如果是 <= 0 的整数，则禁用思维链
    else:
        # 用户未指定 thinking_budget，回退到基于模型名称的逻辑
        include_thoughts_flag = should_include_thoughts(openai_request.model)
        if include_thoughts_flag:
            final_thinking_budget = get_thinking_budget(openai_request.model)

    # 为thinking模型添加thinking配置
    if include_thoughts_flag:
        if "generationConfig" not in request_data:
            request_data["generationConfig"] = {}

        thinking_config: Dict[str, Union[bool, int]] = {"includeThoughts": True}

        if final_thinking_budget is not None:
            thinking_config["thinkingBudget"] = final_thinking_budget

        request_data["generationConfig"]["thinkingConfig"] = thinking_config

    return {
        "model": get_base_model_name(openai_request.model),
        "request": request_data,
    }


def _extract_content_and_reasoning(parts: list) -> tuple[list, str]:
    openai_parts = []
    reasoning_content = ""
    for part in parts:
        if "text" in part:
            text_content = part["text"]

            # Handle malformed part where text value is a list like [{"type": "text", "text": "..."}]
            if (
                isinstance(text_content, list)
                and len(text_content) > 0
                and isinstance(text_content[0], dict)
                and "text" in text_content[0]
            ):
                text_content = text_content[0].get("text", "")

            # Ensure text_content is a string before further processing
            if not isinstance(text_content, str):
                text_content = str(text_content)

            if part.get("thought", False):
                reasoning_content += text_content
            else:
                openai_parts.append({"type": "text", "text": text_content})
        elif "inlineData" in part:
            mime_type = part["inlineData"].get("mimeType")
            data = part["inlineData"].get("data")
            if mime_type and data:
                image_url = f"data:{mime_type};base64,{data}"
                openai_parts.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
    return openai_parts, reasoning_content


def _convert_usage_metadata(usage_metadata: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not usage_metadata:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
        "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
        "total_tokens": usage_metadata.get("totalTokenCount", 0),
    }


def _build_message_with_reasoning(
    role: str, content: Union[str, list, None], reasoning_content: Optional[str]
) -> OpenAIChatMessage:
    message_data: Dict[str, Any] = {"role": role, "content": content}
    if reasoning_content:
        message_data["reasoning_content"] = reasoning_content
    return OpenAIChatMessage(**message_data)


def gemini_response_to_openai(
    gemini_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    choices = []
    all_reasoning_content = ""

    for index, candidate in enumerate(gemini_response.get("candidates", [])):
        gemini_parts = candidate.get("content", {}).get("parts", [])
        finish_reason = _map_finish_reason(candidate.get("finishReason"))

        openai_parts, reasoning_content = _extract_content_and_reasoning(gemini_parts)
        if reasoning_content:
            all_reasoning_content += reasoning_content

        function_calls = [part for part in gemini_parts if "functionCall" in part]

        message_data: Dict[str, Any] = {"role": "assistant", "content": None}

        if function_calls:
            tool_calls = []
            for fc in function_calls:
                function_call_data = fc["functionCall"]
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": function_call_data.get("name"),
                            "arguments": json.dumps(function_call_data.get("args", {})),
                        },
                    }
                )
            message_data["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            final_text_parts = []
            for part in openai_parts:
                if part["type"] == "text":
                    final_text_parts.append(part["text"])
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    final_text_parts.append(f"![image]({url})")
            final_content = "\n\n".join(final_text_parts)
            message_data["content"] = final_content

        choices.append(
            {
                "index": index,
                "message": OpenAIChatMessage(**message_data),
                "finish_reason": finish_reason,
            }
        )

    if choices and all_reasoning_content:
        # Attach the aggregated reasoning content to the first choice's message
        first_message_dict = choices[0]["message"].dict()
        first_message_dict["reasoning_content"] = all_reasoning_content
        choices[0]["message"] = OpenAIChatMessage(**first_message_dict)

    usage = _convert_usage_metadata(gemini_response.get("usageMetadata"))
    response_data = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": c["index"],
                "message": c["message"].dict(exclude_none=True),
                "finish_reason": c["finish_reason"],
            }
            for c in choices
        ],
        "system_fingerprint": "gcli2api",
    }
    if usage and usage.get("total_tokens", 0) > 0:
        response_data["usage"] = usage
    return response_data


def gemini_stream_chunk_to_openai(
    gemini_chunk: Dict[str, Any], model: str, response_id: str
) -> Dict[str, Any]:
    choices = []
    for candidate in gemini_chunk.get("candidates", []):
        log.debug(
            f"---------- Gemini Stream Candidate Received ----------\n{json.dumps(candidate, indent=2, ensure_ascii=False)}"
        )

        gemini_parts = candidate.get("content", {}).get("parts", [])
        finish_reason = _map_finish_reason(candidate.get("finishReason"))
        delta_data: Dict[str, Any] = {}

        openai_parts, reasoning_content = _extract_content_and_reasoning(gemini_parts)
        function_calls = [part for part in gemini_parts if "functionCall" in part]

        if function_calls:
            tool_calls = []
            for fc in function_calls:
                function_call_data = fc["functionCall"]
                tool_calls.append(
                    {
                        "index": 0,
                        "id": f"call_{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": function_call_data.get("name"),
                            "arguments": json.dumps(function_call_data.get("args", {})),
                        },
                    }
                )
            delta_data["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            stream_content_parts = []
            for part in openai_parts:
                if part.get("type") == "text":
                    stream_content_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    stream_content_parts.append(f"![image]({url})")
            if stream_content_parts:
                delta_data["content"] = "".join(stream_content_parts)

        if reasoning_content:
            delta_data["reasoning_content"] = reasoning_content

        if delta_data or finish_reason:
            choices.append(
                {
                    "index": candidate.get("index", 0),
                    "delta": OpenAIDelta(**delta_data),
                    "finish_reason": finish_reason,
                }
            )

    usage = _convert_usage_metadata(gemini_chunk.get("usageMetadata"))
    response_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": c["index"],
                "delta": c["delta"].dict(exclude_none=True),
                "finish_reason": c["finish_reason"],
            }
            for c in choices
        ],
        "system_fingerprint": "gcli2api",
    }
    if usage and usage.get("total_tokens", 0) > 0:
        has_finish_reason = any(choice.get("finish_reason") for choice in choices)
        if has_finish_reason:
            response_data["usage"] = usage
    return response_data


def _map_finish_reason(gemini_reason: Optional[str]) -> Optional[str]:
    if not gemini_reason:
        return None
    if gemini_reason == "STOP":
        return "stop"
    elif gemini_reason == "MAX_TOKENS":
        return "length"
    elif gemini_reason in ["SAFETY", "RECITATION"]:
        return "content_filter"
    elif gemini_reason == "OTHER":
        return "stop"
    return None


def validate_openai_request(request_data: Dict[str, Any]) -> ChatCompletionRequest:
    try:
        return ChatCompletionRequest(**request_data)
    except Exception as e:
        raise ValueError(f"Invalid OpenAI request format: {str(e)}")


def normalize_openai_request(
    request_data: ChatCompletionRequest,
) -> ChatCompletionRequest:
    if request_data.max_tokens is not None and request_data.max_tokens > 65535:
        request_data.max_tokens = 65535
    setattr(request_data, "top_k", 64)
    filtered_messages = []
    for m in request_data.messages:
        content = getattr(m, "content", None)
        if content:
            if isinstance(content, str) and content.strip():
                filtered_messages.append(m)
            elif isinstance(content, list) and len(content) > 0:
                has_valid_content = False
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text", "").strip():
                            has_valid_content = True
                            break
                        elif part.get("type") == "image_url" and part.get(
                            "image_url", {}
                        ).get("url"):
                            has_valid_content = True
                            break
                if has_valid_content:
                    filtered_messages.append(m)
    request_data.messages = filtered_messages
    return request_data


def is_health_check_request(request_data: ChatCompletionRequest) -> bool:
    return (
        len(request_data.messages) == 1
        and getattr(request_data.messages[0], "role", None) == "user"
        and getattr(request_data.messages[0], "content", None) == "Hi"
    )


def create_health_check_response() -> Dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": "gcli2api正常工作中"}}]
    }


def extract_model_settings(model: str) -> Dict[str, Any]:
    return {
        "base_model": get_base_model_name(model),
        "use_fake_streaming": model.endswith("-假流式"),
        "thinking_budget": get_thinking_budget(model),
        "include_thoughts": should_include_thoughts(model),
    }
