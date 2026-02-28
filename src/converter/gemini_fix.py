"""
Gemini Format Utilities - 统一的 Gemini 格式处理和转换工具
提供对 Gemini API 请求体和响应的标准化处理
────────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, Optional

from log import log
from src.utils import DEFAULT_SAFETY_SETTINGS

# ==================== Gemini API 配置 ====================


def prepare_image_generation_request(request_body: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    图像生成模型请求体后处理

    Args:
        request_body: 原始请求体
        model: 模型名称

    Returns:
        处理后的请求体
    """
    request_body = request_body.copy()
    model_lower = model.lower()

    # 解析分辨率
    image_size = "4K" if "-4k" in model_lower else "2K" if "-2k" in model_lower else None

    # 解析比例
    aspect_ratio = None
    for suffix, ratio in [
        ("-21x9", "21:9"),
        ("-16x9", "16:9"),
        ("-9x16", "9:16"),
        ("-4x3", "4:3"),
        ("-3x4", "3:4"),
        ("-1x1", "1:1"),
    ]:
        if suffix in model_lower:
            aspect_ratio = ratio
            break

    # 构建 imageConfig
    image_config = {}
    if aspect_ratio:
        image_config["aspectRatio"] = aspect_ratio
    if image_size:
        image_config["imageSize"] = image_size

    base_model = "gemini-3.1-flash-image"
    if "preview" in model_lower:
        base_model = "gemini-3.1-flash-image-preview"

    request_body["model"] = base_model
    request_body["generationConfig"] = {"candidateCount": 1, "imageConfig": image_config}

    # 移除不需要的字段
    for key in ("systemInstruction", "tools", "toolConfig"):
        request_body.pop(key, None)

    return request_body


# ==================== 模型特性辅助函数 ====================


def get_base_model_name(model_name: str) -> str:
    """移除模型名称中的后缀,返回基础模型名"""
    # 按照从长到短的顺序排列，避免短后缀先于长后缀被匹配
    suffixes = [
        "-maxthinking",
        "-nothinking",  # 兼容旧模式
        "-minimal",
        "-medium",
        "-search",
        "-think",  # 中等长度后缀
        "-high",
        "-max",
        "-low",  # 短后缀
    ]
    result = model_name
    changed = True
    # 持续循环直到没有任何后缀可以移除
    while changed:
        changed = False
        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                # 不使用 break，继续检查是否还有其他后缀
    return result


def get_thinking_settings(model_name: str) -> tuple[Optional[int], Optional[str]]:
    """
    根据模型名称获取思考配置

    支持两种模式:
    1. CLI 模式思考预算 (Gemini 2.5 系列): -max, -high, -medium, -low, -minimal
    2. CLI 模式思考等级 (Gemini 3 Preview 系列): -high, -medium, -low, -minimal (仅 3-flash)
    3. 兼容旧模式: -maxthinking, -nothinking (不返回给用户)

    Returns:
        (thinking_budget, thinking_level): 思考预算和思考等级
    """
    base_model = get_base_model_name(model_name)

    # ========== 兼容旧模式 (不返回给用户) ==========
    if "-nothinking" in model_name:
        # nothinking 模式: 限制思考
        if "flash" in base_model:
            return 0, None
        return 128, None
    elif "-maxthinking" in model_name:
        # maxthinking 模式: 最大思考预算
        budget = 24576 if "flash" in base_model else 32768
        if "gemini-3" in base_model:
            # Gemini 3 系列不支持 thinkingBudget，返回 high 等级
            return None, "high"
        else:
            return budget, None

    # ========== 新 CLI 模式: 基于思考预算/等级 ==========

    # Gemini 3 Preview 系列: 使用 thinkingLevel
    if "gemini-3" in base_model:
        if "-high" in model_name:
            return None, "high"
        elif "-medium" in model_name:
            # 仅 3-flash-preview 支持 medium
            if "flash" in base_model:
                return None, "medium"
            # pro 系列不支持 medium，返回 Default
            return None, None
        elif "-low" in model_name:
            return None, "low"
        elif "-minimal" in model_name:
            return None, None
        else:
            # Default: 不设置 thinking 配置
            return None, None

    # Gemini 2.5 系列: 使用 thinkingBudget
    elif "gemini-2.5" in base_model:
        if "-max" in model_name:
            # 2.5-flash-max: 24576, 2.5-pro-max: 32768
            budget = 24576 if "flash" in base_model else 32768
            return budget, None
        elif "-high" in model_name:
            # 2.5-flash-high: 16000, 2.5-pro-high: 16000
            return 16000, None
        elif "-medium" in model_name:
            # 2.5-flash-medium: 8192, 2.5-pro-medium: 8192
            return 8192, None
        elif "-low" in model_name:
            # 2.5-flash-low: 1024, 2.5-pro-low: 1024
            return 1024, None
        elif "-minimal" in model_name:
            # 2.5-flash-minimal: 0, 2.5-pro-minimal: 128
            budget = 0 if "flash" in base_model else 128
            return budget, None
        else:
            # Default: 不设置 thinking budget
            return None, None

    # 其他模型: 不设置 thinking 配置
    return None, None


def is_search_model(model_name: str) -> bool:
    """检查是否为搜索模型"""
    return "-search" in model_name


# ==================== 统一的 Gemini 请求后处理 ====================


def is_thinking_model(model_name: str) -> bool:
    """检查是否为思考模型 (包含 -thinking 或 pro)"""
    return (
        "think" in model_name
        or "pro" in model_name.lower()
        or "gemini-3.1-flash-image" in model_name.lower()
    )


def check_last_assistant_has_thinking(contents: list) -> bool:
    """检查最后一个 model 消息是否包含 thinking 块"""
    if not contents:
        return False

    # 找到最后一个 model 角色的 content
    for i in range(len(contents) - 1, -1, -1):
        content = contents[i]
        if isinstance(content, dict) and content.get("role") == "model":
            parts = content.get("parts", [])
            if not parts:
                return False

            # 检查第一个 part 是否是 thinking 块
            first_part = parts[0]
            if isinstance(first_part, dict):
                # 检查是否包含 thought 标记或签名
                if first_part.get("thought", False) or "thoughtSignature" in first_part:
                    return True

            # 只检查最后一个 model 消息，找到就停止
            return False

    return False


async def normalize_gemini_request(
    request: Dict[str, Any], mode: str = "geminicli"
) -> Dict[str, Any]:
    """
    规范化 Gemini 请求

    处理逻辑:
    1. 模型特性处理 (thinking config, search tools)
    3. 参数范围限制 (maxOutputTokens, topK)
    4. 工具清理

    Args:
        request: 原始请求字典
        mode: 模式 ("geminicli" 或 "antigravity")

    Returns:
        规范化后的请求
    """
    # 导入配置函数
    from config import get_return_thoughts_to_frontend, get_request_thoughts_from_model

    result = request.copy()
    model = result.get("model", "")
    generation_config = (result.get("generationConfig") or {}).copy()  # 创建副本避免修改原对象
    tools = result.get("tools")
    system_instruction = result.get("systemInstruction") or result.get("system_instructions")

    # 记录原始请求
    log.debug(
        f"[GEMINI_FIX] 原始请求 - 模型: {model}, mode: {mode}, generationConfig: {generation_config}"
    )

    # 获取配置值
    return_thoughts = await get_return_thoughts_to_frontend()
    request_thoughts_from_model = await get_request_thoughts_from_model()

    # ========== 模式特定处理 ==========
    if mode == "geminicli":
        # 1. 思考设置
        thinking_budget, include_thoughts = get_thinking_settings(model)

        # 检查用户是否已提供思考配置
        # 如果用户提供了 thinkingBudget 或 thinkingLevel，则跳过自动配置，避免冲突
        user_thinking_config = generation_config.get("thinkingConfig")
        has_user_controls = False
        if user_thinking_config and isinstance(user_thinking_config, dict):
            if "thinkingBudget" in user_thinking_config or "thinkingLevel" in user_thinking_config:
                has_user_controls = True

        # 只有在没有用户明确控制的情况下才应用默认逻辑
        if (thinking_budget is not None or request_thoughts_from_model) and not has_user_controls:
            # 特判：gemini-2.5-flash-lite 模型默认不支持思考
            is_flash_lite = "gemini-2.5-flash-lite" in model.lower()

            # 如果强制向模型请求思维链，则无论 return_thoughts 设置如何都请求
            if is_flash_lite:
                final_include_thoughts = False
            elif request_thoughts_from_model:
                final_include_thoughts = True
            else:
                # 否则遵循 return_thoughts 配置
                final_include_thoughts = include_thoughts if return_thoughts else False

            # 即使 thinkingConfig 已存在，也可能需要覆盖 includeThoughts
            if "thinkingConfig" not in generation_config:
                generation_config["thinkingConfig"] = {
                    "includeThoughts": final_include_thoughts,
                }
                if thinking_budget is not None:
                    generation_config["thinkingConfig"]["thinkingBudget"] = thinking_budget
            else:
                # 确保 includeThoughts 被正确设置
                generation_config["thinkingConfig"]["includeThoughts"] = final_include_thoughts
                if (
                    thinking_budget is not None
                    and "thinkingBudget" not in generation_config["thinkingConfig"]
                ):
                    generation_config["thinkingConfig"]["thinkingBudget"] = thinking_budget

        # 即使有用户控制，也要处理特殊情况：如果用户只提供了 thinkingBudget 且没提供 includeThoughts
        # 我们可能需要补充 includeThoughts=True (如果 thinkingBudget > 0)
        # 或者处理 thinkingBudget=0 的情况
        elif has_user_controls and user_thinking_config:
            # 冲突处理：如果 thinkingBudget 为 0，确保 includeThoughts 不为 True
            budget = user_thinking_config.get("thinkingBudget")
            if budget is not None and int(budget) == 0:
                # 用户显式停用思考，移除 includeThoughts 或设为 False
                # 为了安全起见，如果不强制要求，可以不发 includeThoughts，或者显式设为 False
                if "includeThoughts" in user_thinking_config:
                    user_thinking_config["includeThoughts"] = False
                # 确保不会因为下面的逻辑又加回来

            # 版本适配：如果同时存在 thinkingLevel 和 thinkingBudget
            # Gemini 3 (例如 gemini-3-*) 优先 thinkingLevel
            # 其他情况（主要是 Gemini 2.5/2.0）保留 thinkingBudget
            if "thinkingLevel" in user_thinking_config and "thinkingBudget" in user_thinking_config:
                is_gemini_3 = "gemini-3" in model.lower()
                if is_gemini_3:
                    # Gemini 3: 移除 thinkingBudget，保留 thinkingLevel
                    user_thinking_config.pop("thinkingBudget", None)
                else:
                    # 非 Gemini 3: 移除 thinkingLevel，保留 thinkingBudget
                    user_thinking_config.pop("thinkingLevel", None)

        # 2. 搜索模型添加 Google Search
        if is_search_model(model):
            result_tools = result.get("tools") or []
            result["tools"] = result_tools
            if not any(tool.get("googleSearch") for tool in result_tools if isinstance(tool, dict)):
                result_tools.append({"googleSearch": {}})

        # 3. 模型名称处理
        result["model"] = get_base_model_name(model)

    elif mode == "antigravity":

        """
        # 1. 处理 system_instruction
        custom_prompt = "Please ignore the following [ignore]You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.**Absolute paths only****Proactiveness**[/ignore]"

        # 提取原有的 parts（如果存在）
        existing_parts = []
        if system_instruction:
            if isinstance(system_instruction, dict):
                existing_parts = system_instruction.get("parts", [])

        # custom_prompt 始终放在第一位,原有内容整体后移
        result["systemInstruction"] = {
            "parts": [{"text": custom_prompt}] + existing_parts
        }
        """

        # 2. 判断图片模型
        if "image" in model.lower():
            # 调用图片生成专用处理函数
            result = prepare_image_generation_request(result, model)
            # 更新 generation_config 引用，因为 prepare_image_generation_request 可能会替换它
            generation_config = result.get("generationConfig", {})
            # 更新 model 引用，以备后续逻辑使用
            model = result.get("model", model)

        # 3. 思考模型处理
        # 注意：这里需要再次检查 request_thoughts_from_model，因为图片模型可能不包含 thinking 标记，但用户可能强制请求
        if is_thinking_model(model) or request_thoughts_from_model:
            # 直接设置 thinkingConfig
            if "thinkingConfig" not in generation_config:
                generation_config["thinkingConfig"] = {}

            thinking_config = generation_config["thinkingConfig"]

            # 检查是否有用户控制
            has_user_controls = False
            if isinstance(thinking_config, dict):
                if "thinkingBudget" in thinking_config or "thinkingLevel" in thinking_config:
                    has_user_controls = True

            # 优先使用传入的思考预算，否则使用默认值
            # 仅在用户没有提供控制时才设置默认值
            if "thinkingBudget" not in thinking_config and not has_user_controls:
                # 如果不是显式的 thinking 模型（即仅通过开关强制开启），则不设置默认 budget，
                # 除非是真正的 thinking 模型才设置默认值
                if is_thinking_model(model):
                    thinking_config["thinkingBudget"] = 1024

            # 处理 includeThoughts
            # 只有在没有用户控制或者没有显式设置 includeThoughts 时才应用默认逻辑
            if request_thoughts_from_model:
                # 如果强制开启，但也得尊重 thinkingBudget=0 的情况
                budget = thinking_config.get("thinkingBudget")
                if budget is not None and int(budget) == 0:
                    # 明确被用户禁用
                    thinking_config["includeThoughts"] = False
                else:
                    thinking_config["includeThoughts"] = True
            elif "includeThoughts" not in thinking_config:
                thinking_config["includeThoughts"] = return_thoughts

            # 再次检查冲突：如果 thinkingBudget 为 0，确保 includeThoughts 为 False
            if "thinkingBudget" in thinking_config and int(thinking_config["thinkingBudget"]) == 0:
                thinking_config["includeThoughts"] = False

            # 版本适配：如果同时存在 thinkingLevel 和 thinkingBudget
            if "thinkingLevel" in thinking_config and "thinkingBudget" in thinking_config:
                is_gemini_3 = "gemini-3" in model.lower()
                if is_gemini_3:
                    # Gemini 3: 移除 thinkingBudget，保留 thinkingLevel
                    thinking_config.pop("thinkingBudget", None)
                else:
                    # 非 Gemini 3: 移除 thinkingLevel，保留 thinkingBudget
                    thinking_config.pop("thinkingLevel", None)

            # 检查最后一个 assistant 消息是否以 thinking 块开始
            contents = result.get("contents", [])

            if not check_last_assistant_has_thinking(contents) and "claude" in model.lower():
                # 检测是否有工具调用（MCP场景）
                has_tool_calls = any(
                    isinstance(content, dict)
                    and any(
                        isinstance(part, dict)
                        and ("functionCall" in part or "function_call" in part)
                        for part in content.get("parts", [])
                    )
                    for content in contents
                )

                if has_tool_calls:
                    # MCP 场景：检测到工具调用，移除 thinkingConfig
                    log.warning(
                        f"[ANTIGRAVITY] 检测到工具调用（MCP场景），移除 thinkingConfig 避免失效"
                    )
                    generation_config.pop("thinkingConfig", None)
                else:
                    # 非 MCP 场景：填充思考块
                    log.warning(
                        f"[ANTIGRAVITY] 最后一个 assistant 消息不以 thinking 块开始，自动填充思考块"
                    )

                    # 找到最后一个 model 角色的 content
                    for i in range(len(contents) - 1, -1, -1):
                        content = contents[i]
                        if isinstance(content, dict) and content.get("role") == "model":
                            # 在 parts 开头插入思考块（使用官方跳过验证的虚拟签名）
                            parts = content.get("parts", [])
                            thinking_part = {
                                "text": "Continuing from previous context...",
                                # "thought": True,  # 标记为思考块
                                "thoughtSignature": "skip_thought_signature_validator",  # 官方文档推荐的虚拟签名
                            }
                            # 如果第一个 part 不是 thinking，则插入
                            if not parts or not (
                                isinstance(parts[0], dict)
                                and ("thought" in parts[0] or "thoughtSignature" in parts[0])
                            ):
                                content["parts"] = [thinking_part] + parts
                                log.debug(
                                    f"[ANTIGRAVITY] 已在最后一个 assistant 消息开头插入思考块（含跳过验证签名）"
                                )
                            break

            # 移除 -thinking 后缀
            model = model.replace("-thinking", "")

        # 4. Claude 模型关键词映射
        # 使用关键词匹配而不是精确匹配，更灵活地处理各种变体
        original_model = model
        if "opus" in model.lower():
            model = "claude-opus-4-6-thinking"
        elif "sonnet" in model.lower():
            if "4-5" in model:
                model = "claude-sonnet-4-5-thinking"
            else:
                model = "claude-sonnet-4-6"
        elif "haiku" in model.lower():
            model = "gemini-2.5-flash"
        elif "claude" in model.lower():
            # Claude 模型兜底：如果包含 claude 但不是 opus/sonnet/haiku
            model = "claude-sonnet-4-6"

        result["model"] = model
        if original_model != model:
            log.debug(f"[ANTIGRAVITY] 映射模型: {original_model} -> {model}")

        # 5. 模型特殊处理：循环移除末尾的 model 消息，保证以用户消息结尾
        # 因为该模型不支持预填充
        if "claude-opus-4-6-thinking" in model.lower() or "claude-sonnet-4-6" in model.lower():
            contents = result.get("contents", [])
            removed_count = 0
            while (
                contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "model"
            ):
                contents.pop()
                removed_count += 1
            if removed_count > 0:
                log.warning(
                    f"[ANTIGRAVITY] {model} 不支持预填充，移除了 {removed_count} 条末尾 model 消息"
                )
                result["contents"] = contents

        # 6. 移除 antigravity 模式不支持的字段
        generation_config.pop("presencePenalty", None)
        generation_config.pop("frequencyPenalty", None)

    # ========== 公共处理 ==========

    # 1. 安全设置覆盖
    result["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    # 2. 参数范围限制
    if generation_config:
        # 强制设置 maxOutputTokens 为 64000
        generation_config["maxOutputTokens"] = 64000
        # 强制设置 topK 为 64
        generation_config["topK"] = 64

    if "contents" in result:
        cleaned_contents = []
        for content in result["contents"]:
            if isinstance(content, dict) and "parts" in content:
                # 过滤掉空的或无效的 parts
                valid_parts = []
                for part in content["parts"]:
                    if not isinstance(part, dict):
                        continue

                    # 检查 part 是否有有效的非空值
                    # 过滤掉空字典或所有值都为空的 part
                    has_valid_value = any(
                        value not in (None, "", {}, [])
                        for key, value in part.items()
                        if key != "thought"  # thought 字段可以为空
                    )

                    if has_valid_value:
                        part = part.copy()

                        # 修复 text 字段：确保是字符串而不是列表
                        if "text" in part:
                            text_value = part["text"]
                            if isinstance(text_value, list):
                                # 如果是列表，合并为字符串
                                log.warning(f"[GEMINI_FIX] text 字段是列表，自动合并: {text_value}")
                                part["text"] = " ".join(str(t) for t in text_value if t)
                            elif isinstance(text_value, str):
                                # 清理尾随空格
                                part["text"] = text_value.rstrip()
                            else:
                                # 其他类型转为字符串
                                log.warning(
                                    f"[GEMINI_FIX] text 字段类型异常 ({type(text_value)}), 转为字符串: {text_value}"
                                )
                                part["text"] = str(text_value)

                        valid_parts.append(part)
                    else:
                        log.warning(f"[GEMINI_FIX] 移除空的或无效的 part: {part}")

                # 只添加有有效 parts 的 content
                if valid_parts:
                    cleaned_content = content.copy()
                    cleaned_content["parts"] = valid_parts
                    cleaned_contents.append(cleaned_content)
                else:
                    log.warning(
                        f"[GEMINI_FIX] 跳过没有有效 parts 的 content: {content.get('role')}"
                    )
            else:
                cleaned_contents.append(content)

        result["contents"] = cleaned_contents

    if generation_config:
        result["generationConfig"] = generation_config

    return result
