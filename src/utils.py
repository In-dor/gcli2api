import platform

CLI_VERSION = "0.1.5"  # Match current gemini-cli version


def get_user_agent():
    """Generate User-Agent string matching gemini-cli format."""
    version = CLI_VERSION
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{version} ({system}; {arch})"


def count_tokens(text: str) -> int:
    """
    简单的token计数估算 - 基于文本长度。
    大约4个字符 = 1个token。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_tokens_from_messages(messages: list) -> int:
    """
    从OpenAI格式的消息列表中估算token数量。
    """
    total_tokens = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_tokens += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_tokens += count_tokens(part.get("text", ""))
    return total_tokens


def estimate_tokens_from_gemini_contents(contents: list) -> int:
    """
    从Gemini格式的contents列表中估算token数量。
    """
    total_tokens = 0
    for content in contents:
        if "parts" in content:
            for part in content["parts"]:
                if "text" in part:
                    total_tokens += count_tokens(part["text"])
    return total_tokens
