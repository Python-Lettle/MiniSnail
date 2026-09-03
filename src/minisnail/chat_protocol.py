"""Canonical MiniSnail chat serialization shared by training and inference."""

from collections.abc import Mapping, Sequence
from typing import Any


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
MESSAGE_SEPARATOR = "\n"
CHAT_PROTOCOL_VERSION = "minisnail-chat-v1"
SUPPORTED_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _message_parts(message: Mapping[str, Any]) -> tuple[str, str]:
    role = message.get("role")
    content = message.get("content")
    if role not in SUPPORTED_ROLES:
        raise ValueError(
            f"不支持的对话角色 {role!r}; 可用角色: {sorted(SUPPORTED_ROLES)}"
        )
    if not isinstance(content, str):
        raise ValueError(f"{role!r} 消息的 content 必须是字符串")
    return role, content


def render_message(message: Mapping[str, Any]) -> str:
    """Render one complete message, including its trailing separator."""
    role, content = _message_parts(message)
    return f"{IM_START}{role}\n{content}{IM_END}{MESSAGE_SEPARATOR}"


def render_chat(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render complete messages without an assistant generation prompt."""
    return "".join(render_message(message) for message in messages)


def render_chat_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render complete messages followed by the assistant generation header."""
    return render_chat(messages) + f"{IM_START}assistant\n"


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return list(input_ids)


def encode_message_parts(
    tokenizer, message: Mapping[str, Any]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Encode header, content, end marker, and separator independently.

    Identical segmented encoding at every call site keeps the assistant boundary
    exact even when a tokenizer could merge across a newline boundary.
    """
    role, content = _message_parts(message)
    return (
        _encode(tokenizer, f"{IM_START}{role}\n"),
        _encode(tokenizer, content),
        _encode(tokenizer, IM_END),
        _encode(tokenizer, MESSAGE_SEPARATOR),
    )


def encode_chat(tokenizer, messages: Sequence[Mapping[str, Any]]) -> list[int]:
    """Encode complete messages using the canonical segmented protocol."""
    input_ids: list[int] = []
    for message in messages:
        for part in encode_message_parts(tokenizer, message):
            input_ids.extend(part)
    return input_ids


def encode_chat_prompt(tokenizer, messages: Sequence[Mapping[str, Any]]) -> list[int]:
    """Encode complete messages plus the assistant generation header."""
    input_ids = encode_chat(tokenizer, messages)
    input_ids.extend(_encode(tokenizer, f"{IM_START}assistant\n"))
    return input_ids


def encode_assistant_completion(tokenizer, content: str) -> list[int]:
    """Encode the supervised/generated portion of an assistant message."""
    if not isinstance(content, str):
        raise ValueError("assistant completion 必须是字符串")
    return _encode(tokenizer, content) + _encode(tokenizer, IM_END)


def encode_sft_example(
    tokenizer, messages: Sequence[Mapping[str, Any]]
) -> tuple[list[int], list[int]]:
    """Encode an SFT example with one shared label-masking convention.

    System/user/tool messages, all message headers, and separators are context.
    Only assistant content and its ``<|im_end|>`` token are supervised.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    supervised_tokens = 0

    for message in messages:
        role, _ = _message_parts(message)
        header_ids, content_ids, end_ids, separator_ids = encode_message_parts(
            tokenizer, message
        )
        message_ids = header_ids + content_ids + end_ids + separator_ids
        input_ids.extend(message_ids)

        if role == "assistant":
            labels.extend([-100] * len(header_ids))
            labels.extend(content_ids)
            labels.extend(end_ids)
            labels.extend([-100] * len(separator_ids))
            supervised_tokens += len(content_ids) + len(end_ids)
        else:
            labels.extend([-100] * len(message_ids))

    if supervised_tokens == 0:
        raise ValueError("对话中没有可监督的 assistant token")
    return input_ids, labels
