"""Redaction utilities for SEB log data.

This module provides functions to redact sensitive information from log data
before it reaches prompts or is stored as evidence.
"""

from __future__ import annotations

import re

# Patterns for sensitive data that should be redacted
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
_IP_ADDRESS_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_MAC_ADDRESS_PATTERN = re.compile(
    r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
)
_PASSWORD_PATTERN = re.compile(
    r"(?i)(?:password|passwd|pwd|secret|token|key|api[_-]?key|auth[_-]?token)"
    r"['\"]?\s*[:=]\s*['\"]?([^'\"\s,;]+)"
)
_BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+",
)
_BASIC_AUTH_PATTERN = re.compile(
    r"(?i)basic\s+[a-zA-Z0-9+/=]+",
)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_PATH_USER_PATTERN = re.compile(
    r"(?i)(C:\\Users\\|/home/|/Users/)([^/\\]+)"
)
_QUERY_PARAM_SENSITIVE = re.compile(
    r"(?i)([?&](?:token|key|password|secret|auth|session|sid|jwt)=)([^&\s]+)"
)

# Redaction placeholder
_REDACTED = "[REDACTED]"
_REDACTED_EMAIL = "[REDACTED_EMAIL]"
_REDACTED_IP = "[REDACTED_IP]"
_REDACTED_MAC = "[REDACTED_MAC]"
_REDACTED_TOKEN = "[REDACTED_TOKEN]"
_REDACTED_UUID = "[REDACTED_UUID]"
_REDACTED_USER = "[REDACTED_USER]"


def redact_text(text: str, preserve_structure: bool = True) -> str:
    """Redact sensitive information from text.

    Args:
        text: The text to redact.
        preserve_structure: If True, uses specific placeholders that indicate
            the type of data redacted. If False, uses generic placeholder.

    Returns:
        Text with sensitive information redacted.
    """
    if not text:
        return text

    result = text

    # Redact passwords and secrets first (before other patterns might match)
    result = _PASSWORD_PATTERN.sub(
        lambda m: m.group(0).replace(m.group(1), _REDACTED_TOKEN)
        if m.group(1) else m.group(0),
        result,
    )

    # Redact bearer tokens
    result = _BEARER_TOKEN_PATTERN.sub(
        f"Bearer {_REDACTED_TOKEN}" if preserve_structure else _REDACTED,
        result,
    )

    # Redact basic auth
    result = _BASIC_AUTH_PATTERN.sub(
        f"Basic {_REDACTED_TOKEN}" if preserve_structure else _REDACTED,
        result,
    )

    # Redact sensitive query parameters
    result = _QUERY_PARAM_SENSITIVE.sub(
        lambda m: f"{m.group(1)}{_REDACTED_TOKEN}",
        result,
    )

    # Redact emails
    if preserve_structure:
        result = _EMAIL_PATTERN.sub(_REDACTED_EMAIL, result)
    else:
        result = _EMAIL_PATTERN.sub(_REDACTED, result)

    # Redact IP addresses (but preserve localhost and common internal ranges description)
    def redact_ip(match: re.Match[str]) -> str:
        ip = match.group(0)
        # Preserve localhost
        if ip == "127.0.0.1":
            return ip
        # Preserve link-local
        if ip.startswith("169.254."):
            return ip
        return _REDACTED_IP if preserve_structure else _REDACTED

    result = _IP_ADDRESS_PATTERN.sub(redact_ip, result)

    # Redact MAC addresses
    result = _MAC_ADDRESS_PATTERN.sub(
        _REDACTED_MAC if preserve_structure else _REDACTED,
        result,
    )

    # Redact user paths (keep the directory structure indicator)
    def redact_user_path(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return f"{prefix}{_REDACTED_USER}"

    result = _PATH_USER_PATTERN.sub(redact_user_path, result)

    return result


def redact_url(url: str) -> str:
    """Redact sensitive parts of a URL while preserving structure.

    Args:
        url: The URL to redact.

    Returns:
        URL with sensitive query parameters and credentials redacted.
    """
    if not url:
        return url

    result = url

    # Redact credentials in URL (user:pass@host)
    result = re.sub(
        r"://([^:]+):([^@]+)@",
        f"://{_REDACTED_USER}:{_REDACTED_TOKEN}@",
        result,
    )

    # Redact sensitive query parameters
    result = _QUERY_PARAM_SENSITIVE.sub(
        lambda m: f"{m.group(1)}{_REDACTED_TOKEN}",
        result,
    )

    return result


def redact_window_title(title: str) -> str:
    """Redact potentially sensitive information from window titles.

    Args:
        title: The window title to redact.

    Returns:
        Window title with sensitive information redacted.
    """
    if not title:
        return title

    # Apply general redaction
    result = redact_text(title)

    # Additional window-title specific patterns could be added here
    # e.g., file paths, document names with PII

    return result


def redact_navigation_url(url: str) -> str:
    """Redact a navigation URL, preserving domain but protecting parameters.

    Args:
        url: The navigation URL to redact.

    Returns:
        URL with sensitive parts redacted.
    """
    return redact_url(url)
