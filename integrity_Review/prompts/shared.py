"""Shared prompt version tags and token limits."""

PERCEPTION_PROMPT_VERSION = "scope2-v12"
SCREEN_PERCEPTION_PROMPT_VERSION = "screen-v5"
SCREEN_CAMERA_PERCEPTION_PROMPT_VERSION = "screen-camera-v2"
PERCEPTION_MAX_OUTPUT_TOKENS = 4096
GEMINI_MIN_VIDEO_DURATION_MS = 1_500
DEFAULT_GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
# Deliberation runs on flash. Kept separate from DEFAULT_GEMINI_PRO_MODEL so the
# SEB log analysis that also reads that constant is not switched by proxy.
DELIBERATION_MODEL = "gemini-3.8-flash"
PERCEPTION_ASSEMBLY_LOGIC_VERSION = "assembly-v3-observation-per-event"
PERCEPTION_MODEL_VERSION = DEFAULT_GEMINI_FLASH_MODEL
