"""Duck-typed attribute access for Pydantic models, SimpleNamespace, and dicts."""

from __future__ import annotations

from typing import Any, Mapping


def attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        camel = _snake_to_camel(name)
        if camel != name:
            if hasattr(obj, camel):
                value = getattr(obj, camel)
                if value is not None:
                    return value
            if isinstance(obj, Mapping) and camel in obj:
                return obj[camel]
    return default


def fact_kind(fact: Any) -> str:
    return str(attr(fact, "kind", default=""))


def fact_start_ms(fact: Any) -> int:
    return int(attr(fact, "start_offset_ms", "startOffsetMs", default=0) or 0)


def fact_end_ms(fact: Any) -> int:
    end = attr(fact, "end_offset_ms", "endOffsetMs")
    return int(end if end is not None else fact_start_ms(fact))


def mapping_view(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    data = getattr(obj, "__dict__", None)
    if isinstance(data, Mapping):
        return data
    return {}


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
