"""Generic dataclass <-> JSON conversion.

Kept deliberately small and dependency-free so the project data model can grow
without hand-written serialisation boilerplate for every new field.
"""
from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any, get_args, get_origin


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is typing.Union and type(None) in get_args(tp)


def _strip_optional(tp: Any) -> Any:
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0] if len(args) == 1 else Any


def from_jsonable(tp: Any, data: Any) -> Any:
    """Rebuild ``tp`` from plain JSON data, tolerating missing/extra keys."""
    if tp is Any or tp is None:
        return data
    if _is_optional(tp):
        if data is None:
            return None
        return from_jsonable(_strip_optional(tp), data)

    origin = get_origin(tp)
    if origin in (list, typing.List):
        (inner,) = get_args(tp) or (Any,)
        return [from_jsonable(inner, v) for v in (data or [])]
    if origin in (dict, typing.Dict):
        args = get_args(tp) or (str, Any)
        return {k: from_jsonable(args[1], v) for k, v in (data or {}).items()}
    if origin in (tuple, typing.Tuple):
        args = get_args(tp)
        return tuple(from_jsonable(a, v) for a, v in zip(args, data or []))

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(data)
        except ValueError:
            return list(tp)[0]

    if dataclasses.is_dataclass(tp):
        data = data or {}
        hints = typing.get_type_hints(tp)
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name in data:
                kwargs[f.name] = from_jsonable(hints.get(f.name, Any), data[f.name])
        return tp(**kwargs)

    if tp in (int, float) and isinstance(data, (int, float)):
        return tp(data)
    if tp is str and data is not None and not isinstance(data, str):
        return str(data)
    return data
