from dataclasses import dataclass
from typing import Any, Callable
import difflib
import inspect

_REG: dict[str, Callable[..., Any]] = {}
_DESC: dict[str, str] = {}


def register(name: str, desc: str = ""):
    """Decorator to register a preset factory under a stable string key."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        k = name.lower()
        if k in _REG:
            raise KeyError(f"Duplicate preset name '{name}'")
        _REG[k] = fn
        _DESC[k] = str(desc or "")
        return fn

    return deco


def _resolve(name: str) -> str:
    k = name.lower()
    if k in _REG:
        return k
    sugg = difflib.get_close_matches(k, list(_REG.keys()), n=1)
    hint = f" Did you mean '{sugg[0]}'?" if sugg else ""
    raise KeyError(f"Unknown preset '{name}'.{hint}")


def make(name: str, /, **kwargs: Any) -> Any:
    """Construct a standard runnable Somax method from a string key (for config/CLI)."""
    k = _resolve(name)
    return _REG[k](**kwargs)


def list_methods() -> dict[str, str]:
    """Return {name: description} for all registered presets."""
    return dict(sorted(_DESC.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class MethodInfo:
    name: str
    desc: str
    signature: str


def describe(name: str) -> MethodInfo:
    """Return description and signature for one preset."""
    k = _resolve(name)
    fn = _REG[k]
    return MethodInfo(name=k, desc=_DESC.get(k, ""), signature=str(inspect.signature(fn)))


def registered_methods() -> dict[str, Callable[..., Any]]:
    """Return a copy of registered methods {name: factory}."""
    return dict(_REG)
