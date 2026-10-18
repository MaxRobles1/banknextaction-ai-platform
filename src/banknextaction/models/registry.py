"""Model registry and factory: from a name in the configuration to a ready model object.

Strategy + Factory (Clase 3): every model is an interchangeable strategy behind BaseModel,
and `create_model` builds one from its registered name. Adding a third model means one new
class with @register_model; no training, evaluation or API code changes (open/closed).
"""

from __future__ import annotations

from collections.abc import Callable

from banknextaction.config import Config
from banknextaction.models.base import BaseModel

ModelFactory = Callable[[Config], BaseModel]

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Class decorator that registers a BaseModel implementation under `name`."""

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> tuple[str, ...]:
    """Names of the registered models."""
    return tuple(sorted(_REGISTRY))


def model_class(name: str) -> type[BaseModel]:
    """Registered class for `name`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown model '{name}'; available: {available_models()}") from exc


def create_model(name: str, cfg: Config) -> BaseModel:
    """Factory: build a new, untrained model from its name and the configuration."""
    return model_class(name).from_config(cfg)
