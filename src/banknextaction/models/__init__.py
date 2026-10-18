"""Model implementations; importing this package registers every model in the registry."""

from banknextaction.models import sklearn_model  # noqa: F401  (registers "logreg")
from banknextaction.models.base import BaseModel
from banknextaction.models.registry import available_models, create_model, model_class

__all__ = ["BaseModel", "available_models", "create_model", "model_class"]
