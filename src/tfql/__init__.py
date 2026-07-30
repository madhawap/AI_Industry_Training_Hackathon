"""TFQL -- the typed financial query language.

Qwen selects typed operations, this package validates and executes them, and the
fine-tuned model writes the answer from the resulting evidence bundle. Importing
this package registers the full operation catalogue.
"""

from . import registry
from .errors import ErrorCode, TFQLError
from .executor import execute, validate_plan
from .models import SCHEMA_VERSION, PlanRequest, PlanResult

# Importing the operation modules is what populates the registry.
from .operations import afr, asx, cross, rba  # noqa: F401  (side-effecting)
from .store import Store

__all__ = [
    "SCHEMA_VERSION",
    "ErrorCode",
    "PlanRequest",
    "PlanResult",
    "Store",
    "TFQLError",
    "execute",
    "registry",
    "validate_plan",
]
