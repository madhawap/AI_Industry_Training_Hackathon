"""Shared fixtures.

The store is built once per session, mirroring production: everything expensive
happens at startup, and the built store is then treated as immutable and shared
across concurrent readers.
"""

from __future__ import annotations

import pytest
from src.tfql import Store


@pytest.fixture(scope="session")
def store() -> Store:
    return Store.build()


@pytest.fixture(scope="session")
def run(store):
    """Call one operation directly, returning its OpOutput."""
    from src.tfql import registry

    def _run(op: str, **args):
        spec = registry.get(op)
        return spec.fn(registry.parse_args(spec, args), store)

    return _run
