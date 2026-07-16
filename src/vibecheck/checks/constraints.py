"""Constraint checks: known physics the surrogate must not violate.

Classical solvers respect physical laws by construction; surrogates do not.
These checks test user-declared constraints on the model outputs: conservation
(mass / energy), positivity, monotonicity, symmetry, and hard bounds.
Constraints are supplied through ``metadata`` so the package stays domain
agnostic.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def physical(**context: Any) -> CheckResult:
    """Test declared physical constraints against the surrogate's outputs."""
    raise NotImplementedError


physical.check_name = "constraints.physical"
