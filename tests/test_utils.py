"""Tests for the shared helpers in vibecheck._utils."""

from __future__ import annotations

import numpy as np

from vibecheck._utils import as_2d


def test_as_2d_flattens_trailing_axes():
    assert as_2d(np.zeros((5, 2, 3))).shape == (5, 6)


def test_as_2d_promotes_1d():
    assert as_2d(np.zeros(7)).shape == (7, 1)


def test_as_2d_handles_zero_row_2d():
    # reshape(0, -1) is ambiguous in numpy; the empty case is handled explicitly.
    assert as_2d(np.zeros((0, 3))).shape == (0, 3)


def test_as_2d_handles_zero_row_nd():
    assert as_2d(np.zeros((0, 2, 3))).shape == (0, 6)


def test_as_2d_handles_empty_1d():
    assert as_2d(np.zeros(0)).shape == (0, 1)
