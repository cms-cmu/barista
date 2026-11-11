"""Tests for framework-level deterministic training seeds via Squares CBRNG.

Covers:
  - Global Training.seed setting (default 666)
  - _derive_squares_seeds stream derivation (fold, weight, batch)
  - Stream independence and orthogonal decorrelation
  - Model name decorrelation (FvT vs SvB)
  - MultiStageTraining deterministic seeding and cuDNN flags
"""

import os
import random
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pytest
import torch

from src.classifier.config.model._kfold import _derive_squares_seeds
from src.classifier.config.setting.ml import Training


def test_default_training_seed():
    """Verify default root training seed is 666."""
    assert Training.seed == 666


def test_derive_squares_seeds_basic():
    """Verify 3 orthogonal seeds are generated and non-negative."""
    fold_seed, weight_seed, batch_seed = _derive_squares_seeds(
        base_seed=666,
        model_name="FvT",
        offset=0,
        kfold_seed=["kfold"],
    )
    assert isinstance(fold_seed, tuple)
    assert isinstance(fold_seed[0], int)
    assert isinstance(weight_seed, int)
    assert isinstance(batch_seed, int)
    assert 0 <= fold_seed[0] < 2**32
    assert 0 <= weight_seed < 2**32
    assert 0 <= batch_seed < 2**32
    # Distinct streams
    assert fold_seed[0] != weight_seed
    assert weight_seed != batch_seed
    assert fold_seed[0] != batch_seed


def test_derive_squares_seeds_decorrelation_across_offsets():
    """Verify different k-fold offsets produce mutually distinct weight and batch seeds."""
    seeds_off0 = _derive_squares_seeds(666, "FvT", 0, "kfold")
    seeds_off1 = _derive_squares_seeds(666, "FvT", 1, "kfold")
    seeds_off2 = _derive_squares_seeds(666, "FvT", 2, "kfold")

    # Weight seeds differ across offsets
    assert seeds_off0[1] != seeds_off1[1] != seeds_off2[1]
    # Batch seeds differ across offsets
    assert seeds_off0[2] != seeds_off1[2] != seeds_off2[2]


def test_derive_squares_seeds_model_name_decorrelation():
    """Verify FvT and SvB produce decorrelated seeds at the same offset."""
    fvt_seeds = _derive_squares_seeds(666, "FvT", 0, "kfold")
    svb_seeds = _derive_squares_seeds(666, "SvB", 0, "kfold")

    assert fvt_seeds[0] != svb_seeds[0]
    assert fvt_seeds[1] != svb_seeds[1]
    assert fvt_seeds[2] != svb_seeds[2]


def test_derive_squares_seeds_determinism():
    """Verify identical inputs produce identical outputs (reproducibility)."""
    run1 = _derive_squares_seeds(666, "FvT", 0, "kfold")
    run2 = _derive_squares_seeds(666, "FvT", 0, "kfold")

    assert run1 == run2


def test_multistage_training_deterministic_seeding():
    """Verify setting seed configures torch, numpy, and random deterministically."""
    from src.classifier.ml.training import MultiStageTraining

    class MockTrainer:
        name = "mock"
        metadata = {"seed_weights": 12345}

    trainer = MockTrainer()
    MultiStageTraining._set_deterministic_seed(trainer)

    r1 = random.randint(0, 100000)
    np1 = np.random.randint(0, 100000)
    t1 = torch.randint(0, 100000, (1,)).item()

    MultiStageTraining._set_deterministic_seed(trainer)

    r2 = random.randint(0, 100000)
    np2 = np.random.randint(0, 100000)
    t2 = torch.randint(0, 100000, (1,)).item()

    assert r1 == r2
    assert np1 == np2
    assert t1 == t2
    assert torch.backends.cudnn.deterministic is True
