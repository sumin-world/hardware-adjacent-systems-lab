"""Unit tests for common.py — band helpers, perturbation, statistics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import skrf as rf

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (
    band_mask,
    compute_scalar_objective,
    percentile_stats,
    perturb_network_single,
    perturb_network_vectorized,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_network() -> rf.Network:
    """Create a minimal 2-port network from scikit-rf built-in data."""
    return rf.data.ring_slot


# ---------------------------------------------------------------------------
# band_mask
# ---------------------------------------------------------------------------

class TestBandMask:
    def test_full_range_returns_all_true(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = band_mask(f, None, None)
        assert mask.all()

    def test_fmin_only(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = band_mask(f, 3.0, None)
        assert mask.tolist() == [False, False, True, True, True]

    def test_fmax_only(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = band_mask(f, None, 3.0)
        assert mask.tolist() == [True, True, True, False, False]

    def test_band_limits(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = band_mask(f, 2.0, 4.0)
        assert mask.tolist() == [False, True, True, True, False]

    def test_empty_band_raises(self):
        f = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="No frequency points"):
            band_mask(f, 10.0, 20.0)


# ---------------------------------------------------------------------------
# compute_scalar_objective
# ---------------------------------------------------------------------------

class TestScalarObjective:
    def test_s21_min_mode(self):
        f = np.array([1.0, 2.0, 3.0])
        s21 = np.array([-1.0, -3.5, -2.0])
        result = compute_scalar_objective(f, s21, mode="s21_min")
        assert result == pytest.approx(-3.5)

    def test_s21_mean_mode(self):
        f = np.array([1.0, 2.0, 3.0])
        s21 = np.array([-1.0, -2.0, -3.0])
        result = compute_scalar_objective(f, s21, mode="s21_mean")
        assert result == pytest.approx(-2.0)

    def test_with_band_limits(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s21 = np.array([-10.0, -1.0, -2.0, -3.0, -20.0])
        result = compute_scalar_objective(f, s21, fmin=2.0, fmax=4.0, mode="s21_min")
        assert result == pytest.approx(-3.0)

    def test_unknown_mode_raises(self):
        f = np.array([1.0, 2.0])
        s21 = np.array([-1.0, -2.0])
        with pytest.raises(ValueError, match="Unknown objective mode"):
            compute_scalar_objective(f, s21, mode="invalid")


# ---------------------------------------------------------------------------
# percentile_stats
# ---------------------------------------------------------------------------

class TestPercentileStats:
    def test_keys_present(self):
        x = np.arange(100, dtype=float)
        stats = percentile_stats(x)
        expected_keys = {"min", "max", "mean", "std", "var", "p01", "p05", "p50", "p95", "p99"}
        assert set(stats.keys()) == expected_keys

    def test_min_max(self):
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        stats = percentile_stats(x)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_single_element_std_nan(self):
        x = np.array([42.0])
        stats = percentile_stats(x)
        assert np.isnan(stats["std"])

    def test_median_correct(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = percentile_stats(x)
        assert stats["p50"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# perturb_network_vectorized
# ---------------------------------------------------------------------------

class TestPerturbVectorized:
    def test_output_shape(self, sample_network):
        rng = np.random.default_rng(42)
        s21_db, params = perturb_network_vectorized(sample_network, n=100, rng=rng)
        assert s21_db.shape == (100, len(sample_network.f))
        assert params.shape == (100, 3)

    def test_deterministic_with_seed(self, sample_network):
        s21_a, _ = perturb_network_vectorized(sample_network, n=50, rng=np.random.default_rng(42))
        s21_b, _ = perturb_network_vectorized(sample_network, n=50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s21_a, s21_b)

    def test_no_nan_in_output(self, sample_network):
        rng = np.random.default_rng(42)
        s21_db, params = perturb_network_vectorized(sample_network, n=200, rng=rng)
        assert not np.isnan(s21_db).any()
        assert not np.isnan(params).any()

    def test_perturbation_varies_across_samples(self, sample_network):
        rng = np.random.default_rng(42)
        s21_db, _ = perturb_network_vectorized(sample_network, n=50, rng=rng)
        # Different samples should produce different traces
        assert not np.allclose(s21_db[0], s21_db[1])


# ---------------------------------------------------------------------------
# perturb_network_single
# ---------------------------------------------------------------------------

class TestPerturbSingle:
    def test_returns_network(self, sample_network):
        out = perturb_network_single(sample_network, 0.5, 0.1, 0.2)
        assert isinstance(out, rf.Network)
        assert out.nports == 2
        assert len(out.f) == len(sample_network.f)

    def test_zero_perturbation_preserves_shape(self, sample_network):
        out = perturb_network_single(sample_network, 0.0, 0.0, 0.0)
        # S21 should be nearly identical with zero perturbation
        np.testing.assert_allclose(
            np.abs(out.s[:, 1, 0]),
            np.abs(sample_network.s[:, 1, 0]),
            rtol=1e-10,
        )

    def test_loss_reduces_s21(self, sample_network):
        out = perturb_network_single(sample_network, 3.0, 0.0, 0.0)
        # Adding 3 dB loss should reduce S21 magnitude
        original_s21_db = 20 * np.log10(np.abs(sample_network.s[:, 1, 0]) + 1e-15)
        perturbed_s21_db = 20 * np.log10(np.abs(out.s[:, 1, 0]) + 1e-15)
        assert np.mean(perturbed_s21_db) < np.mean(original_s21_db)
