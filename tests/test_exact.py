"""Exact tests on hand-computable cases and against brute-force enumeration."""

from itertools import permutations, product

import numpy as np
import pytest
from scipy import stats

from dyad_sync_metrics.exact import (
    MAX_EXACT_PERMUTATIONS,
    distinct_permutation_count,
    distinct_permutations,
    exact_paired_test,
    exact_spearman,
    nesting_caveat,
)


def brute_force_spearman_p(x, y) -> float:
    rho = stats.spearmanr(x, y).statistic
    rhos = [stats.spearmanr(x, p).statistic for p in permutations(y)]
    return float(np.mean([abs(r) >= abs(rho) - 1e-9 for r in rhos]))


@pytest.mark.parametrize("values", [[0, 0, 0, 1, 2], [1, 2, 3, 4], [5, 5, 5, 5], ["b", "a", "b"]])
def test_distinct_permutations_are_the_distinct_itertools_permutations(values):
    distinct = list(distinct_permutations(values))
    assert len(distinct) == len(set(distinct))
    assert set(distinct) == set(permutations(values))
    assert distinct_permutation_count(values) == len(distinct)


def test_distinct_permutation_count_formula():
    assert distinct_permutation_count([0] * 8 + [1, 2, 3, 4]) == 12 * 11 * 10 * 9
    assert distinct_permutation_count(list(range(12))) == 479_001_600


def test_perfect_monotone_relation_at_n4_gives_p_one_twelfth():
    # Only the identity and the reversed ordering reach |rho| = 1: p = 2 / 4!.
    result = exact_spearman([0.0, 1.0, 2.0, 3.0], [7.0, 5.0, 3.0, 1.0])
    assert result.n == 4
    assert result.rho == pytest.approx(-1.0)
    assert result.n_permutations == 24
    assert result.p_exact == pytest.approx(1 / 12)


def test_exact_p_with_ties_equals_brute_force_enumeration():
    x, y = [3, 1, 4, 1, 5], [0, 0, 2, 0, 1]
    result = exact_spearman(x, y)
    assert result.n_permutations == 20
    assert result.p_exact == pytest.approx(brute_force_spearman_p(x, y))
    assert result.rho == pytest.approx(stats.spearmanr(x, y).statistic)


@pytest.mark.parametrize("seed", range(5))
def test_exact_p_equals_brute_force_on_random_tied_data(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.integers(0, 4, 7).tolist(), rng.integers(0, 3, 7).tolist()
    if len(set(x)) < 2 or len(set(y)) < 2:
        pytest.skip("constant draw")
    assert exact_spearman(x, y).p_exact == pytest.approx(brute_force_spearman_p(x, y))


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [0.0, np.nan, 2.0, 3.0]),
        ([0.0, np.nan, 2.0, 3.0], [1, 2, 3, 4]),
    ],
)
def test_undefined_rho_gives_nan_not_p_zero(x, y):
    result = exact_spearman(x, y)
    assert np.isnan(result.rho)
    assert np.isnan(result.p_exact)
    assert result.n_permutations == 0
    assert result.n == 4


def test_exact_spearman_refuses_an_intractable_enumeration():
    with pytest.raises(ValueError, match="infeasible") as info:
        exact_spearman(list(range(12)), list(range(12)), name="scores")
    assert "scores" in str(info.value)
    assert str(distinct_permutation_count(list(range(12)))) in str(info.value)
    assert str(MAX_EXACT_PERMUTATIONS) in str(info.value)
    assert exact_spearman(range(6), range(6), max_permutations=720).n_permutations == 720
    with pytest.raises(ValueError, match="infeasible"):
        exact_spearman(range(6), range(6), max_permutations=719)


def test_exact_spearman_rejects_mismatched_input():
    with pytest.raises(ValueError, match="differ in length"):
        exact_spearman([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="one entry per observation"):
        exact_spearman([1, 2, 3], [1, 2, 3], groups=["a", "b"])


def test_nesting_caveat_appears_only_for_repeated_groups():
    assert nesting_caveat(None) is None
    assert nesting_caveat(["d0", "d1", "d2"]) is None
    caveat = nesting_caveat(["d0", "d0", "d1", "d1"])
    assert "4 observations come from 2 groups" in caveat
    result = exact_spearman([1, 2, 3, 4], [1, 3, 2, 4], groups=["d0", "d0", "d1", "d1"])
    assert result.nesting_caveat == caveat
    assert exact_spearman([1, 2, 3, 4], [1, 3, 2, 4]).nesting_caveat is None


def brute_force_sign_flip_p(diffs) -> float:
    observed = abs(sum(diffs))
    sums = [
        abs(sum(s * d for s, d in zip(signs, diffs, strict=True)))
        for signs in product([1, -1], repeat=len(diffs))
    ]
    return float(np.mean([s >= observed - 1e-9 for s in sums]))


def test_paired_test_on_a_hand_computable_case():
    # Usable differences 1, 2, -1, 3, 4: four of five positive.
    result = exact_paired_test([1, 2, 0, -1, 3, 4])
    assert (result.n_pairs, result.n_usable_pairs) == (6, 5)
    assert result.sign_test_p_two_sided == pytest.approx(2 * (5 + 1) / 32)
    assert result.sign_test_p_greater == pytest.approx((5 + 1) / 32)
    # Flipping a subset with sum s gives |9 - 2s|, at least 9 for s <= 0 (flip nothing,
    # the -1, or the 1 and the -1) or s >= 9 (their three complements): 6 of 32.
    assert result.permutation_p == pytest.approx(6 / 32)
    assert result.permutation_p == pytest.approx(brute_force_sign_flip_p([1, 2, -1, 3, 4]))
    assert result.sum_of_differences == 9.0


@pytest.mark.parametrize("seed", range(5))
def test_sign_flip_p_equals_brute_force(seed):
    diffs = np.random.default_rng(seed).normal(size=8).round(1).tolist()
    nonzero = [d for d in diffs if d != 0]
    assert exact_paired_test(diffs).permutation_p == pytest.approx(brute_force_sign_flip_p(nonzero))


def test_all_tied_pairs_give_p_one():
    result = exact_paired_test([0, 0, 0])
    assert result.n_usable_pairs == 0
    assert result.sign_test_p_two_sided == 1.0
    assert result.sign_test_p_greater == 1.0
    assert result.permutation_p == 1.0


def test_paired_test_guards():
    with pytest.raises(ValueError, match="NaN"):
        exact_paired_test([1.0, np.nan])
    with pytest.raises(ValueError, match="infeasible"):
        exact_paired_test(np.ones(21))
    groups = ["d0", "d0", "d1"]
    assert exact_paired_test([1, 2, 3], groups=groups).nesting_caveat == nesting_caveat(groups)
