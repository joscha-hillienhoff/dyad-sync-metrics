"""Exact permutation tests for small samples.

With a handful of dyads, asymptotic p-values (the t approximation behind
scipy.stats.spearmanr, the normal approximation of a signed-rank test) are not
trustworthy, and rank tests that drop ties lose much of a small sample. The tests
here enumerate the full permutation distribution instead:

- exact_spearman: Spearman's rho with a two-sided p-value from every distinct
  permutation of y against fixed x.
- exact_paired_test: the exact sign test and the exact sign-flip permutation test on
  paired differences.

Enumeration grows quickly, so each test computes its permutation count first and
raises above max_permutations rather than running for hours.

Both tests treat their observations as exchangeable. When observations are nested,
for example two conditions per dyad entered as separate rows, they are not. Pass the
cluster of each observation as groups and the result carries a nesting_caveat saying
so; the p-value itself does not account for the nesting.
"""

from collections import Counter
from collections.abc import Hashable, Iterator, Sequence
from itertools import islice
from math import factorial
from typing import NamedTuple

import numpy as np
from scipy import stats

MAX_EXACT_PERMUTATIONS = 2_000_000
"""Default ceiling on the number of permutations a test enumerates."""

_TOLERANCE = 1e-9
"""Relative tolerance when comparing a permuted statistic with the observed one, so
that floating-point noise does not drop a tie with the observed value."""


def distinct_permutations(values: Sequence) -> Iterator[tuple]:
    """Every distinct ordering of values exactly once, in lexicographic order.

    itertools.permutations yields n! orderings, with repeats whenever values contains
    ties. For a permutation test, enumerating each distinct ordering once gives the
    same p-value as all n!: every distinct ordering occurs equally often among the n!,
    namely prod(c_i!) times for repeat counts c_i, a property of the multiset alone.
    """
    counts = Counter(values)
    keys = sorted(counts)
    n = len(values)
    current: list = []

    def extend() -> Iterator[tuple]:
        if len(current) == n:
            yield tuple(current)
            return
        for key in keys:
            if counts[key]:
                counts[key] -= 1
                current.append(key)
                yield from extend()
                current.pop()
                counts[key] += 1

    return extend()


def distinct_permutation_count(values: Sequence) -> int:
    """How many orderings distinct_permutations yields: n! / prod(c_i!)."""
    total = factorial(len(values))
    for repeats in Counter(values).values():
        total //= factorial(repeats)
    return total


def nesting_caveat(groups: Sequence[Hashable] | None) -> str | None:
    """A caveat when several observations share a group, else None."""
    if groups is None:
        return None
    n_groups = len(set(groups))
    if n_groups == len(groups):
        return None
    return (
        f"the {len(groups)} observations come from {n_groups} groups, not "
        f"{len(groups)} independent units; this test treats them as exchangeable, which "
        "a nested design does not justify, and its p-value does not account for that"
    )


def _require_tractable(count: int, max_permutations: int, name: str, n: int) -> None:
    if count > max_permutations:
        raise ValueError(
            f"exact test is infeasible: {name} (n={n}) has {count} permutations, above "
            f"the ceiling of {max_permutations}"
        )


class SpearmanResult(NamedTuple):
    n: int
    rho: float
    """Spearman's rho; NaN when undefined."""
    p_exact: float
    """Two-sided exact p-value; NaN when rho is undefined."""
    n_permutations: int
    """Distinct permutations enumerated; 0 when nothing was."""
    nesting_caveat: str | None


def exact_spearman(
    x: Sequence[float],
    y: Sequence[float],
    *,
    groups: Sequence[Hashable] | None = None,
    max_permutations: int = MAX_EXACT_PERMUTATIONS,
    name: str = "y",
) -> SpearmanResult:
    """Spearman's rho of x and y with an exact two-sided p-value: the share of distinct
    permutations of y whose |rho| is at least the observed |rho|.

    Rho is undefined when x or y is constant or contains NaN. Every comparison with an
    undefined rho is false, so a naive enumeration would report p = 0, the strongest
    possible result read off no result. Here rho and p are both NaN in that case, and
    n_permutations is 0.

    Raises ValueError, naming name and the count, when y has more distinct
    permutations than max_permutations. groups, if given, is the cluster of each
    observation (see nesting_caveat).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y differ in length ({len(x)} and {len(y)})")
    if groups is not None and len(groups) != len(x):
        raise ValueError("groups must have one entry per observation")
    caveat = nesting_caveat(groups)
    n = len(x)
    if np.isnan(x).any() or np.isnan(y).any() or np.ptp(x) == 0 or np.ptp(y) == 0:
        return SpearmanResult(n, np.nan, np.nan, 0, caveat)
    count = distinct_permutation_count(y.tolist())
    _require_tractable(count, max_permutations, name, n)

    rho = float(stats.spearmanr(x, y).statistic)
    # Rho is the Pearson correlation of the ranks, and permuting y permutes its ranks
    # without changing their mean or spread, so |rho_perm| >= |rho| is equivalent to
    # |centred x ranks . permuted centred y ranks| >= |observed dot product|.
    rx = stats.rankdata(x) - (n + 1) / 2
    ry = stats.rankdata(y) - (n + 1) / 2
    rank_of = dict(zip(y.tolist(), ry.tolist(), strict=False))
    observed = abs(float(rx @ ry))
    threshold = observed - _TOLERANCE * max(observed, 1.0)
    extreme = 0
    perms = distinct_permutations(y.tolist())
    while chunk := list(islice(perms, 65536)):
        ranks = np.array([[rank_of[v] for v in perm] for perm in chunk])
        extreme += int((np.abs(ranks @ rx) >= threshold).sum())
    return SpearmanResult(n, rho, extreme / count, count, caveat)


class PairedTestResult(NamedTuple):
    n_pairs: int
    n_usable_pairs: int
    """Pairs with a non-zero difference; the sample size both tests actually rest on."""
    sign_test_p_two_sided: float
    sign_test_p_greater: float
    """One-sided: positive differences are more frequent than negative ones."""
    permutation_p: float
    """Two-sided exact sign-flip permutation p-value of the sum of differences."""
    sum_of_differences: float
    nesting_caveat: str | None


def exact_paired_test(
    differences: Sequence[float],
    *,
    groups: Sequence[Hashable] | None = None,
    max_permutations: int = MAX_EXACT_PERMUTATIONS,
) -> PairedTestResult:
    """Exact sign test and exact sign-flip permutation test on paired differences.

    Zero differences carry no sign and are dropped from both tests; n_usable_pairs
    says how many remain, and should be reported with any p-value. The sign test uses
    only the signs. The permutation test flips the sign of every subset of the usable
    differences (2^n patterns) and reports the share whose |sum| is at least the
    observed |sum|. With no usable pair both p-values are 1.

    Raises ValueError on a missing difference, and when 2^n exceeds max_permutations.
    """
    diffs = np.asarray(differences, dtype=float)
    if np.isnan(diffs).any():
        raise ValueError("differences contain NaN; drop or resolve incomplete pairs first")
    if groups is not None and len(groups) != len(diffs):
        raise ValueError("groups must have one entry per difference")
    caveat = nesting_caveat(groups)
    usable = diffs[diffs != 0]
    n = len(usable)
    total = float(diffs.sum())
    if n == 0:
        return PairedTestResult(len(diffs), 0, 1.0, 1.0, 1.0, total, caveat)
    _require_tractable(2**n, max_permutations, "differences", n)

    n_positive = int((usable > 0).sum())
    sign_two = stats.binomtest(n_positive, n, 0.5, alternative="two-sided").pvalue
    sign_greater = stats.binomtest(n_positive, n, 0.5, alternative="greater").pvalue

    # The sums of all 2^n sign patterns, built one difference at a time.
    sums = np.zeros(1)
    for d in usable:
        sums = np.concatenate([sums + d, sums - d])
    sums = np.abs(sums)
    observed = abs(float(usable.sum()))
    threshold = observed - _TOLERANCE * max(observed, 1.0)
    perm_p = float((sums >= threshold).mean())
    return PairedTestResult(
        len(diffs), n, float(sign_two), float(sign_greater), perm_p, total, caveat
    )
