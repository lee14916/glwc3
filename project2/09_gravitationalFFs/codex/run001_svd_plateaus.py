"""run001: exact-Q2 and binned SVD points from the saved stout10 plateaus.

    python3 -u run001_svd_plateaus.py

Uses the existing run001_gluon_plateau/matrix_elements.h5 and joint-fit curves.
The ratio and plateau fits are unchanged. Fixed diagonal weights and paired
jackknife fits follow run001_fit_gffs.py. Q2=0 is isolated in every scheme.
Binned points assume constant form factors within each bin; both existing
curve models are used to quantify the resulting approximation in a closure
test. No new form-factor curves are fitted to these extracted points.

    python3 -u run001_svd_plateaus.py --cumulative --cumulative-q2-max .2

Cumulative mode excludes Q2=0, fixes the smallest retained positive Q2, and
adds every distinct exact-Q2 group in order. It writes separate cumulative
outputs. Nested-cut significances are descriptive and unadjusted for scanning;
the paired samples retain their cross-cut correlations. This is not a Q2=0
extrapolation. --zoom-start changes display limits only.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2

import run001_fit_gffs as base

SECTORS = base.SECTORS
SCHEMES = ("exact", "width025", "width050", "kmeans34")
SOLVER_UTILITY = None

def exact_Q2_groups(moms, q2, atol=1e-12):
    """Return exact-Q2 groups in increasing Q2, without approximate physical bins.

    Each result has original row ``indices``, representative ``q2`` (midrange),
    ``q2_min/max/span``, ``invariants`` ordered as (q²,p_i²,p_f²), ``physical_keys``
    and ``n_momenta``. Known exact equalities are first checked: swapping leg
    norms preserves Q2, and equal-energy rows have Q2 fixed by spatial q² alone.
    Numerical merging requires the ENTIRE resulting Q2 span to be <=atol; it
    never chains neighbouring near-equalities into a wider approximate bin.
    """
    moms, q2 = np.asarray(moms), np.asarray(q2, dtype=float)
    if moms.ndim != 2 or moms.shape[1] != 6 or q2.shape != (len(moms),):
        raise ValueError("Expected moms[Nrows,6] and Q2[Nrows]")
    if not np.all(np.isfinite(moms)) or not np.all(moms == np.rint(moms)):
        raise ValueError("Momentum components must be finite integers")
    if not np.isfinite(atol) or atol <= 0 or np.any(~np.isfinite(q2)) or np.any(q2 < -atol):
        raise ValueError("Q2 must be finite/nonnegative and atol positive")
    if not len(moms):
        return []
    integer_moms = moms.astype(np.int64)
    pf, q = integer_moms[:, :3], integer_moms[:, 3:]
    pi = pf+q
    invariants = np.column_stack((np.sum(q*q, axis=1), np.sum(pi*pi, axis=1), np.sum(pf*pf, axis=1)))
    physical = {}
    for index, (nq, ni, nf) in enumerate(invariants):
        key = (f"equal_energy:q2={nq}" if ni == nf
               else f"q2={nq},leg_norms={min(ni,nf)},{max(ni,nf)}")
        physical.setdefault(key, []).append(index)
    buckets = []
    for key, indices in physical.items():
        values = q2[indices]
        low, high = float(values.min()), float(values.max())
        if high-low > atol:
            raise ValueError(f"Inconsistent Q2 for known identical kinematics {key}: span={high-low:g}")
        buckets.append(dict(indices=indices, low=low, high=high, keys=[key]))
    buckets.sort(key=lambda group: (group["low"], group["high"], group["keys"][0]))
    merged = []
    for bucket in buckets:
        if merged and max(merged[-1]["high"], bucket["high"])-merged[-1]["low"] <= atol:
            merged[-1]["indices"].extend(bucket["indices"])
            merged[-1]["keys"].extend(bucket["keys"])
            merged[-1]["high"] = max(merged[-1]["high"], bucket["high"])
        else:
            merged.append(bucket)
    result = []
    for group in merged:
        indices = np.asarray(sorted(group["indices"]), dtype=np.int64)
        low, high = group["low"], group["high"]
        result.append(dict(indices=indices, q2=max(0., (low+high)/2), q2_min=low, q2_max=high,
                           q2_span=high-low, invariants=np.unique(invariants[indices], axis=0),
                           physical_keys=sorted(group["keys"]),
                           n_momenta=len(np.unique(integer_moms[indices], axis=0))))
    return result


def _signed_svd(matrix):
    # Thin U avoids a large Nrow² allocation; full V is necessary when Nrow<3.
    u, singular, vh = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    for index in range(len(vh)):
        pivot = int(np.argmax(np.abs(vh[index])))
        if vh[index, pivot] < 0:
            vh[index] *= -1
            if index < u.shape[1]:
                u[:, index] *= -1
    return u, singular, vh


def solve_group(y_jk, K, utility, rank_rtol=1e-10, rank_atol=0.,
                identifiability_tol=1e-9, response_rtol=1e-7):
    """Fit one constant-(A20,B20,C20) group with fixed diagonal ME errors.

    Inputs are real paired ``y_jk[Njk,Nrows]`` and central ``K[Nrows,3]``.
    Returns ``ff_jk/mean/error/covariance/central`` with unidentifiable entries
    masked; measured ``combination_*`` refer to rows of ``combination_basis=V_r``.
    ``response_matrix`` is the diagnostic minimum-norm F response P (3,Nrows),
    and ``combination_response`` maps data to V_r F. Their action is checked
    against every utility-fitted sample and the mean fit before returning.
    Correlated samples propagate through the same fits; the fit weights alone
    are diagonal. No configurations or rows are silently discarded.
    """
    if np.iscomplexobj(y_jk) or np.iscomplexobj(K):
        raise ValueError("Supply the real signed scalar amplitudes and matching real K")
    y_jk, K = np.asarray(y_jk, dtype=float), np.asarray(K, dtype=float)
    if y_jk.ndim != 2 or K.shape != (y_jk.shape[1], 3) or len(y_jk) < 3 or not K.shape[0]:
        raise ValueError("Expected y_jk[Njk>=3,Nrows>=1] and K[Nrows,3]")
    tolerances = (rank_rtol, rank_atol, identifiability_tol, response_rtol)
    if not all(np.isfinite(value) for value in tolerances) or min(rank_rtol, rank_atol) < 0:
        raise ValueError("Rank tolerances must be finite/nonnegative")
    if identifiability_tol <= 0 or response_rtol <= 0:
        raise ValueError("Identifiability/response tolerances must be positive")
    mean, error = utility.jackme(y_jk)
    if (np.any(~np.isfinite(y_jk)) or np.any(~np.isfinite(K))
            or np.any(~np.isfinite(error)) or np.any(error <= 0)):
        raise ValueError("All selected samples/K must be finite and row errors strictly positive")
    inverse_sigma = 1/error
    weighted = K*inverse_sigma[:, None]
    u, singular, vh = _signed_svd(weighted)
    leading = float(singular[0]) if len(singular) else 0.
    machine_tolerance = leading*max(weighted.shape)*np.finfo(float).eps
    tolerance = max(rank_atol, leading*rank_rtol, machine_tolerance)
    rank = int(np.count_nonzero(singular > tolerance))
    vr, null = vh[:rank], vh[rank:]
    identifiable = np.linalg.norm(null, axis=0) < identifiability_tol if len(null) else np.ones(3, bool)
    reduced = K@vr.T
    recorded_warnings = []
    if rank:
        xs = np.arange(len(mean), dtype=float)

        def model(_xs, *parameters):
            return reduced@np.asarray(parameters)

        def jacobian(_xs, *parameters):
            return reduced

        def fit_one(values, start):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = utility.doSimpleFit(model, xs, values, sigma=error, p0=start,
                                             jackQ=False, jac=jacobian, maxfev=2000,
                                             ftol=1e-10, xtol=1e-10)
            recorded_warnings.extend(str(item.message) for item in caught)
            if not all(np.all(np.isfinite(fitted[i])) for i in (0, 1, 2, 3)):
                raise ValueError("The user fitter returned nonfinite parameters/covariance/chi2")
            return fitted

        central_fit = fit_one(mean, np.zeros(rank))
        central_beta = np.asarray(central_fit[0])
        fitter_covariance = np.asarray(central_fit[2])

        def fit_sample(values):
            fitted = fit_one(values, central_beta)
            return fitted[0], fitted[3]

        beta_jk, chi2_jk = utility.jackmap(fit_sample, y_jk)
        beta_jk = np.asarray(beta_jk).reshape(len(y_jk), rank)
        combo_response = (u[:, :rank].T*inverse_sigma[None, :])/singular[:rank, None]
        response = vr.T@combo_response
        diagnostic_beta = y_jk@combo_response.T
        diagnostic_central = combo_response@mean
        diagonal_beta_error = np.sqrt(np.sum((combo_response*error[None, :])**2, axis=1))
        scale = np.maximum(np.maximum(np.abs(beta_jk), np.abs(diagnostic_beta)), diagonal_beta_error[None, :])
        central_scale = np.maximum(np.maximum(np.abs(central_beta), np.abs(diagnostic_central)), diagonal_beta_error)
        normalized_difference = float(max(np.max(np.abs(beta_jk-diagnostic_beta)/np.maximum(scale, np.finfo(float).tiny)),
                                          np.max(np.abs(central_beta-diagnostic_central)/np.maximum(central_scale, np.finfo(float).tiny))))
        absolute_difference = float(max(np.max(np.abs(beta_jk-diagnostic_beta)),
                                        np.max(np.abs(central_beta-diagnostic_central))))
        if normalized_difference > response_rtol:
            raise RuntimeError(f"SVD diagnostic response disagrees with user fitter: normalized difference {normalized_difference:g}")
        combo_mean, combo_error, combo_covariance = utility.jackmec(beta_jk)
    else:
        central_beta, beta_jk = np.empty(0), np.empty((len(y_jk), 0))
        fitter_covariance = np.empty((0, 0))
        combo_mean, combo_error, combo_covariance = np.empty(0), np.empty(0), np.empty((0, 0))
        combo_response, response = np.empty((0, len(mean))), np.zeros((3, len(mean)))
        chi2_jk = np.sum((y_jk/error)**2, axis=1)
        absolute_difference = normalized_difference = 0.
    minimum_norm_jk = beta_jk@vr
    central_minimum_norm = central_beta@vr
    ff_jk, ff_central = minimum_norm_jk.copy(), central_minimum_norm.copy()
    ff_jk[:, ~identifiable], ff_central[~identifiable] = np.nan, np.nan
    ff_mean, ff_error = np.full(3, np.nan), np.full(3, np.nan)
    ff_covariance = np.full((3, 3), np.nan)
    if np.any(identifiable):
        fitted_mean, fitted_error, fitted_covariance = utility.jackmec(minimum_norm_jk[:, identifiable])
        ff_mean[identifiable], ff_error[identifiable] = fitted_mean, fitted_error
        ff_covariance[np.ix_(identifiable, identifiable)] = fitted_covariance
    prediction = reduced@central_beta
    retained_condition = float(singular[0]/singular[rank-1]) if rank else np.inf
    return dict(
        ff_names=("A20", "B20", "C20"), ff_jk=ff_jk, ff_central=ff_central,
        ff_mean=ff_mean, ff_error=ff_error, ff_covariance=ff_covariance, identifiable=identifiable,
        combination_jk=beta_jk, combination_central=central_beta, combination_mean=combo_mean,
        combination_error=combo_error, combination_covariance=combo_covariance,
        combination_basis=vr, null_basis=null, minimum_norm_jk=minimum_norm_jk,
        central_minimum_norm=central_minimum_norm,
        rank=rank, singular_values=singular, rank_tolerance=tolerance,
        rank_rtol=rank_rtol, rank_atol=rank_atol, machine_rank_tolerance=machine_tolerance,
        identifiability_tol=identifiability_tol,
        condition=retained_condition if rank == 3 else np.inf, retained_condition=retained_condition,
        ndata=len(mean), njk=len(y_jk), dof=len(mean)-rank,
        row_sigma=error, inverse_sigma=inverse_sigma, matrix_element_mean=mean,
        prediction=prediction, residual=mean-prediction, pull=(mean-prediction)/error,
        chi2=float(np.sum(((mean-prediction)/error)**2)), chi2_jk=np.asarray(chi2_jk),
        response_matrix=response, combination_response=combo_response,
        diagnostic_diagonal_covariance=(response*error[None, :])@(response*error[None, :]).T,
        fitter_combination_covariance=fitter_covariance,
        response_check=dict(passed=True, normalized_max=normalized_difference,
                            absolute_max=absolute_difference, tolerance=response_rtol),
        warning_count=len(recorded_warnings), warnings=sorted(set(recorded_warnings)),
        convention="Bare A20,B20,C20; combinations are V_r F. Minimum-norm diagnostics do not identify missing form factors.",
    )


def make_schemes(exact):
    """Partition distinct positive Q2 values, never splitting an exact group."""
    zero = [g for g in exact if abs(g["q2"]) <= 1e-12]
    positive = [g for g in exact if g["q2"] > 1e-12]
    values = np.asarray([g["q2"] for g in positive])

    def collect(labels):
        result = list(zero)
        for label in sorted(set(labels)):
            members = [g for g, lab in zip(positive, labels) if lab == label]
            result.append(dict(indices=np.concatenate([g["indices"] for g in members]),
                               q2=float(np.mean([g["q2"] for g in members])),
                               q2_min=min(g["q2_min"] for g in members),
                               q2_max=max(g["q2_max"] for g in members),
                               distinct_q2=len(members)))
        return sorted(result, key=lambda g: g["q2"])

    result = {"exact": exact}
    for name, width in (("width025", .025), ("width050", .05)):
        result[name] = collect(np.floor(values/width).astype(int))
    if len(values):
        count = min(33, len(values))  # 33 positive clusters plus isolated Q2=0.
        initial = np.quantile(values, (np.arange(count)+.5)/count)
        centers, labels = kmeans2(values, initial, iter=200, minit="matrix", missing="raise")
        # Relabel clusters by increasing Q2, independent of scipy's label order.
        ordered = np.argsort(centers, kind="stable")
        inverse = np.empty(count, dtype=int)
        inverse[ordered] = np.arange(count)
        result["kmeans34"] = collect(inverse[labels])
    else:
        result["kmeans34"] = list(zero)
    for scheme, groups in result.items():
        membership = np.concatenate([g["indices"] for g in groups])
        reference = np.concatenate([g["indices"] for g in exact])
        if not np.array_equal(np.sort(membership), np.sort(reference)):
            raise ValueError(f"{scheme} changed the retained matrix-element rows")
    return result


def make_cumulative_groups(exact, q2_max):
    """One nested group at every positive exact-Q2 entry, with no duplicate fits."""
    positive = [group for group in exact if group["q2"] > 1e-12
                and group["q2_max"] <= q2_max+1e-12]
    if not positive:
        raise ValueError("No positive exact-Q2 groups lie inside the cumulative range")
    positive.sort(key=lambda group: group["q2"])
    minimum = positive[0]["q2"]
    indices, centers, result = [], [], []
    for group in positive:
        indices.append(group["indices"])
        centers.append(group["q2"])
        result.append(dict(indices=np.concatenate(indices), q2=float(np.mean(centers)),
                           q2_min=positive[0]["q2_min"], q2_max=group["q2_max"],
                           actual_q2_max=group["q2_max"], requested_cut=group["q2_max"],
                           fixed_q2_min=minimum, distinct_q2=len(centers)))
    return result


def initialize_solver(path):
    global SOLVER_UTILITY
    SOLVER_UTILITY = base.load_utility(path)[0]


def calculate_group(payload):
    y_jk, coefficients = payload
    return solve_group(y_jk, coefficients, SOLVER_UTILITY)


def load_curves(path):
    result, predictions, parameters = {}, {}, {}
    with h5py.File(path, "r") as source:
        if not source.attrs["complete"]:
            raise ValueError("Joint-fit curve file is incomplete")
        report = json.loads(source["metadata/report"][()])
        tcut = report["settings"]["tcut"]
        for sector in SECTORS:
            result[sector], parameters[sector] = {}, {}
            for model in ("zlinear", "zquadratic"):
                group = source[f"selections/baseline/{sector}/{model}"]
                result[sector][model] = dict(grid=group["grid_Q2"][:],
                    mean=group["curve_mean"][:], error=group["curve_error"][:])
                parameters[sector][model] = group["parameter_mean"][:]
                if not np.all(np.isfinite(parameters[sector][model])):
                    raise ValueError("Closure requires identifiable existing curve parameters")
    return result, parameters, tcut


def add_closure(point, coefficients, q2, parameters, tcut, lowest_q2=None):
    response = point["response_matrix"]
    basis = point["combination_basis"]
    projection = basis.T@basis
    constant_error = np.max(np.abs(response@coefficients-projection))
    if constant_error > 1e-7:
        raise ValueError(f"Constant-form-factor closure failed: {constant_error}")
    point["constant_closure_max_abs"] = float(constant_error)
    point["closure"] = {}
    for model, pars in parameters.items():
        order = base.MODELS[model]
        matrix_elements = base.design_matrix(coefficients, q2, order, tcut)@pars
        center = (pars.reshape(3, order+1)@base.z_basis(np.asarray([point["q2"]]), order, tcut)[0])
        projected_bias = response@matrix_elements-projection@center
        bias = np.where(point["identifiable"], projected_bias, np.nan)
        combo_bias = point["combination_response"]@matrix_elements-basis@center
        if point["q2_max"]-point["q2_min"] < 1e-12 and np.max(np.abs(projected_bias)) > 1e-7:
            raise ValueError("Exact-Q2 closure failed")
        point["closure"][model] = dict(bias=bias,
            bias_over_error=np.divide(bias, point["ff_error"],
                out=np.full(3, np.nan), where=point["ff_error"] > 0),
            combination_bias=combo_bias)
        if lowest_q2 is not None:
            lowest = pars.reshape(3, order+1)@base.z_basis(np.asarray([lowest_q2]), order, tcut)[0]
            transport = response@matrix_elements-projection@lowest
            transport = np.where(point["identifiable"], transport, np.nan)
            point["closure"][model].update(
                lowest_q2_reference=lowest, lowest_q2_bias=transport,
                lowest_q2_bias_over_error=np.divide(transport, point["ff_error"],
                    out=np.full(3, np.nan), where=point["ff_error"] > 0),
                lowest_q2_combination_bias=point["combination_response"]@matrix_elements-basis@lowest)


def save_points(output, results, cfgs, report):
    output.create_dataset("cfgs", data=cfgs)
    for scheme, sectors in results.items():
        for sector, points in sectors.items():
            group = output.create_group(f"schemes/{scheme}/{sector}")
            for key in ("q2", "q2_min", "q2_max", "rank", "distinct_q2", "constant_closure_max_abs",
                        "ff_mean", "ff_error", "identifiable", "condition", "retained_condition",
                        "rank_tolerance", "ndata", "dof", "chi2", "warning_count"):
                group.create_dataset(key, data=np.asarray([p[key] for p in points]))
            for key in ("requested_cut", "fixed_q2_min", "actual_q2_max", "bin_width", "C_z"):
                if all(key in point for point in points):
                    group.create_dataset(key, data=np.asarray([point[key] for point in points]))
            singular = np.zeros((len(points), 3))
            for i, point in enumerate(points):
                singular[i, :len(point["singular_values"])] = point["singular_values"]
            group.create_dataset("singular_values", data=singular)
            for key in ("ff_jk", "ff_covariance"):
                group.create_dataset(key, data=np.stack([p[key] for p in points], axis=1 if key == "ff_jk" else 0),
                                     compression="gzip", shuffle=True)
            rank = [p["rank"] for p in points]
            basis = np.full((len(points), 3, 3), np.nan)
            combinations = np.full((len(cfgs), len(points), 3), np.nan)
            for i, point in enumerate(points):
                basis[i, :rank[i]] = point["combination_basis"]
                combinations[:, i, :rank[i]] = point["combination_jk"]
            group.create_dataset("combination_basis", data=basis)
            group.create_dataset("combination_jk", data=combinations, compression="gzip", shuffle=True)
            offsets = np.cumsum([0]+[len(p["row_indices"]) for p in points])
            group.create_dataset("row_offsets", data=offsets)
            group.create_dataset("row_indices", data=np.concatenate([p["row_indices"] for p in points]))
            group.create_dataset("response_matrix", data=np.concatenate([p["response_matrix"] for p in points], axis=1), compression="gzip")
            group.create_dataset("row_sigma", data=np.concatenate([p["row_sigma"] for p in points]))
            for model in ("zlinear", "zquadratic"):
                for key in ("bias", "bias_over_error"):
                    group.create_dataset(f"closure/{model}/{key}", data=np.asarray([p["closure"][model][key] for p in points]))
                combination_bias = np.full((len(points), 3), np.nan)
                for i, point in enumerate(points):
                    combination_bias[i, :rank[i]] = point["closure"][model]["combination_bias"]
                group.create_dataset(f"closure/{model}/combination_bias", data=combination_bias)
                if all("lowest_q2_bias" in point["closure"][model] for point in points):
                    for key in ("lowest_q2_reference", "lowest_q2_bias", "lowest_q2_bias_over_error"):
                        group.create_dataset(f"closure/{model}/{key}",
                            data=np.asarray([point["closure"][model][key] for point in points]))
                    lowest_combinations = np.full((len(points), 3), np.nan)
                    for i, point in enumerate(points):
                        lowest_combinations[i, :rank[i]] = point["closure"][model]["lowest_q2_combination_bias"]
                    group.create_dataset(f"closure/{model}/lowest_q2_combination_bias", data=lowest_combinations)
                    for label, original in (("bincenter_bias", "bias"),
                                            ("bincenter_bias_over_error", "bias_over_error"),
                                            ("bincenter_combination_bias", "combination_bias")):
                        group[f"closure/{model}/{label}"] = group[f"closure/{model}/{original}"]
    output.create_dataset("metadata/report", data=json.dumps(base.json_ready(report), allow_nan=False))
    output.create_dataset("metadata/producer_source", data=Path(__file__).read_text())


def summarize(points):
    summary = dict(groups=len(points), rank_counts=dict(Counter(p["rank"] for p in points)),
                   identifiable_points=dict(zip(("A20", "B20", "C20"),
                        np.sum([p["identifiable"] for p in points], axis=0).tolist())),
                   rows=sum(len(p["row_indices"]) for p in points),
                   max_width=max(p["q2_max"]-p["q2_min"] for p in points),
                   max_constant_closure=max(p["constant_closure_max_abs"] for p in points))
    summary["closure"] = {}
    for model in ("zlinear", "zquadratic"):
        ratios = np.abs(np.asarray([p["closure"][model]["bias_over_error"] for p in points]))
        summary["closure"][model] = {}
        for i, ff in enumerate(("A20", "B20", "C20")):
            finite = ratios[:, i][np.isfinite(ratios[:, i])]
            summary["closure"][model][ff] = dict(median=float(np.median(finite)),
                maximum=float(np.max(finite)), above_one_sigma=int(np.sum(finite > 1))) if len(finite) else None
    return summary


SVD_PLOT_SECTORS = ("equal", "unequal")
SVD_PLOT_COLORS = {"equal": "tab:blue", "unequal": "tab:orange"}
SVD_PLOT_SECTOR_LABELS = {"equal": r"$\mu=\nu$", "unequal": r"$\mu\ne\nu$"}
SVD_PLOT_SCHEMES = {
    "exact": ("o", r"exact $Q^2$"),
    "width025": ("o", r"$\Delta Q^2=0.025$"),
    "width050": ("s", r"$\Delta Q^2=0.05$"),
    "kmeans34": ("^", r"$k$-means"),
}


def _svd_point_arrays(points, ff):
    selected = []
    for point in points:
        if not point["identifiable"][ff]:
            continue
        x, lo, hi = (float(point[key]) for key in ("q2", "q2_min", "q2_max"))
        mean, error = float(point["ff_mean"][ff]), float(point["ff_error"][ff])
        if not np.all(np.isfinite([x, lo, hi, mean, error])) or error < 0:
            raise ValueError("An identifiable SVD estimate has invalid coordinates/errors")
        if lo > x+1e-12 or hi < x-1e-12:
            raise ValueError("A plotted Q2 center lies outside its bin")
        selected.append((x, mean, error, max(0, x-lo), max(0, hi-x)))
    if not selected:
        return np.empty((5, 0))
    return np.asarray(selected, dtype=float).T


def _svd_curve_limits(curves):
    """Common limits per GFF from the union of both sectors/models' 1-sigma bands."""
    limits = []
    minimum_spans = (.10, .10, .20)
    for ff, minimum_span in enumerate(minimum_spans):
        lower, upper = [], []
        for sector in SVD_PLOT_SECTORS:
            for curve in curves.get(sector, {}).values():
                mean = np.asarray(curve["mean"], dtype=float)[ff]
                error = np.asarray(curve["error"], dtype=float)[ff]
                good = np.isfinite(mean) & np.isfinite(error)
                if np.any(error[good] < 0):
                    raise ValueError("A global curve has a negative error")
                if np.any(good):
                    lower.append(float(np.min(mean[good]-error[good])))
                    upper.append(float(np.max(mean[good]+error[good])))
        lo, hi = (min(lower), max(upper)) if lower else (-minimum_span/2, minimum_span/2)
        center = (lo+hi)/2
        span = max(hi-lo, minimum_span)
        limits.append((center-.7*span, center+.7*span))
    return limits


def _svd_draw_curves(ax, sector_curves, sector, ff):
    color = SVD_PLOT_COLORS[sector]
    for name, linestyle, label in (("zlinear", "-", r"linear $z$"),
                                   ("zquadratic", "--", r"quadratic $z$")):
        if name not in sector_curves:
            continue
        curve = sector_curves[name]
        x = np.asarray(curve["grid"], dtype=float)
        mean = np.asarray(curve["mean"], dtype=float)[ff]
        error = np.asarray(curve["error"], dtype=float)[ff]
        if mean.shape != x.shape or error.shape != x.shape:
            raise ValueError("Global curve axes differ")
        ax.fill_between(x, mean-error, mean+error, color=color,
                        alpha=.14 if name == "zlinear" else .065, linewidth=0, zorder=0)
        ax.plot(x, mean, color=color, linestyle=linestyle, marker="", linewidth=1.4,
                label=label, zorder=3)


def _svd_plot_view(results, curves, outdir, filename, schemes, dpi, zoom_limits=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for row, sector in enumerate(SVD_PLOT_SECTORS):
        color = SVD_PLOT_COLORS[sector]
        for ff, ax in enumerate(axes[row]):
            _svd_draw_curves(ax, curves.get(sector, {}), sector, ff)
            for scheme in schemes:
                marker, label = SVD_PLOT_SCHEMES[scheme]
                x, mean, error, left, right = _svd_point_arrays(results[scheme].get(sector, []), ff)
                if not len(x):
                    continue
                exact = scheme == "exact"
                # Draw errors first, then opaque open markers. All observations
                # remain present even when the explicit zoom clips their errors.
                ax.errorbar(x, mean, yerr=error, xerr=None if exact else np.stack((left, right)),
                            fmt="none", ecolor=color, elinewidth=.5 if exact else .65,
                            alpha=.25 if exact else .50, capsize=0, zorder=1)
                ax.plot(x, mean, linestyle="", marker=marker, markersize=2.5 if exact else 3.3,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=.5 if exact else .65, label=label, zorder=4)
            if zoom_limits is not None:
                ax.set_ylim(*zoom_limits[ff])
            lo, hi = ax.get_ylim()
            if lo <= 0 <= hi:
                ax.axhline(0, color="0.5", linewidth=.7, linestyle=":", marker="", zorder=0)
            ax.set_ylabel(r"$"+("A", "B", "C")[ff]+r"_{20}^{g,\mathrm{bare}}(Q^2)$")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            if row == 1:
                ax.set_xlabel(r"$Q^2\;[\mathrm{GeV}^2]$")
        axes[row, 0].text(.035, .965, SVD_PLOT_SECTOR_LABELS[sector],
                          transform=axes[row, 0].transAxes, ha="left", va="top", fontsize=12)
        handles, labels = axes[row, 0].get_legend_handles_labels()
        if handles:
            axes[row, 0].legend(handles, labels, frameon=False, loc="upper right",
                               fontsize=10.5, markerscale=1.6, handlelength=2.2)
    fig.tight_layout()
    path = outdir/filename
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


def plot_svd_results(results, curves, outdir, dpi=200):
    """Return three PNG paths; no observation filtering or display thinning."""
    if any(scheme not in results for scheme in SVD_PLOT_SCHEMES):
        raise ValueError("Expected exact, width025, width050 and kmeans34 results")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "serif", "mathtext.fontset": "dejavuserif",
             "axes.labelsize": 18, "xtick.labelsize": 13, "ytick.labelsize": 13,
             "axes.linewidth": 1.3, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True, "figure.facecolor": "white",
             "axes.grid": False}
    with plt.rc_context(style):
        return [
            _svd_plot_view(results, curves, outdir, "exact_svd_full.png", ("exact",), dpi),
            _svd_plot_view(results, curves, outdir, "exact_svd_zoom.png", ("exact",), dpi,
                           zoom_limits=_svd_curve_limits(curves)),
            _svd_plot_view(results, curves, outdir, "binned_svd_comparison.png",
                           ("width025", "width050", "kmeans34"), dpi),
        ]


def reload_points(path, schemes=SCHEMES):
    results = {s: {} for s in schemes}
    with h5py.File(path, "r") as source:
        if not source.attrs["complete"]:
            raise ValueError("SVD output is incomplete")
        report = json.loads(source["metadata/report"][()])
        for scheme in schemes:
            for sector in SECTORS:
                g = source[f"schemes/{scheme}/{sector}"]
                keys = ("q2", "q2_min", "q2_max", "rank", "ff_mean", "ff_error", "identifiable")
                arrays = {k: g[k][:] for k in keys}
                for key in ("requested_cut", "fixed_q2_min", "actual_q2_max", "bin_width", "C_z"):
                    if key in g:
                        arrays[key] = g[key][:]
                results[scheme][sector] = [{k: v[i] for k, v in arrays.items()} for i in range(len(arrays["q2"]))]
    return results, report


def summarize_cumulative(points, scan_max):
    summary = summarize(points)
    summary["cutoffs"] = summary.pop("groups")
    summary["summed_nested_row_uses"] = summary.pop("rows")
    summary["unique_rows_at_max"] = len(points[-1]["row_indices"])
    summary["fixed_q2_min"] = points[0]["fixed_q2_min"]
    summary["actual_q2_max"] = points[-1]["q2_max"]
    summary["interpretation"] = "Nested, highly correlated interval estimates; counts are cutoffs, not independent observations. C/error crossings are unadjusted descriptive scan diagnostics."

    def describe(index):
        point = points[index]
        return dict(group_index=index, requested_cut=point["requested_cut"],
                    actual_q2_max=point["q2_max"], bin_width=point["bin_width"],
                    center_distinct_q2=point["q2"], distinct_q2=point["distinct_q2"],
                    rows=len(point["row_indices"]), rank=point["rank"],
                    ff_mean=point["ff_mean"], ff_error=point["ff_error"], C_z=point["C_z"])

    z = np.asarray([point["C_z"] for point in points])
    for level in (2, 3):
        crossings = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= level))
        summary[f"first_{level}sigma_crossing"] = describe(int(crossings[0])) if len(crossings) else None
    finite = np.flatnonzero(np.isfinite(z))
    summary["maximum_absolute_C_z"] = describe(int(finite[np.argmax(np.abs(z[finite]))])) if len(finite) else None
    summary["requested_endpoints"] = []
    for requested in sorted(set((.1, .2, float(scan_max)))):
        available = [i for i, point in enumerate(points) if point["q2_max"] <= requested+1e-12]
        if requested > scan_max+1e-12 or not available:
            summary["requested_endpoints"].append(dict(requested_cut=requested, group_index=None,
                status="above_scan_limit" if requested > scan_max+1e-12 else "below_first_positive_Q2"))
            continue
        index = available[-1]
        endpoint = describe(index)
        endpoint.update(requested_cut=requested, status="mapped_to_last_included_exact_Q2")
        endpoint["model_closure"] = {
            model: dict(bincenter_bias=values["bias"],
                        lowest_q2_transport_bias=values["lowest_q2_bias"],
                        lowest_q2_reference=values["lowest_q2_reference"])
            for model, values in points[index]["closure"].items()}
        summary["requested_endpoints"].append(endpoint)
    return summary


def save_cumulative_covariance(output, results, utility):
    """Retain covariance between nested C estimates with undefined entries masked."""
    for sector, points in results["cumulative"].items():
        samples = np.stack([point["ff_jk"][:, 2] for point in points], axis=1)
        valid = np.asarray([point["identifiable"][2] for point in points]) & np.all(np.isfinite(samples), axis=0)
        covariance = np.full((len(points), len(points)), np.nan)
        if np.any(valid):
            _, _, measured = utility.jackmec(samples[:, valid])
            covariance[np.ix_(valid, valid)] = measured
            expected = np.asarray([point["ff_error"][2] for point in points])[valid]**2
            if not np.allclose(np.diag(measured), expected, rtol=1e-9, atol=0):
                raise ValueError("C covariance diagonal disagrees with the paired jackknife errors")
        group = output[f"schemes/cumulative/{sector}"]
        group.create_dataset("C_cov", data=covariance, compression="gzip")
        group.create_dataset("C_cov_valid", data=valid)
        group.attrs["cross_cut_covariance"] = "C_cov indexes all nested cutoffs; invalid rows/columns are NaN. Full joint ff_jk permits covariance for every FF/cut pair."


def cumulative_zoom_limits(results, zoom_start):
    limits = []
    for ff, minimum_span in enumerate((.1, .1, .2)):
        intervals = []
        fallback = []
        for points in results["cumulative"].values():
            for point in points:
                mean, error = point["ff_mean"][ff], point["ff_error"][ff]
                if not point["identifiable"][ff] or not np.all(np.isfinite([mean, error])):
                    continue
                if error < 0:
                    raise ValueError("Negative cumulative error")
                interval = (float(mean-error), float(mean+error))
                fallback.append(interval)
                if point["q2_max"] >= zoom_start:
                    intervals.append(interval)
        intervals = intervals or fallback or [(-minimum_span/2, minimum_span/2)]
        low, high = min(pair[0] for pair in intervals), max(pair[1] for pair in intervals)
        span = max(high-low, minimum_span)
        center = (low+high)/2
        limits.append((center-.7*span, center+.7*span))
    return limits


def plot_cumulative_results(results, outdir, scan_max, zoom_start=.05, dpi=200):
    """Every nested cutoff remains in both views; zoom only sets display limits."""
    outdir = Path(outdir)
    limits = cumulative_zoom_limits(results, zoom_start)
    style = {"font.family": "serif", "mathtext.fontset": "dejavuserif",
             "axes.labelsize": 18, "xtick.labelsize": 13, "ytick.labelsize": 13,
             "axes.linewidth": 1.3, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True, "figure.facecolor": "white", "axes.grid": False}
    paths = []
    with plt.rc_context(style):
        for filename, zoom in (("cumulative_svd_full.png", None), ("cumulative_svd.png", limits)):
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
            for row, sector in enumerate(SECTORS):
                points = results["cumulative"][sector]
                x = np.asarray([point["q2_max"] for point in points])
                color = SVD_PLOT_COLORS[sector]
                for ff, ax in enumerate(axes[row]):
                    mean = np.asarray([point["ff_mean"][ff] if point["identifiable"][ff] else np.nan for point in points])
                    error = np.asarray([point["ff_error"][ff] if point["identifiable"][ff] else np.nan for point in points])
                    ax.fill_between(x, mean-error, mean+error, step="post", color=color, alpha=.2, linewidth=0)
                    ax.step(x, mean, where="post", color=color, linestyle="-", marker="", linewidth=1.4,
                            label=r"cumulative SVD ($1\sigma$)")
                    if zoom is not None:
                        ax.set_ylim(*zoom[ff])
                    lo, hi = ax.get_ylim()
                    if lo <= 0 <= hi:
                        ax.axhline(0, color=".5", linestyle=":", marker="", linewidth=.7)
                    if x[0] <= .1 <= scan_max:
                        ax.axvline(.1, color=".65", linestyle="--", marker="", linewidth=.8)
                    ax.set_xlim(max(0., x[0]-.02*(scan_max-x[0])), scan_max)
                    ax.set_ylabel(r"$\widehat{"+("A", "B", "C")[ff]+r"}_{20}^{g,\mathrm{bare}}$")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
                    if row == 1:
                        ax.set_xlabel(r"$Q^2_{\max}\;[\mathrm{GeV}^2]$")
                axes[row, 0].text(.035, .965, SVD_PLOT_SECTOR_LABELS[sector], transform=axes[row, 0].transAxes,
                                  ha="left", va="top", fontsize=12)
                axes[row, 0].legend(frameon=False, loc="upper right", fontsize=11)
            fig.tight_layout()
            path = outdir/filename
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(str(path))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        for ax, sector in zip(axes, SECTORS):
            points = results["cumulative"][sector]
            x = np.asarray([point["q2_max"] for point in points])
            z = np.asarray([point["C_z"] for point in points])
            ax.step(x, z, where="post", color=SVD_PLOT_COLORS[sector], linestyle="-", marker="", linewidth=1.4)
            ax.axhline(0, color=".5", linestyle="-", marker="", linewidth=.7)
            for level in (-3, -2, 2, 3):
                ax.axhline(level, color=".65", linestyle="--" if abs(level) == 2 else ":", marker="", linewidth=.8)
            if x[0] <= .1 <= scan_max:
                ax.axvline(.1, color=".65", linestyle="--", marker="", linewidth=.8)
            ax.set_xlim(max(0., x[0]-.02*(scan_max-x[0])), scan_max)
            ax.set_xlabel(r"$Q^2_{\max}\;[\mathrm{GeV}^2]$")
            ax.set_ylabel(r"$C_{20}^{g,\mathrm{bare}}/\sigma_C$")
            ax.text(.035, .965, SVD_PLOT_SECTOR_LABELS[sector], transform=ax.transAxes,
                    ha="left", va="top", fontsize=12)
        fig.tight_layout()
        path = outdir/"cumulative_C_significance.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(str(path))
    return paths, limits


def run_cumulative(args):
    started = time.monotonic()
    _, parameters, tcut = load_curves(args.curves)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/"cumulative_svd.h5"
    if args.plot_only:
        results, report = reload_points(path, ("cumulative",))
        scan_max = report["settings"]["cumulative_q2_max"]
    else:
        scan_max = args.cumulative_q2_max
        if path.exists() or path.with_suffix(".partial.h5").exists():
            raise FileExistsError("Cumulative output exists; use --plot-only or a distinct --output-dir")
        utility, utility_signature = base.load_utility(args.util)
        with h5py.File(args.input, "r") as source:
            if not source.attrs["complete"]:
                raise ValueError("Plateau input is incomplete")
            cfgs = source["cfgs"][:]
            sectors = source["sector"].asstr()[:]
            moms, k, q2 = source["moms"][:], source["K"][:], source["Q2"][:]
            keep = source["selections/baseline/fit_ok"][:]&~source["kinematic_null"][:]
            keep &= (q2 > 1e-12)&(q2 <= scan_max+1e-12)
            y_jk = source["selections/baseline/plateau_jk"][:]
        if len(cfgs) != len(y_jk):
            raise ValueError("Configuration metadata and paired jackknife sample count differ")
        report = dict(procedure_id="run001", mode="cumulative", input=str(args.input.resolve()),
            input_sha256=base.digest(args.input), curves=str(args.curves.resolve()),
            curves_sha256=base.digest(args.curves), producer_sha256=base.digest(__file__),
            utility_sha256=base.digest(args.util), utility_ast_sha256=utility_signature,
            jackknife_samples=len(cfgs), settings=dict(cumulative_q2_max=scan_max,
                thresholds="Every distinct positive exact-Q2 group enters once; no duplicate fits at requested endpoints.",
                zero_excluded=True, zero_tolerance=1e-12, exact_q2_tolerance=1e-12,
                rank_relative_tolerance=1e-10, weighting="fixed diagonal plateau errors; paired jackknife fits",
                sector_separation=True, source_sink_exchange=False, ratio_and_plateaus_unchanged=True,
                renormalized=False, kinematics="central K and Q2", threshold_indexing="zero-based",
                bin_center="arithmetic mean of DISTINCT included exact Q2 values",
                bincenter_closure="apply the estimator to K_i F_model(Q2_i), compare with the identifiable projection of F_model at the distinct-Q2 center",
                lowest_q2_transport="same estimator, compared with F_model at the fixed smallest positive Q2; no extrapolation to Q2=0",
                significance="Signed C/paired-JK error; nested, correlated and unadjusted for scanning. No calibrated scan p-value or selective stopping.",
                cross_cut_covariance="C_cov plus valid mask is stored; joint ff_jk retains all FF/cut correlations."),
            summaries={"cumulative": {}})
        results = {"cumulative": {}}
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_solver,
            initargs=(args.util,), mp_context=multiprocessing.get_context("spawn")) if args.workers > 1 else None
        if executor is None:
            initialize_solver(args.util)
        try:
            for sector in SECTORS:
                rows = np.flatnonzero(keep&(sectors == sector))
                groups = make_cumulative_groups(exact_Q2_groups(moms[rows], q2[rows]), scan_max)
                payloads = ((y_jk[:, rows[group["indices"]]], k[rows[group["indices"]]]) for group in groups)
                fitted = executor.map(calculate_group, payloads, chunksize=1) if executor else map(calculate_group, payloads)
                points = []
                for number, (group, solution) in enumerate(zip(groups, fitted), start=1):
                    indices = rows[group["indices"]]
                    point = dict(solution, **{key: group[key] for key in (
                        "q2", "q2_min", "q2_max", "requested_cut", "actual_q2_max", "fixed_q2_min", "distinct_q2")},
                        row_indices=indices, bin_width=group["q2_max"]-group["fixed_q2_min"])
                    point["C_z"] = (float(point["ff_mean"][2]/point["ff_error"][2])
                        if point["identifiable"][2] and point["ff_error"][2] > 0 else np.nan)
                    add_closure(point, k[indices], q2[indices], parameters[sector], tcut,
                                lowest_q2=point["fixed_q2_min"])
                    points.append(point)
                    if number % 25 == 0 or number == len(groups):
                        print(f"cumulative_progress: {sector} {number}/{len(groups)}", flush=True)
                results["cumulative"][sector] = points
                report["summaries"]["cumulative"][sector] = summarize_cumulative(points, scan_max)
        finally:
            if executor:
                executor.shutdown()
        report["elapsed_seconds"] = time.monotonic()-started
        partial = path.with_suffix(".partial.h5")
        with h5py.File(partial, "x") as output:
            output.attrs.update(schema="gff-plateau-cumulative-svd-v1", procedure_id="run001", complete=False)
            save_points(output, results, cfgs, report)
            save_cumulative_covariance(output, results, utility)
            output.attrs["complete"] = True
        partial.rename(path)
    report["plots"], limits = plot_cumulative_results(results, args.output_dir, scan_max, args.zoom_start, args.dpi)
    report["plot_display"] = dict(zoom_start=args.zoom_start, zoom_limits=limits,
        limits_rule="Union of identified mean plus/minus1sigma for cutoffs>=zoom_start, across sectors, with20percent padding; fallback to all estimates if none qualify.",
        all_cutoffs_preserved=True, no_global_curve_overlay=True)
    report["plot_producer_sha256"] = base.digest(__file__)
    report_path = args.output_dir/"cumulative_svd.json"
    report_path.write_text(json.dumps(base.json_ready(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(path), report=str(report_path), mode="cumulative")), flush=True)


def run(args):
    started = time.monotonic()
    curves, parameters, tcut = load_curves(args.curves)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/"svd_points.h5"
    if args.plot_only:
        results, report = reload_points(path)
    else:
        if path.exists() or path.with_suffix(".partial.h5").exists():
            raise FileExistsError("SVD output exists; use --plot-only or a distinct --output-dir")
        _, utility_signature = base.load_utility(args.util)
        with h5py.File(args.input, "r") as source:
            if not source.attrs["complete"]:
                raise ValueError("Plateau input is incomplete")
            cfgs = source["cfgs"][:]
            sectors = source["sector"].asstr()[:]
            moms, k, q2 = source["moms"][:], source["K"][:], source["Q2"][:]
            keep = source["selections/baseline/fit_ok"][:]&~source["kinematic_null"][:]
            keep &= q2 <= args.q2_max
            y_jk = source["selections/baseline/plateau_jk"][:]
        report = dict(procedure_id="run001", input=str(args.input.resolve()),
            input_sha256=base.digest(args.input), curves=str(args.curves.resolve()),
            curves_sha256=base.digest(args.curves), producer_sha256=base.digest(__file__),
            utility_sha256=base.digest(args.util), utility_ast_sha256=utility_signature,
            jackknife_samples=len(cfgs), settings=dict(q2_max=args.q2_max,
                exact_q2_tolerance=1e-12, rank_relative_tolerance=1e-10,
                weighting="fixed diagonal plateau errors, paired jackknife fits",
                sector_separation=True, source_sink_exchange=False, ratio_and_plateaus_unchanged=True,
                renormalized=False, kinematics="central K and Q2",
                bin_center="arithmetic mean of DISTINCT exact Q2 values",
                bins=dict(width025=.025, width050=.05,
                    kmeans34="33 positive-Q2 clusters plus isolated zero; uniformly weighted distinct Q2; deterministic quantile initialization and 200 Lloyd iterations"),
                closure="apply the bin's estimator to K_i F_model(Q2_i) and compare to the identifiable projection of F_model at the displayed center"),
            summaries={})
        results = {s: {} for s in SCHEMES}
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_solver,
                        initargs=(args.util,), mp_context=multiprocessing.get_context("spawn")) if args.workers > 1 else None
        if executor is None:
            initialize_solver(args.util)
        try:
            for sector in SECTORS:
                rows = np.flatnonzero(keep&(sectors == sector))
                exact = exact_Q2_groups(moms[rows], q2[rows])
                for g in exact:
                    g["distinct_q2"] = 1
                schemes = make_schemes(exact)
                for scheme, groups in schemes.items():
                    payloads = ((y_jk[:, rows[g["indices"]]], k[rows[g["indices"]]]) for g in groups)
                    fitted = executor.map(calculate_group, payloads, chunksize=2) if executor else map(calculate_group, payloads)
                    points = []
                    for i, (g, solution) in enumerate(zip(groups, fitted)):
                        indices = rows[g["indices"]]
                        point = dict(solution, q2=g["q2"], q2_min=g["q2_min"], q2_max=g["q2_max"],
                                     distinct_q2=g["distinct_q2"], row_indices=indices)
                        add_closure(point, k[indices], q2[indices], parameters[sector], tcut)
                        points.append(point)
                        if (i+1) % 100 == 0:
                            print(f"svd_progress: {sector} {scheme} {i+1}/{len(groups)}", flush=True)
                    results[scheme][sector] = points
                    report["summaries"].setdefault(scheme, {})[sector] = summarize(points)
                    print(json.dumps(base.json_ready(dict(sector=sector, scheme=scheme,
                        **report["summaries"][scheme][sector]))), flush=True)
        finally:
            if executor:
                executor.shutdown()
        report["elapsed_seconds"] = time.monotonic()-started
        partial = path.with_suffix(".partial.h5")
        with h5py.File(partial, "x") as output:
            output.attrs.update(schema="gff-plateau-svd-v1", procedure_id="run001", complete=False)
            save_points(output, results, cfgs, report)
            output.attrs["complete"] = True
        partial.rename(path)
    report["plots"] = plot_svd_results(results, curves, args.output_dir, args.dpi)
    report["plot_producer_sha256"] = base.digest(__file__)
    report_path = args.output_dir/"svd_points.json"
    report_path.write_text(json.dumps(base.json_ready(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(path), report=str(report_path))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=base.OUTPUT/"matrix_elements.h5")
    parser.add_argument("--curves", type=Path, default=base.OUTPUT/"stage2_bare_gffs.h5")
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    parser.add_argument("--util", type=Path, default=base.UTILITY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--q2-max", type=float, default=1.)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cumulative", action="store_true", help="nested positive-Q2 scan; writes separate cumulative outputs")
    parser.add_argument("--cumulative-q2-max", type=float, default=.2)
    parser.add_argument("--zoom-start", type=float, default=.05, help="cumulative display-limit reference only; all cutoffs remain fitted/plotted")
    args = parser.parse_args()
    if args.workers < 1 or args.dpi < 1 or not np.isfinite(args.q2_max) or args.q2_max <= 0:
        parser.error("Positive workers, dpi and Q2 cutoff are required")
    if not np.isfinite(args.cumulative_q2_max) or args.cumulative_q2_max <= 0 or not np.isfinite(args.zoom_start) or args.zoom_start < 0:
        parser.error("Positive cumulative Q2 limit and nonnegative finite zoom start are required")
    run_cumulative(args) if args.cumulative else run(args)


if __name__ == "__main__":
    main()
