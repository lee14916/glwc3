"""Preview separate diagonal/off-diagonal GFF kinematics without correlator data.

Run from any directory with the glwc3 Python environment::

    python momentum_preview.py
    python momentum_preview.py --ensembles b --source-n2-max 4 --sink-n2-max 6
    python momentum_preview.py --scenario pilot:4:4 --scenario extended:10:10

Defaults compare source/sink squared-momentum caps 4 and 10 at transfer cap 16.
These are provisional scenarios, not a verified inventory of available data.
Only joint spatial cubic rotations are folded. Source/sink exchange is not used.
The established 05_moments helpers supply coefficients, Q2, and symmetry filters.
Outputs are diagnostic PDF/PNG figures and CSV/JSON tables under _ignore/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parents[1] / "05_moments"
SECTORS = ("equal", "unequal")
SECTOR_LABELS = {"equal": r"Diagonal ($\mu=\nu$)", "unequal": r"Off-diagonal ($\mu\ne\nu$)"}
PROJS = ("P0", "Px", "Py", "Pz")
INSERTS = ("tt", "tx", "ty", "tz", "xx", "xy", "xz", "yy", "yz", "zz")


def load_legacy():
    """Honor util_moments' existing import-time relative metadata reads."""
    previous = Path.cwd()
    sys.path.insert(0, str(LEGACY))
    try:
        os.chdir(LEGACY)
        yu = importlib.import_module("util")
        yum = importlib.import_module("util_moments")
    finally:
        os.chdir(previous)
    if Path(yu.__file__).resolve() != LEGACY / "util.py":
        raise RuntimeError("A different util module was already imported.")
    return yu, yum


def lattice_vectors(cap):
    extent = math.isqrt(cap)
    return [v for v in product(range(-extent, extent + 1), repeat=3) if sum(x*x for x in v) <= cap]


def enumerate_groups(yu, source_cap, sink_cap, transfer_cap):
    """Each group retains its oriented (q2, source2, sink2) invariants."""
    representatives = set()
    seen = set()
    for sink in lattice_vectors(sink_cap):
        for source in lattice_vectors(source_cap):
            transfer = tuple(p - pp for p, pp in zip(source, sink))
            if sum(q*q for q in transfer) <= transfer_cap:
                mom = sink + transfer
                if mom in seen:
                    continue
                # All caps are rotation invariant. Remember the orbit before
                # invoking the legacy canonical ordering once for that orbit.
                seen.update(tuple(m) for m in yu.mom2moms(mom))
                representatives.add(tuple(yu.mom3pt2standard(mom)))
    groups = defaultdict(list)
    for mom in sorted(representatives, key=yu.getSortkey_mom3pt):
        key = tuple(yu.mom2n2qpp1(mom))
        assert key[1] <= source_cap and key[2] <= sink_cap
        groups[key].append(mom)
    return dict(sorted(groups.items()))


class Coefficients:
    def __init__(self, yu, yum, mass_mev, tolerance):
        self.yu, self.yum = yu, yum
        self.mass_mev, self.tolerance = mass_mev, tolerance
        self.restored_blocks = {}

    @lru_cache(maxsize=None)
    def selections(self, mom):
        # tt is explicitly excluded by useQ; prefiltering avoids its evaluation.
        selected = []
        mom_list = list(mom)  # Legacy mpi2standard_pi compares Python lists.
        for proj in PROJS:
            for insert in INSERTS:
                if insert == "tt":
                    continue
                # Skip symmetry duplicates before nonzeroQ's coefficient work.
                if (proj, insert) != self.yum.mpi2standard_pi(mom_list, proj, insert):
                    continue
                real_ok, imag_ok = self.yum.useQ(mom_list, proj, insert)
                if real_ok or imag_ok:
                    selected.append((proj, insert, real_ok, imag_ok))
        return tuple(selected)

    def evaluate(self, mom, ensemble, proj, insert):
        scale = 2 * np.pi / self.yu.ens2NL[ensemble]
        sink = np.asarray(mom[:3]) * scale
        source = (np.asarray(mom[:3]) + np.asarray(mom[3:])) * scale
        return self.yum.ME2FF(self.mass_mev / self.yu.ens2aInv[ensemble], source, sink, proj, insert)

    @lru_cache(maxsize=None)
    def blocks(self, mom, ensemble):
        if (mom, ensemble) in self.restored_blocks:
            return self.restored_blocks[(mom, ensemble)]
        rows = {s: [] for s in SECTORS}
        for proj, insert, real_ok, imag_ok in self.selections(mom):
            sector = "equal" if insert[0] == insert[1] else "unequal"
            value = self.evaluate(mom, ensemble, proj, insert)
            if real_ok:
                rows[sector].append(np.real(value))
            if imag_ok:
                rows[sector].append(np.imag(value))
        return {s: np.asarray(rows[s], dtype=float).reshape(-1, 3) for s in SECTORS}

    def metrics(self, rows):
        singular = np.linalg.svd(rows, compute_uv=False) if len(rows) else np.array([])
        padded = np.pad(singular, (0, 3-len(singular)))
        rank = int(np.count_nonzero(singular > self.tolerance))
        return {
            "rank": rank, "n_rows": len(rows),
            "s1": float(padded[0]), "s2": float(padded[1]), "s3": float(padded[2]),
            "condition": float(padded[0]/padded[2]) if rank == 3 else None,
        }

    def group_metrics(self, moms, ensemble):
        return {s: self.metrics(np.concatenate([self.blocks(m, ensemble)[s] for m in moms])) for s in SECTORS}


def validate(coefficients, groups, ensemble):
    """Compare optimization with old getRank and unreduced spatial-orbit rows."""
    yu, yum = coefficients.yu, coefficients.yum
    reference = [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (2, 1, 1), (4, 2, 2)]
    checks = []
    for key in reference:
        if key not in groups:
            continue
        moms = groups[key]
        reduced = coefficients.group_metrics(moms, ensemble)
        legacy_rows = {s: [] for s in SECTORS}
        raw_rows = {s: [] for s in SECTORS}
        for mom in moms:
            for proj in PROJS:
                for insert in INSERTS:
                    sector = "equal" if insert[0] == insert[1] else "unequal"
                    eligible = yum.useQ(list(mom), proj, insert)
                    if any(eligible):
                        value = coefficients.evaluate(mom, ensemble, proj, insert)
                        legacy_rows[sector].extend(func(value) for ok, func in zip(eligible, (np.real, np.imag)) if ok)
            # This deliberately includes tt, duplicate projectors and every
            # spatially rotated pair: a separate check of the symmetry reduction.
            for raw_mom in yu.mom2moms(mom):
                for proj in PROJS:
                    for insert in INSERTS:
                        sector = "equal" if insert[0] == insert[1] else "unequal"
                        value = coefficients.evaluate(raw_mom, ensemble, proj, insert)
                        raw_rows[sector].extend((np.real(value), np.imag(value)))
        for sector in SECTORS:
            old = coefficients.metrics(np.asarray(legacy_rows[sector]).reshape(-1, 3))
            raw = coefficients.metrics(np.asarray(raw_rows[sector]).reshape(-1, 3))
            actual = reduced[sector]
            if not (actual["rank"] == old["rank"] == raw["rank"]):
                raise AssertionError(f"Rank mismatch for {key}, {sector}: {actual}, {old}, {raw}")
            if not np.allclose([actual[f"s{i}"] for i in (1, 2, 3)],
                               [old[f"s{i}"] for i in (1, 2, 3)], atol=1e-12, rtol=1e-12):
                raise AssertionError(f"Coefficient selection mismatch for {key}, {sector}")
            checks.append({"group": key, "sector": sector, "rank": actual["rank"],
                           "legacy_rows": old["n_rows"], "raw_rows": raw["n_rows"]})
    return {"ensemble": ensemble, "checks": checks, "passed": True}


def export_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool_exact_q2(rows, groups, coefficients, ensemble, sector, cut, tolerance):
    """Stack only Q2 values equal within roundoff, never broad physics bins."""
    selected = sorted((r for r in rows if r["ensemble"] == ensemble and r["sector"] == sector
                       and r["sink_n2"] <= cut), key=lambda r: r["Q2_GeV2"])
    pools = []
    for row in selected:
        if not pools or row["Q2_GeV2"] - pools[-1][0]["Q2_GeV2"] > tolerance:
            pools.append([])
        pools[-1].append(row)
    pooled = []
    for pool in pools:
        keys = [(r["transfer_n2"], r["source_n2"], r["sink_n2"]) for r in pool]
        blocks = [coefficients.blocks(m, ensemble)[sector] for key in keys for m in groups[key]]
        pooled.append({"scenario": pool[0]["scenario"], "ensemble": ensemble, "sector": sector,
                       "sink_cut": cut, "Q2_GeV2": pool[0]["Q2_GeV2"],
                       "Q2_spread_GeV2": pool[-1]["Q2_GeV2"]-pool[0]["Q2_GeV2"],
                       "n_invariant_groups": len(pool), "n_sink0_groups": sum(r["sink_n2"] == 0 for r in pool),
                       "n_spatial_orbits": sum(r["n_spatial_orbits"] for r in pool),
                       "group_invariants": json.dumps(keys),
                       **coefficients.metrics(np.concatenate(blocks))})
    return pooled


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def coverage_plot(rows, ensembles, scenario, args, output, yu, pooled=False):
    name, source_cap, sink_cap = scenario
    height = 3.1*len(ensembles) + 1.2
    fig, axes = plt.subplots(len(ensembles), 2, figsize=(13.6, height),
                             sharex=True, sharey=True, squeeze=False)
    styles = {0: ("x", "none"), 1: ("o", "none"), 2: ("s", "white"), 3: ("D", None)}
    for row_index, ensemble in enumerate(ensembles):
        for col_index, sector in enumerate(SECTORS):
            ax = axes[row_index, col_index]
            selected = [r for r in rows if r["ensemble"] == ensemble and r["sector"] == sector
                        and r["Q2_GeV2"] <= args.q2_display_max]
            for rank, (marker, fill) in styles.items():
                for anchored in (False, True):
                    group = [r for r in selected if r["rank"] == rank and
                             (r["n_sink0_groups"] > 0 if pooled else r["sink_n2"] == 0) == anchored]
                    if not group:
                        continue
                    color = "#be6a19" if anchored else "#28669c"
                    xs = [r["Q2_GeV2"] for r in group]
                    ys = [r["sink_cut"] if pooled else r["sink_n2"] for r in group]
                    kwargs = {"marker": marker, "s": 28, "linewidths": 1.0, "zorder": 3 if anchored else 2}
                    if rank == 0:
                        ax.scatter(xs, ys, color=color, **kwargs)
                    else:
                        ax.scatter(xs, ys, edgecolors=color, facecolors=color if fill is None else fill, **kwargs)
            ax.set_xlim(-0.02, args.q2_display_max + 0.02)
            ax.set_ylim(-0.6, sink_cap + 0.6)
            ax.set_yticks(range(0, sink_cap+1, 1 if sink_cap <= 6 else 2))
            ax.text(0.98, 0.91, yu.ens2label[ensemble], transform=ax.transAxes, ha="right", fontsize=12)
            if row_index == 0:
                ax.set_title(SECTOR_LABELS[sector], fontsize=15)
            if col_index == 0:
                ax.set_ylabel(r"Sink cutoff $|\mathbf{n}'|^2_{\max}$" if pooled else r"Sink $|\mathbf{n}'|^2$", fontsize=13)
            if row_index == len(ensembles)-1:
                ax.set_xlabel(r"$Q^2$ [GeV$^2$]", fontsize=14)
    handles = [Line2D([], [], linestyle="None", marker=styles[r][0], color="#333333",
                       markerfacecolor="#333333" if r == 3 else "white",
                       label="rank 3: A20, B20, C20" if r == 3 else f"rank {r}") for r in (0, 1, 2, 3)]
    handles.extend([Line2D([], [], linestyle="None", marker="s", color=c, label=label)
                    for c, label in (("#be6a19", "contains sink = 0" if pooled else "sink = 0 kinematics"), ("#28669c", "moving sink only" if pooled else "moving sink"))])
    level = "Exact-Q2 pooled ranks" if pooled else "Per-invariant-group ranks"
    fig.suptitle(f"{level}: {name} (source n2 <= {source_cap}, sink n2 <= {sink_cap})", fontsize=15, y=1-0.06/height)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1-0.42/height), fontsize=10)
    note = "Each point pools exactly matching Q2 at the specified cutoff." if pooled else "Each point is one oriented invariant group."
    fig.text(0.5, 0.002, f"Kinematic rank only; mN = {args.mass_mev:g} MeV; transfer n2 <= {args.transfer_n2_max}. {note}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1-1.12/height))
    save_figure(fig, output / f"{'coverage_exactQ2' if pooled else 'coverage'}_{name}")


def nested_plot(rows, ensembles, scenario, args, output, yu):
    name, source_cap, _ = scenario
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1), sharey=True)
    for ax, sector in zip(axes, SECTORS):
        for index, ensemble in enumerate(ensembles):
            subset = [r for r in rows if r["ensemble"] == ensemble and r["sector"] == sector]
            ax.plot([r["sink_cut"] for r in subset], [r["rank3_Q2_in_display"] for r in subset],
                    color=yu.colors8[index], marker="o", linestyle="-", markersize=4,
                    label=yu.ens2label[ensemble])
        ax.set_title(SECTOR_LABELS[sector], fontsize=15)
        ax.set_xlabel(r"Sink cutoff $|\mathbf{n}'|^2_{\max}$", fontsize=13)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel(r"Full-rank distinct $Q^2$", fontsize=13)
    axes[-1].legend(fontsize=10)
    fig.suptitle(f"{name}: fixed source n2 <= {source_cap}, Q2 <= {args.q2_display_max:g} GeV2", fontsize=14)
    fig.text(0.5, 0.005, "Exact-Q2 pooled kinematic ranks; correlated momenta do not represent independent information or a precision forecast.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    save_figure(fig, output / f"nested_{name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensembles", nargs="+", choices=("b", "c", "d", "e"), default=["b", "c", "d", "e"])
    parser.add_argument("--scenario", action="append", metavar="NAME:SOURCE_N2:SINK_N2")
    parser.add_argument("--source-n2-max", type=int)
    parser.add_argument("--sink-n2-max", type=int)
    parser.add_argument("--transfer-n2-max", type=int, default=16)
    parser.add_argument("--sink-cuts", help="Comma-separated intermediate nested cutoffs; 0 and scenario maximum always included. Default: every integer.")
    parser.add_argument("--q2-display-max", type=float, default=1.0, help="Display/count window only; the group CSV retains all Q2.")
    parser.add_argument("--mass-mev", type=float, default=939.0)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10, help="Absolute singular-value cutoff, matching general.ipynb.")
    parser.add_argument("--q2-equality-tolerance", type=float, default=1e-12, help="Exact-Q2 numerical equality tolerance in GeV2; not a physics bin width.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "_ignore" / "momentum_preview")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.scenario and (args.source_n2_max is not None or args.sink_n2_max is not None):
        parser.error("Use either --scenario or the individual source/sink caps.")
    if args.scenario:
        try:
            scenarios = [(n, int(p), int(pp)) for n, p, pp in (s.split(":") for s in args.scenario)]
        except ValueError:
            parser.error("Each scenario must have the form NAME:SOURCE_N2:SINK_N2.")
    elif args.source_n2_max is not None or args.sink_n2_max is not None:
        scenarios = [("custom", 4 if args.source_n2_max is None else args.source_n2_max,
                      4 if args.sink_n2_max is None else args.sink_n2_max)]
    else:
        scenarios = [("restricted", 4, 4), ("prospective", 10, 10)]
    if len({n for n, _, _ in scenarios}) != len(scenarios):
        parser.error("Scenario names must be unique.")
    if any(not n or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in n)
           for n, _, _ in scenarios):
        parser.error("Scenario names may contain letters, digits, underscores, and hyphens.")
    if any(min(p, pp) < 0 for _, p, pp in scenarios) or args.transfer_n2_max < 0:
        parser.error("Squared-momentum caps must be nonnegative.")
    if min(args.q2_display_max, args.mass_mev, args.rank_tolerance, args.q2_equality_tolerance) <= 0:
        parser.error("Q2 display limit, mass, and tolerances must be positive.")
    args.ensembles = list(dict.fromkeys(args.ensembles))
    if args.sink_cuts:
        try:
            args.sink_cuts = sorted(set(int(x) for x in args.sink_cuts.split(",")))
        except ValueError:
            parser.error("--sink-cuts must contain comma-separated integers.")
        if not args.sink_cuts or min(args.sink_cuts) < 0:
            parser.error("Nested sink cuts must be nonnegative.")
    return args, scenarios


def main():
    args, scenarios = parse_args()
    print("Loading established 05_moments utilities", flush=True)
    yu, yum = load_legacy()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coefficients = Coefficients(yu, yum, args.mass_mev, args.rank_tolerance)
    largest_source = max(p for _, p, _ in scenarios)
    largest_sink = max(pp for _, _, pp in scenarios)
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__).resolve(), LEGACY / "util.py", LEGACY / "util_moments.py")}
    cache_identity = {
        "scientific_source_sha256": hashlib.sha256("\n".join(inspect.getsource(obj) for obj in
                                                              (lattice_vectors, enumerate_groups, Coefficients, validate)).encode()).hexdigest(),
        "legacy_sha256": {p: h for p, h in source_hashes.items() if p != str(Path(__file__).resolve())},
        "scenarios": scenarios, "ensembles": args.ensembles, "mass_mev": args.mass_mev,
        "rank_tolerance": args.rank_tolerance, "transfer_cap": args.transfer_n2_max,
        "sectors": SECTORS, "projectors": PROJS, "insertions": INSERTS,
        "numpy_version": np.__version__,
    }
    cache_path = output / "coefficients.pkl"
    checkpoint = yu.load_pkl(str(cache_path)) if cache_path.is_file() else None
    if checkpoint and checkpoint["identity"] == cache_identity:
        groups, results = checkpoint["groups"], checkpoint["results"]
        coefficients.restored_blocks = checkpoint["blocks"]
        validation = checkpoint["validation"]
        print("Loaded coefficient checkpoint with matching scientific source and inputs", flush=True)
    else:
        groups = enumerate_groups(yu, largest_source, largest_sink, args.transfer_n2_max)
        # Avoid computing cross-scenario combinations that no requested cap allows.
        groups = {k: v for k, v in groups.items() if any(k[1] <= p and k[2] <= pp for _, p, pp in scenarios)}
        print(f"Spatial orbits: {sum(map(len, groups.values()))}; oriented invariant groups: {len(groups)}", flush=True)
        results = {}
        for ensemble in args.ensembles:
            for index, (key, moms) in enumerate(groups.items()):
                results[(ensemble, key)] = coefficients.group_metrics(moms, ensemble)
                if (index+1) % 100 == 0 or index+1 == len(groups):
                    print(f"{ensemble}: {index+1}/{len(groups)} groups", flush=True)
        validation = None
    if not args.skip_validation and validation is None:
        validation = validate(coefficients, groups, args.ensembles[0])
    yu.save_pkl(str(cache_path), {"identity": cache_identity, "groups": groups, "results": results,
                                "blocks": {(m, e): coefficients.blocks(m, e) for e in args.ensembles
                                           for moms in groups.values() for m in moms}, "validation": validation})
    if validation:
        print(f"Validation passed: {len(validation['checks'])} sector/group checks", flush=True)
    summary = {
        "scope": "Kinematic rank only; no measured correlators or statistical precision forecast.",
        "caps_status": "Provisional scenarios; actual source, sink and loop momentum inventories are unverified.",
        "restricted_scenario_note": "Both-leg cap 4 is conservative pending ratio two-point inventory; it is not a proven hard data limit.",
        "connected_coverage_note": "Sink=0 denotes potential connected kinematics only. Measured connected momenta, operators and projectors are unverified; ranks use the full legacy permitted row set.",
        "legacy_util": str(LEGACY / "util.py"), "legacy_util_moments": str(LEGACY / "util_moments.py"),
        "source_sha256": source_hashes,
        "coefficient_cache_identity": cache_identity,
        "sectors": list(SECTORS), "source_sink_exchange": False,
        "rank_definition": "Rank of the three-column A20, B20, C20 coefficient matrix. Rank 3 identifies all three kinematically.",
        "group_invariants": ["transfer_n2", "source_n2", "sink_n2"],
        "mass_MeV": args.mass_mev, "absolute_rank_tolerance": args.rank_tolerance,
        "transfer_n2_max": args.transfer_n2_max, "Q2_display_max_GeV2": args.q2_display_max,
        "Q2_equality_tolerance_GeV2": args.q2_equality_tolerance,
        "Q2_pooling": "Within each ensemble and nested sink cut, stack invariant groups with Q2 equal within numerical tolerance; no coarse bins.",
        "ensembles": {e: {"NL": yu.ens2NL[e], "a_fm": yu.ens2a[e]} for e in args.ensembles},
        "counts_note": "Orbit and group counts are nominal, correlated, and not counts of independent measurements.",
        "singular_values_note": "Unweighted kinematic matrix, including row normalization; not statistical information eigenvalues.",
        "validation": validation, "scenarios": [],
    }
    with plt.rc_context({"axes.labelsize": 13, "axes.titlesize": 15, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 1.1}):
        for scenario in scenarios:
            name, source_cap, sink_cap = scenario
            rows, nested, pooled_rows = [], [], []
            selected_groups = {k: v for k, v in groups.items() if k[1] <= source_cap and k[2] <= sink_cap}
            for ensemble in args.ensembles:
                for key, moms in selected_groups.items():
                    for sector in SECTORS:
                        rows.append({"scenario": name, "ensemble": ensemble, "sector": sector,
                                     "transfer_n2": key[0], "source_n2": key[1], "sink_n2": key[2],
                                     "Q2_GeV2": float(yum.mom2Q2(moms[0], ensemble, args.mass_mev/yu.ens2aInv[ensemble])),
                                     "n_spatial_orbits": len(moms),
                                     "n_oriented_pairs": sum(len(yu.mom2moms(m)) for m in moms),
                                     "momenta_sink_transfer": json.dumps(moms),
                                     **results[(ensemble, key)][sector]})
                cuts = sorted({0, sink_cap, *(c for c in args.sink_cuts if c <= sink_cap)}) if args.sink_cuts else list(range(sink_cap+1))
                for sector in SECTORS:
                    for cut in cuts:
                        subset = [r for r in rows if r["ensemble"] == ensemble and r["sector"] == sector and r["sink_n2"] <= cut]
                        shown = [r for r in subset if r["Q2_GeV2"] <= args.q2_display_max]
                        pooled = pool_exact_q2(rows, selected_groups, coefficients, ensemble, sector, cut, args.q2_equality_tolerance)
                        pooled_rows.extend(pooled)
                        pooled_shown = [r for r in pooled if r["Q2_GeV2"] <= args.q2_display_max]
                        nested.append({"scenario": name, "ensemble": ensemble, "sector": sector,
                                       "source_cap": source_cap, "sink_cut": cut,
                                       "n_groups_total": len(subset), "n_groups_in_display": len(shown),
                                       "n_spatial_orbits_in_display": sum(r["n_spatial_orbits"] for r in shown),
                                       "n_distinct_Q2_total": len(pooled), "n_distinct_Q2_in_display": len(pooled_shown),
                                       **{f"rank{rank}_groups_in_display": sum(r["rank"] == rank for r in shown) for rank in range(4)},
                                       **{f"rank{rank}_Q2_in_display": sum(r["rank"] == rank for r in pooled_shown) for rank in range(4)}})
            export_csv(output / f"groups_{name}.csv", rows)
            export_csv(output / f"nested_{name}.csv", nested)
            export_csv(output / f"exactQ2_{name}.csv", pooled_rows)
            coverage_plot(rows, args.ensembles, scenario, args, output, yu)
            coverage_plot(pooled_rows, args.ensembles, scenario, args, output, yu, pooled=True)
            nested_plot(nested, args.ensembles, scenario, args, output, yu)
            summary["scenarios"].append({"name": name, "source_cap": source_cap, "sink_cap": sink_cap,
                                         "groups_per_ensemble": len(selected_groups), "nested": nested})
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
