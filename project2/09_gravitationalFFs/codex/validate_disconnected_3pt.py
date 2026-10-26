"""Compare compact disconnected C3 output with existing legacy momentum files.

Example::

    python validate_disconnected_3pt.py --compact CFG.h5 \
        --legacy-avgmore OLD_AVGMORE_DIR --legacy-avgsrc OLD_AVGSRC_DIR \
        --currents j+,js --stouts 10 --tfs 2,8,22 --report comparison.json

Directories are searched for the compact file's canonical disc_<momentum>.h5
names. Explicit legacy files can also be supplied. Only selected datasets and
one momentum row at a time are read. No production module is imported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import h5py
import numpy as np


PROJS = ("P0", "Px", "Py", "Pz")
TINY = np.finfo(np.float64).tiny


def strings(dataset):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]]


def integer_selection(text):
    if ":" in text:
        low, high = map(int, text.split(":"))
        return list(range(low, high+1))
    return [int(x) for x in text.split(",") if x]


def channel_name(name):
    return name[:-5] if name.endswith(";disc") else name


def data_keys(group):
    result = {}
    for key in group:
        try:
            channel, tf = key.rsplit("_", 1)
            slot = (channel_name(channel), int(tf))
        except ValueError as error:
            raise ValueError(f"Unrecognized data key {group.name}/{key}") from error
        if slot in result:
            raise ValueError(f"Duplicate normalized channel/time key {slot} in {group.name}")
        result[slot] = key
    return result


def momentum_rows(dataset):
    rows = dataset[:]
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"Expected an integer (N,6) momentum table: {dataset.name}")
    rows = [tuple(map(int, row)) for row in rows]
    if len(set(rows)) != len(rows):
        raise ValueError(f"Duplicate momentum rows: {dataset.name}")
    return rows


def reference_files(paths, canonical_moms):
    files = []
    for path in paths:
        if path.is_dir():
            candidates = [path/("disc_"+",".join(map(str, mom))+".h5") for mom in canonical_moms]
            selected = [p for p in candidates if p.is_file()]
            if not selected:
                raise ValueError(f"No matching legacy momentum filenames in {path}")
            files.extend(selected)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"Legacy reference does not exist: {path}")
    return list(dict.fromkeys(p.resolve() for p in files))


def vector_norm(values):
    """Scaled norm avoids squaring tiny or very large complex correlators."""
    magnitude = np.abs(values).ravel()
    scale = float(magnitude.max(initial=0))
    return 0.0 if scale == 0 else scale*float(np.sqrt(np.sum((magnitude/scale)**2)))


def empty_metric():
    return dict(momentum_rows=0, elements=0, absmax=0.0, scale=0.0,
                difference_l2=0.0, legacy_l2=0.0, compact_l2=0.0,
                max_per_momentum_normalized_max=0.0, nonfinite_rows=0,
                worst_momentum=None, worst_legacy_file=None)


def accumulate(metric, compact, legacy, momentum, filename):
    if compact.shape != legacy.shape:
        raise ValueError(f"Shape mismatch for {filename}, {momentum}: {compact.shape} versus {legacy.shape}")
    metric["momentum_rows"] += 1
    metric["elements"] += compact.size
    if not (np.all(np.isfinite(compact)) and np.all(np.isfinite(legacy))):
        metric["nonfinite_rows"] += 1
        metric["worst_momentum"], metric["worst_legacy_file"] = momentum, str(filename)
        return
    difference = compact-legacy
    maximum = float(np.abs(difference).max(initial=0))
    scale = max(float(np.abs(compact).max(initial=0)), float(np.abs(legacy).max(initial=0)), TINY)
    normalized = maximum/scale
    metric["absmax"] = max(metric["absmax"], maximum)
    metric["scale"] = max(metric["scale"], scale)
    metric["difference_l2"] = math.hypot(metric["difference_l2"], vector_norm(difference))
    metric["legacy_l2"] = math.hypot(metric["legacy_l2"], vector_norm(legacy))
    metric["compact_l2"] = math.hypot(metric["compact_l2"], vector_norm(compact))
    if normalized > metric["max_per_momentum_normalized_max"]:
        metric["max_per_momentum_normalized_max"] = normalized
        metric["worst_momentum"], metric["worst_legacy_file"] = momentum, str(filename)


def compare_kind(compact, kind, paths, slots, canonical_moms, metrics):
    prefix = "" if kind == "avgmore" else "avgsrc/"
    if prefix+"moms" not in compact or prefix+"data" not in compact:
        raise ValueError(f"Compact file lacks {kind} data")
    compact_moms = momentum_rows(compact[prefix+"moms"])
    lookup = {mom: i for i, mom in enumerate(compact_moms)}
    compact_keys = data_keys(compact[prefix+"data"])
    projs, inserts = strings(compact["projs"]), strings(compact["inserts"])
    files = reference_files(paths, canonical_moms)
    matched = set()
    for path in files:
        with h5py.File(path, "r") as legacy:
            old_moms = momentum_rows(legacy["moms"])
            pairs = [(i, lookup[mom], mom) for i, mom in enumerate(old_moms) if mom in lookup]
            if not pairs:
                raise ValueError(f"No momentum-table overlap with compact {kind} data: {path}")
            if any(mom in matched for _, _, mom in pairs):
                raise ValueError(f"Repeated legacy momentum coverage: {path}")
            matched.update(mom for _, _, mom in pairs)
            old_projs = strings(legacy["projs"]) if "projs" in legacy else list(PROJS)
            old_inserts = strings(legacy["inserts"])
            if set(old_projs) != set(projs) or set(old_inserts) != set(inserts):
                raise ValueError(f"Projector/insertion basis differs: {path}")
            proj_indices = [old_projs.index(x) for x in projs]
            insert_indices = [old_inserts.index(x) for x in inserts]
            old_keys = data_keys(legacy["data"])
            for channel, tf in slots:
                slot = (channel, tf)
                if slot not in compact_keys or slot not in old_keys:
                    raise ValueError(f"Missing selected channel/time {slot} in compact {kind} or {path}")
                new = compact[prefix+"data/"+compact_keys[slot]]
                old = legacy["data/"+old_keys[slot]]
                expected_new = (tf+1, len(compact_moms), len(projs), len(inserts))
                expected_old = (tf+1, len(old_moms), len(old_projs), len(old_inserts))
                if new.shape != expected_new or old.shape != expected_old:
                    raise ValueError(f"Unexpected data axes for {slot}: {new.shape}, {old.shape}")
                metric = metrics[(kind, channel, tf)]
                for old_i, new_i, mom in pairs:
                    old_values = old[:, old_i, :, :][:, proj_indices, :][:, :, insert_indices]
                    accumulate(metric, new[:, new_i, :, :], old_values, mom, path)
    return dict(kind=kind, files=[str(p) for p in files],
                compact_momenta=len(compact_moms), compared_momenta=len(matched))


def validate(args):
    if not math.isfinite(args.rtol) or args.rtol <= 0:
        raise ValueError("--rtol must be finite and positive")
    metrics = defaultdict(empty_metric)
    with h5py.File(args.compact, "r") as compact:
        canonical_moms = momentum_rows(compact["moms"])
        available = data_keys(compact["data"])
        filters_given = args.currents is not None or args.stouts is not None
        channels = set()
        if args.currents is not None:
            currents = {channel_name(x) for x in args.currents.split(",") if x}
            if not currents.issubset({"j+", "js", "jc"}):
                raise ValueError("--currents accepts j+,js,jc (optionally with ;disc)")
            channels.update(currents)
        if args.stouts is not None:
            stouts = integer_selection(args.stouts)
            if any(s < 0 for s in stouts):
                raise ValueError("Stout steps must be nonnegative")
            channels.update(f"jg;stout{s}" for s in stouts)
        if not filters_given:
            channels = {ch for ch, _ in available}
        tfs = sorted(set(integer_selection(args.tfs))) if args.tfs is not None else sorted({tf for _, tf in available})
        if not channels or not tfs or min(tfs) < 1:
            raise ValueError("Select at least one channel and positive source-sink separation")
        slots = [(ch, tf) for ch in sorted(channels) for tf in tfs]
        if any(slot not in available for slot in slots):
            raise ValueError("Compact file lacks one or more selected channel/time datasets")
        coverage = [compare_kind(compact, "avgmore", args.legacy_avgmore, slots, canonical_moms, metrics)]
        if args.legacy_avgsrc:
            coverage.append(compare_kind(compact, "avgsrc", args.legacy_avgsrc, slots, canonical_moms, metrics))
        completed = bool(compact.attrs["complete"]) if "complete" in compact.attrs else None
    results = []
    for (kind, channel, tf), metric in sorted(metrics.items()):
        metric["normalized_max"] = metric["absmax"]/max(metric["scale"], TINY)
        metric["relative_l2"] = metric["difference_l2"]/max(metric["legacy_l2"], metric["compact_l2"], TINY)
        metric["passed"] = metric["nonfinite_rows"] == 0 and metric["max_per_momentum_normalized_max"] <= args.rtol
        results.append(dict(kind=kind, channel=channel, tf=tf, **metric))
    return dict(compact=str(args.compact.resolve()), compact_marked_complete=completed,
                tolerance=args.rtol,
                metric_definition="scale=max(maxabs(legacy),maxabs(compact),float64.tiny); each momentum row must pass absmax/scale<=rtol; relative_l2 uses max of both array norms",
                coverage=coverage, results=results, passed=bool(results) and all(r["passed"] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--legacy-avgmore", "--avgmore", type=Path, nargs="+", required=True)
    parser.add_argument("--legacy-avgsrc", "--avgsrc", type=Path, nargs="+")
    parser.add_argument("--currents", help="comma-separated quark channels; with any channel filter, unlisted channels are excluded")
    parser.add_argument("--stouts", help="comma-separated gluon stout steps")
    parser.add_argument("--tfs", help="comma-separated list or inclusive low:high")
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--report", "--json-report", type=Path)
    args = parser.parse_args()
    report = validate(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    failed = [r for r in report["results"] if not r["passed"]]
    worst = max(r["max_per_momentum_normalized_max"] for r in report["results"])
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['results'])} channel/time comparisons, "
          f"{len(failed)} failed; worst momentum-normalized maximum={worst:.6g}")
    for row in failed[:20]:
        print(f"  {row['kind']} {row['channel']} tf={row['tf']}: "
              f"normalized_max={row['max_per_momentum_normalized_max']:.6g}, "
              f"nonfinite_rows={row['nonfinite_rows']}, momentum={row['worst_momentum']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
