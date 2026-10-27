"""run001: bare B64 gluon plateau matrix elements, stout level10.

Using the usual activated HPC virtual environment, in an allocated CPU job:

    python3 -u run001_fit_matrix_elements.py --workers 8 > _ignore/run001_plateau.out 2>&1

Baseline uses one correlated constant over tf=8..16, tau=4..tf-4. Forward q=0
uses tf=8..13. The earlier asymmetric-cut scan is omitted for this simple common
window. No source/sink exchange or insertion-time averaging is performed.
Ratios use the full energy exponent in paper Eq19, with dispersion energies
from the existing matched jackknife mass. Diagonal gluon operators are made
traceless before local CPT real/imaginary projection. Only bare quantities are
written. Exact operator/cubic dependencies are removed independently per sector;
proportional ground-state coefficient rows remain distinct measurements.

One temporary float64 .npy stores selected correlators. An interrupted cache or
partial output is never silently reused. A completed run removes only its own
new cache and atomically publishes the HDF5 output; failed rows remain flagged.
"""
from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import leastsq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
core = importlib.import_module("run001_post2avgmore")
LEGACY = HERE.parents[1]/"05_moments"
RUN_ID = "run001"
SCHEMA = "gff-plateau-v1"
A_FM = 0.07948
LENGTH = 64
DEFAULT_INPUT = core.OUTPUT_ROOT/"cB211.072.64"/core.RUN_LABEL/"data_avgmore"
DEFAULT_OUTPUT = core.OUTPUT_ROOT/"cB211.072.64/run001_gluon_plateau/matrix_elements.h5"
DEFAULT_C2 = core.DATA_ROOT.parent/"05_moments/cB211.072.64/data_merge/disc_2pt.h5"
DEFAULT_VEV = DEFAULT_C2.with_name("disc_jvev.h5")
DEFAULT_MASS = LEGACY/"pkl/analysis_c2pt/reg_ignore/ens2msq2pars_jk.pkl"
PROJS = ("P0", "Px", "Py", "Pz")
INSERTS = tuple(core.INSERTS)
DIAGONALS = (0, 4, 7, 9)
FIT_CONTEXT = None


def manifest(path, with_hash=False):
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  ctime_ns=stat.st_ctime_ns)
    if with_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_statistics(path):
    names = ("jackknife", "jackme", "jackmec", "jackmap", "jackknife_pseudo", "jackfit", "doFit_const")
    source = Path(path).read_text()
    functions = {node.name: node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name in names}
    if set(functions) != set(names):
        raise ValueError("Established resampling/fitting definitions are missing")
    module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    namespace = dict(np=np, warnings=warnings, cholesky=cholesky, leastsq=leastsq, flag_fast=False)
    exec(compile(module, str(path), "exec"), namespace)
    signature = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    return SimpleNamespace(**{name: namespace[name] for name in names}), signature


def load_kinematics(path, ref):
    """Load the existing standalone gamma/ME2FF block and Q2 helper only."""
    tree = ast.parse(Path(path).read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
                 and any(isinstance(child, ast.FunctionDef) and child.name == "ME2FF" for child in node.body))
    q2 = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mom2Q2")
    yu = SimpleNamespace(**ref, ens2NL={"b": LENGTH}, ens2aInv={"b": 197.3/A_FM})
    module = ast.Module(body=[block, q2], type_ignores=[])
    namespace = dict(np=np, yu=yu)
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(ME2FF=namespace["ME2FF"], mom2Q2=namespace["mom2Q2"])


IO_INSERTS = INSERTS
IO_PROJS = PROJS
IO_DIAGONAL = DIAGONALS
IO_LOOP_ROOT = core.DATA_ROOT/"cB211.072.64/data_post_hold"
_IO_PLAN = None
_IO_CACHE = None


def io_file_manifest(path):
    return manifest(path)


def io_canonical_orbit(momentum):
    return tuple(sorted(abs(int(x)) for x in momentum))


def make_time_slots(tfmin=8, tfmax=16, cut=4):
    return [(tf, tau) for tf in range(tfmin, tfmax+1) for tau in range(cut, tf-cut+1)]


def load_normalization(c2_path, vev_path, mass_path, cfgs, moms, stats, ref=None):
    """Legacy shell weighting; separate orbit C2 for the q0 vacuum product."""
    paths = tuple(map(Path, (c2_path, vev_path, mass_path)))
    manifests = [io_file_manifest(path) for path in paths]
    cfgs, moms = list(cfgs), np.asarray(moms, dtype=int)
    tfs = np.arange(8, 17, dtype=int)
    if moms.ndim != 2 or moms.shape[1] != 6:
        raise ValueError("Expected six-integer momentum rows")
    sink_vectors, source_vectors = moms[:, :3], moms[:, :3]+moms[:, 3:]
    required_shells = sorted(set(np.sum(sink_vectors**2, axis=1)) | set(np.sum(source_vectors**2, axis=1)))
    required_orbits = sorted(set(map(io_canonical_orbit, sink_vectors)))
    with h5py.File(paths[0], "r") as source:
        actual_cfgs = [value.decode() if isinstance(value, bytes) else str(value) for value in source["cfgs"][:]]
        if actual_cfgs != cfgs:
            raise ValueError("C2 configuration identities/order differ from run001")
        stored_moms = np.asarray(source["moms"][:], dtype=int)
        data = source["data/N_N"]
        if data.shape[0] != len(cfgs) or data.shape[1] <= int(tfs[-1]) or data.shape[2] != len(stored_moms):
            raise ValueError("C2 dimensions do not cover the requested cohort/times")
        raw_c2 = np.asarray(data[:, tfs, :], dtype=np.complex128)
    if not np.all(np.isfinite(raw_c2)):
        raise ValueError("Nonfinite C2 input")
    orbit_lookup = {}
    for index, momentum in enumerate(stored_moms):
        key = io_canonical_orbit(momentum)
        if key in orbit_lookup:
            raise ValueError(f"Duplicate C2 cubic orbit: {key}")
        orbit_lookup[key] = index
    missing = [momentum for momentum in required_orbits if momentum not in orbit_lookup]
    if missing:
        raise ValueError(f"Missing sink C2 orbit(s): {missing}")
    stored_shells = np.sum(stored_moms**2, axis=1)
    reference = ref if ref is not None else core.load_reference(core.DEFAULT_REFERENCE)
    multiplicities = np.asarray([len(reference["mom2moms"](momentum.tolist())) for momentum in stored_moms], dtype=float)
    weights = np.zeros((len(stored_moms), len(required_shells)))
    for column, shell in enumerate(required_shells):
        selected = np.flatnonzero(stored_shells == shell)
        if not len(selected):
            raise ValueError(f"Missing required C2 momentum shell: {shell}")
        weights[selected, column] = multiplicities[selected]/multiplicities[selected].sum()
    shell_raw = np.einsum("ctm,ms->cts", raw_c2.real, weights)
    shell_c2_jk = stats.jackknife(shell_raw, d=1)
    indices = [orbit_lookup[momentum] for momentum in required_orbits]
    orbit_c2_jk = stats.jackknife(raw_c2[:, :, indices], d=1)
    with paths[2].open("rb") as stream:
        energy_cache = pickle.load(stream)
    mass_jk = np.asarray(energy_cache["b"][0][:, 0], dtype=np.float64)
    if mass_jk.shape != (len(cfgs),) or not np.all(np.isfinite(mass_jk)) or np.any(mass_jk <= 0):
        raise ValueError("Invalid/misaligned zero-momentum mass jackknife samples")
    shell_values = np.asarray(required_shells, dtype=int)
    energies_jk = np.sqrt(mass_jk[:, None]**2+(2*np.pi/LENGTH)**2*shell_values[None])
    with h5py.File(paths[1], "r") as source:
        vev_raw = np.asarray(source["jg;stout10"][:], dtype=np.complex128)
    if vev_raw.shape != (len(cfgs), len(INSERTS)) or not np.all(np.isfinite(vev_raw)):
        raise ValueError("Invalid gluon VEV dimensions/values")
    spatial = vev_raw[:, [4, 7, 9]].mean(axis=1)
    difference = vev_raw[:, 0]-spatial
    projected = np.zeros_like(vev_raw)
    projected[:, 0] = .75*difference
    projected[:, [4, 7, 9]] = -.25*difference[:, None]
    vev_traceless_jk = stats.jackknife(projected, d=1)
    if manifests != [io_file_manifest(path) for path in paths]:
        raise RuntimeError("A normalization input changed while being read")
    return dict(tfs=tfs, shell_values=shell_values, shell_c2_jk=shell_c2_jk,
                mass_jk=mass_jk, energies_jk=energies_jk,
                orbit_moms=np.asarray(required_orbits, dtype=int), orbit_c2_jk=orbit_c2_jk,
                vev_raw=vev_raw, vev_traceless_jk=vev_traceless_jk, manifests=manifests)


def _initialize_io_worker(plan):
    global _IO_PLAN, _IO_CACHE
    _IO_PLAN = plan
    _IO_CACHE = np.load(plan["cache_path"], mmap_mode="r+")


def _read_cache_configuration(task):
    index, cfg = task
    plan, cache = _IO_PLAN, _IO_CACHE
    c3_path = Path(plan["input_dir"])/(cfg+".h5")
    loop_path = Path(plan["loop_root"])/cfg/"jg.h5"
    c3_manifest, loop_manifest, raw_vev = None, None, None
    try:
        c3_manifest, loop_manifest = io_file_manifest(c3_path), io_file_manifest(loop_path)
        with h5py.File(loop_path, "r") as loop:
            qzero = np.flatnonzero(np.all(loop["moms"][:] == 0, axis=1))
            if len(qzero) != 1:
                raise ValueError("Expected exactly one q=0 gluon-loop row")
            inserts = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in loop["inserts"][:])
            order = [inserts.index(insert) for insert in INSERTS]
            values = np.asarray(loop["data/jg;stout10"][:, int(qzero[0]), :], dtype=np.complex128)
            raw_vev = values[:, order].mean(axis=0)
        expected = plan["expected_vev_raw"]
        if expected is not None and not np.allclose(raw_vev, expected[index], rtol=1e-12, atol=1e-10):
            raise ValueError("Legacy VEV row does not match this configuration's raw loop")
        with h5py.File(c3_path, "r") as source:
            if not source.attrs.get("complete", False):
                raise ValueError("C3 output is incomplete")
            stored_moms = np.asarray(source["moms"][:], dtype=int)
            lookup = {tuple(momentum): i for i, momentum in enumerate(stored_moms)}
            indices = [lookup[tuple(momentum)] for momentum in plan["moms"]]
            if len(set(indices)) != len(indices) or not np.array_equal(stored_moms[indices], plan["moms"]):
                raise ValueError("Duplicated/mismatched planned C3 momentum rows")
            for key, labels in (("inserts", INSERTS), ("projs", PROJS)):
                actual = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in source[key][:])
                if actual != labels:
                    raise ValueError(f"Unexpected C3 {key} ordering")
            for tf, lo, hi, taus in plan["groups"]:
                data = source[f"data/jg;stout10_{tf}"]
                if data.shape != (tf+1, len(stored_moms), 4, 10):
                    raise ValueError(f"Unexpected C3 dimensions at tf={tf}")
                if np.array_equal(stored_moms, plan["moms"]):
                    raw = np.asarray(data[taus[0]:taus[-1]+1], dtype=np.complex128)
                else:
                    order = np.argsort(indices)
                    raw = np.asarray(data[taus[0]:taus[-1]+1, np.asarray(indices)[order]],
                                     dtype=np.complex128)[:, np.argsort(order)]
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"Nonfinite C3 input at tf={tf}")
                trace = raw[..., DIAGONALS].sum(axis=-1)/4
                for ii in DIAGONALS:
                    raw[..., ii] -= trace
                cache[index, lo:hi] = np.where(plan["real_channels"], raw.real, raw.imag)
        if c3_manifest != io_file_manifest(c3_path) or loop_manifest != io_file_manifest(loop_path):
            raise RuntimeError("An input file changed during C3 cache collection")
        return dict(cfg=cfg, manifest=c3_manifest, vev_manifest=loop_manifest, raw_vev=raw_vev, error=None)
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as error:
        return dict(cfg=cfg, manifest=c3_manifest, vev_manifest=loop_manifest, raw_vev=raw_vev,
                    error=f"{type(error).__name__}: {error}")


def cache_c3(cfgs, input_dir, cache_path, moms, slots, workers=1, expected_vev_raw=None, loop_root=IO_LOOP_ROOT):
    """Fill one parent-created .npy with disjoint configuration-plane writes."""
    cfgs, moms, slots = list(cfgs), np.asarray(moms, dtype=int), list(slots)
    if workers < 1:
        raise ValueError("workers must be positive")
    check = np.load(cache_path, mmap_mode="r+")
    if check.shape != (len(cfgs), len(slots), len(moms), 4, 10) or check.dtype != np.dtype("float64"):
        raise ValueError("Parent-created C3 cache has unexpected shape/dtype")
    del check
    groups = []
    for tf in dict.fromkeys(tf for tf, _ in slots):
        positions = [index for index, pair in enumerate(slots) if pair[0] == tf]
        taus = [slots[index][1] for index in positions]
        if positions != list(range(positions[0], positions[-1]+1)) or taus != list(range(taus[0], taus[-1]+1)):
            raise ValueError("Cache slots must be contiguous in tf and insertion time")
        groups.append((tf, positions[0], positions[-1]+1, taus))
    eta = np.asarray([(-1)**insert.count("t") for insert in INSERTS])
    real_channels = np.asarray((1, -1, -1, -1))[:, None]*eta[None] > 0
    plan = dict(cache_path=str(cache_path), input_dir=str(input_dir), loop_root=str(loop_root),
                moms=moms, groups=groups, real_channels=real_channels,
                expected_vev_raw=None if expected_vev_raw is None else np.asarray(expected_vev_raw))
    raw_vev = np.full((len(cfgs), 10), np.nan+1j*np.nan, dtype=np.complex128)
    manifests, vev_manifests, failures = [], [], []
    executor = None
    if workers == 1:
        _initialize_io_worker(plan)
        collected = map(_read_cache_configuration, enumerate(cfgs))
    else:
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_initialize_io_worker, initargs=(plan,))
        collected = executor.map(_read_cache_configuration, enumerate(cfgs))
    try:
        for index, row in enumerate(collected):
            manifests.append(dict(cfg=row["cfg"], **(row["manifest"] or {})))
            vev_manifests.append(dict(cfg=row["cfg"], **(row["vev_manifest"] or {})))
            if row["error"]:
                failures.append(dict(cfg=row["cfg"], error=row["error"]))
            if row["raw_vev"] is not None:
                raw_vev[index] = row["raw_vev"]
            if (index+1) % 50 == 0 or index+1 == len(cfgs):
                print(f"run001_cache_done: {index+1}/{len(cfgs)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
        global _IO_CACHE
        _IO_CACHE = None
        final = np.load(cache_path, mmap_mode="r+")
        final.flush()
        del final
    return dict(manifests=manifests, vev_manifests=vev_manifests, raw_vev=raw_vev, failures=failures)


def candidate_components():
    result, weights = [], []
    for sector, names in (("equal", ("xx", "yy", "zz")),
                          ("unequal", ("tx", "ty", "tz", "xy", "xz", "yz"))):
        for pi, proj in enumerate(PROJS):
            for insert in names:
                ii = INSERTS.index(insert)
                weight = np.zeros((4, 10))
                weight[pi, ii] = 1
                if ii in DIAGONALS:
                    weight[pi, list(DIAGONALS)] -= 1/4
                eta = (-1)**insert.count("t")
                phase = "1" if eta*(1 if pi == 0 else -1) == 1 else "i"
                result.append(dict(proj=proj, insert=insert, sector=sector, phase=phase, pi=pi, ii=ii))
                weights.append(weight.ravel())
    return result, np.asarray(weights)


def select_rows(ref, moms):
    """Select independent signed little-group functionals, never ME2FF directions."""
    candidates, weights = candidate_components()
    rows, basis, ranges = [], [], []
    decisions = np.full((len(moms), len(candidates)), 2, dtype=np.int8)
    for mi, mom in enumerate(moms):
        block = core.orbit_block(ref, [mom])
        effective = np.zeros_like(weights)
        little = [ei for ei in range(48)
                  if np.array_equal(block.raw[block.rotations[0, ei]], mom)]
        for ei in little:
            indices = (block.projs[ei][:, None]*10+block.inserts[ei][None, :]).ravel()
            effective[:, indices] += weights*block.signs[ei].ravel()[None, :]
        effective /= len(little)
        start = len(rows)
        orthogonal = {"equal": [], "unequal": []}
        for ci, component in enumerate(candidates):
            vector = effective[ci]
            if np.linalg.norm(vector) < 1e-12:
                decisions[mi, ci] = 1  # Exactly vanishing full correlator after symmetry projection.
                continue
            residual = vector.copy()
            for previous in orthogonal[component["sector"]]:
                residual -= np.dot(residual, previous)*previous
            norm = np.linalg.norm(residual)
            if norm < 1e-12:
                continue
            orthogonal[component["sector"]].append(residual/norm)
            rows.append(component | dict(momentum_index=mi, candidate_index=ci))
            basis.append(vector.reshape(4, 10))
            decisions[mi, ci] = 0
        ranges.append((start, len(rows)))
    return rows, np.asarray(basis), np.asarray(ranges), decisions, candidates


def normalization_maps(norm, moms, slots):
    shells = {int(value): i for i, value in enumerate(norm["shell_values"])}
    orbits = {tuple(map(int, value)): i for i, value in enumerate(norm["orbit_moms"])}
    tf_lookup = {int(tf): i for i, tf in enumerate(norm["tfs"])}
    mappings, valid_tf, reasons = [], [], []
    for mom in moms:
        pf, initial = np.asarray(mom[:3]), np.asarray(mom[:3])+mom[3:]
        sf, si = shells[int(pf@pf)], shells[int(initial@initial)]
        good = np.all(np.isfinite(norm["shell_c2_jk"][:, :, [sf, si]])
                      & (norm["shell_c2_jk"][:, :, [sf, si]] > 0), axis=(0, 2))
        forward = not np.any(mom[3:])
        points = [i for i, (tf, _) in enumerate(slots) if not forward or tf <= 13]
        required_tfs = sorted({tf_lookup[slots[i][0]] for i in points})
        bad = [int(norm["tfs"][i]) for i in required_tfs if not good[i]]
        reason = "" if not bad else f"nonpositive/nonfinite C2 in a jackknife sample at tf={bad}"
        mappings.append(dict(sink_shell=sf, source_shell=si, orbit=orbits[tuple(sorted(map(abs, pf)))],
                             slot_indices=points, tf_indices=[tf_lookup[slots[i][0]] for i in points],
                             taus=[slots[i][1] for i in points], forward=forward, valid=not bad))
        valid_tf.append(good)
        reasons.append(reason)
    return mappings, np.asarray(valid_tf), reasons


def initialize_fit_worker(cache, slots, moms, rows, ranges, norm, mappings, util_path, kinematics_path, reference):
    global FIT_CONTEXT
    stats, _ = load_statistics(util_path)
    ref = core.load_reference(reference)
    FIT_CONTEXT = dict(cache=np.load(cache, mmap_mode="r"), slots=slots, moms=moms, rows=rows,
                       ranges=ranges, norm=norm, mappings=mappings, stats=stats,
                       kinematics=load_kinematics(kinematics_path, ref))


def fit_momentum(mi):
    context = FIT_CONTEXT
    norm, mapping = context["norm"], context["mappings"][mi]
    start, stop = map(int, context["ranges"][mi])
    rows = context["rows"][start:stop]
    count, width = len(norm["mass_jk"]), stop-start
    plateau = np.full((count, width), np.nan)
    coefficients = np.empty((width, 3))
    null = np.zeros(width, bool)
    okay = np.zeros(width, bool)
    chi2 = np.full(width, np.nan)
    dof = np.full(width, len(mapping["slot_indices"])-1, dtype=np.int32)
    errors, warning_count = [""]*width, np.zeros(width, dtype=np.int32)
    mom = context["moms"][mi]
    mass = float(np.mean(norm["mass_jk"]))
    pf = np.asarray(mom[:3])*(2*np.pi/LENGTH)
    initial = (np.asarray(mom[:3])+mom[3:])*(2*np.pi/LENGTH)
    q2 = float(context["kinematics"].mom2Q2(mom, "b", mN=mass))
    for ri, row in enumerate(rows):
        phase = 1 if row["phase"] == "1" else 1j
        raw = context["kinematics"].ME2FF(mass, initial, pf, row["proj"], row["insert"])
        coefficients[ri] = np.real(np.conj(phase)*np.asarray(raw, dtype=complex))
        null[ri] = np.linalg.norm(coefficients[ri]) < 1e-10
    if not mapping["valid"]:
        errors = ["invalid required C2 normalization"]*width
    else:
        points = np.asarray(mapping["slot_indices"])
        times = np.asarray(mapping["tf_indices"])
        taus = np.asarray(mapping["taus"])
        tfs = np.asarray([context["slots"][i][0] for i in points])
        sf, si = mapping["sink_shell"], mapping["source_shell"]
        denominator = np.sqrt(norm["shell_c2_jk"][:, times, sf]*norm["shell_c2_jk"][:, times, si])
        exponent = (norm["energies_jk"][:, sf]-norm["energies_jk"][:, si])[:, None]*(taus-tfs/2)[None, :]
        denominator *= np.exp(exponent)  # Full Eq19 exponent; no extra square root.
        for ri, row in enumerate(rows):
            if mapping["forward"] and row["pi"] != 0:
                errors[ri] = "q0 polarized kinematic-null control excluded; no spin-C2 vacuum subtraction"
                continue
            data = np.asarray(context["cache"][:, points, mi, row["pi"], row["ii"]])
            y_jk = context["stats"].jackknife(data, d=1)
            if mapping["forward"] and row["pi"] == 0 and row["ii"] in DIAGONALS:
                # Vacuum expectation is cubic-invariant before trace subtraction.
                # Its product uses matched ensemble means, not per-cfg products.
                vacuum = norm["orbit_c2_jk"][:, times, mapping["orbit"]]*norm["vev_traceless_jk"][:, row["ii"]][:, None]
                y_jk -= np.real(vacuum)
            y_jk /= denominator
            try:
                if not np.all(np.isfinite(y_jk)):
                    raise ValueError("nonfinite normalized ratio")
                with warnings.catch_warnings(record=True) as caught:
                    pars, chis, ndof = context["stats"].doFit_const(y_jk, corrQ=True)
                warning_count[ri] = len(caught)
                chi2[ri] = float(np.mean(chis))
                dof[ri] = int(ndof)
                if not np.all(np.isfinite(pars)) or not np.all(np.isfinite(chis)):
                    raise ValueError("nonfinite fitter output")
                if caught:
                    raise ValueError("fitter warning: "+str(caught[0].message))
                plateau[:, ri] = pars[:, 0]
                okay[ri] = True
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                errors[ri] = f"{type(error).__name__}: {error}"
    return dict(momentum_index=mi, start=start, stop=stop, plateau=plateau, K=coefficients,
                Q2=q2, kinematic_null=null, fit_ok=okay, chi2=chi2, dof=dof,
                errors=errors, warning_count=warning_count)


def strings(values):
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def write_setup(output, args, cfgs, moms, rows, basis, ranges, decisions, candidates, slots, norm, valid_tf, reasons, provenance):
    output.attrs.update(schema=SCHEMA, procedure_id=RUN_ID, complete=False, a_fm=A_FM,
                        mass_lattice=float(np.mean(norm["mass_jk"])), normalized_ratio="paperEq19full",
                        source_sink_exchange=False, time_average=False, renormalized=False)
    output.create_dataset("cfgs", data=strings(cfgs))
    output.create_dataset("moms", data=np.asarray([moms[row["momentum_index"]] for row in rows], dtype=np.int32))
    for field in ("proj", "insert", "sector", "phase"):
        output.create_dataset(field, data=strings([row[field] for row in rows]))
    output.create_dataset("input_moms", data=moms)
    output.create_dataset("row_momentum_index", data=[row["momentum_index"] for row in rows])
    output.create_dataset("row_candidate_index", data=[row["candidate_index"] for row in rows])
    output.create_dataset("basis_W", data=basis)
    output.create_dataset("kinematic_null", (len(rows),), dtype=bool)
    output.create_dataset("K", (len(rows), 3), dtype="float64")
    output.create_dataset("Q2", (len(rows),), dtype="float64")
    output.create_dataset("status/momentum_row_ranges", data=ranges)
    output.create_dataset("status/component_decision", data=decisions)
    output.create_dataset("status/normalization_valid_tf", data=valid_tf)
    output.create_dataset("status/normalization_error", data=strings(reasons))
    output.create_dataset("status/normalization_valid", data=np.asarray([not reason for reason in reasons]))
    output.create_dataset("status/fitter_warning_count", (len(rows),), dtype="int32")
    output.create_dataset("status/fit_error", (len(rows),), dtype=h5py.string_dtype("utf-8"))
    output.create_dataset("status/completed_momenta", data=np.zeros(len(moms), dtype=bool))
    group = output.create_group("selections/baseline")
    group.attrs.update(tf_min=8, tf_max=16, q0_tf_max=13, cut_source=4, cut_sink=4,
                       fit_type="constant over all selected (tf,tau)", correlated=True,
                       asymmetric_cut_scan="omitted for the simple common window")
    group.create_dataset("plateau_jk", (len(cfgs), len(rows)), dtype="float64", fillvalue=np.nan,
                         chunks=(len(cfgs), min(32, len(rows))), compression="gzip", compression_opts=1,
                         shuffle=True, fletcher32=True)
    for name, dtype, fill in (("fit_ok", bool, False), ("chi2", "float64", np.nan), ("dof", "int32", -1)):
        group.create_dataset(name, (len(rows),), dtype=dtype, fillvalue=fill)
    group.create_dataset("time_slots", data=np.asarray(slots))
    for key in ("tfs", "shell_values", "shell_c2_jk", "mass_jk", "energies_jk", "orbit_moms",
                "orbit_c2_jk", "vev_traceless_jk"):
        output.create_dataset("normalization/"+key, data=norm[key])
    output.create_dataset("metadata/provenance", data=json.dumps(provenance, sort_keys=True))
    output.create_dataset("metadata/candidate_components", data=json.dumps(candidates))
    output.create_dataset("metadata/row_selection", data="0 selected; 1 exact spatial/operator zero; 2 exact linear dependency. Greedy Gram-Schmidt acts on signed little-group projected W, independently by sector; ME2FF is never used for selection.")
    output.create_dataset("metadata/notes", data="Raw gluon diagonal trace removed. Local CPT amplitude is Re(conj(phase)*C3). No exchange/time averaging. q0 vacuum term is a product of matched jackknife means using orbit C2 and cubic/trace-projected loop VEV. Denominator C2 is the legacy multiplicity-weighted shell average. K is evaluated at the central matched mass; ratio energy corrections use matched mass samples. Bare quantities only.")


def run(args):
    started = time.monotonic()
    cohort = core.load_run_cohort(args.cfg_list)
    cfgs = cohort["cfgs"]
    with h5py.File(args.input_dir/(cfgs[0]+".h5"), "r") as source:
        moms = np.asarray(source["moms"][:], dtype=np.int32)
    if len(moms) != 1685:
        raise ValueError("Expected all1685 run001 canonical momentum representatives")
    if args.limit_momenta:
        if args.output.resolve() == DEFAULT_OUTPUT.resolve():
            raise ValueError("A limited pilot requires a distinct --output")
        moms = moms[:args.limit_momenta]
    partial = args.output.with_name(args.output.stem+".partial.h5")
    cache = args.output.with_name(args.output.stem+".run001_cache.npy")
    if any(path.exists() for path in (args.output, partial, cache)):
        raise FileExistsError("Output, partial output, or cache already exists; review it or choose a new --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats, statistics_signature = load_statistics(args.util)
    ref = core.load_reference(args.reference_script)
    slots = make_time_slots()
    norm = load_normalization(args.c2, args.vev, args.mass_cache, cfgs, moms, stats, ref)
    rows, basis, ranges, decisions, candidates = select_rows(ref, moms)
    mappings, valid_tf, reasons = normalization_maps(norm, moms, slots)
    provenance = dict(cohort=cohort, cfg_list=manifest(args.cfg_list, True),
                      producer=manifest(__file__, True), run001=manifest(core.__file__, True),
                      reference=manifest(args.reference_script, True), statistics=manifest(args.util, True),
                      kinematics=manifest(args.kinematics, True), statistics_ast_sha256=statistics_signature,
                      normalization_inputs=norm["manifests"], numpy=np.__version__, h5py=h5py.__version__,
                      workers=args.workers, pilot=bool(args.limit_momenta), cache_shape=[len(cfgs), len(slots), len(moms), 4, 10],
                      baseline=dict(tf_min=8, tf_max=16, q0_tf_max=13, source_cut=4, sink_cut=4,
                                    source_sink_exchange=False, time_average=False),
                      created_at_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(dict(phase="inputs", momenta=len(moms), rows=len(rows), invalid_normalizations=sum(bool(x) for x in reasons))), flush=True)
    # Exclusive creation reserves this output/cache pair; no preexisting artifact is removed.
    with h5py.File(partial, "x") as output:
        write_setup(output, args, cfgs, moms, rows, basis, ranges, decisions, candidates, slots,
                    norm, valid_tf, reasons, provenance)
    mapped = np.lib.format.open_memmap(cache, mode="w+", dtype="float64",
                                     shape=(len(cfgs), len(slots), len(moms), 4, 10))
    mapped.flush()
    del mapped
    collected = cache_c3(cfgs, args.input_dir, cache, moms, slots, args.workers,
                         expected_vev_raw=norm["vev_raw"], loop_root=args.loop_root)
    if collected["failures"]:
        with h5py.File(partial, "r+") as output:
            output.create_dataset("metadata/collection_failures", data=json.dumps(collected["failures"]))
        raise RuntimeError(f"Selected C3 collection failed for {len(collected['failures'])} configurations; partial/cache preserved")
    print(json.dumps(dict(phase="cache_ready", configurations=len(cfgs), cache_bytes=cache.stat().st_size)), flush=True)
    initargs = (cache, slots, moms, rows, ranges, norm, mappings, args.util, args.kinematics, args.reference_script)
    executor = None
    if args.workers == 1:
        initialize_fit_worker(*initargs)
        fitted = map(fit_momentum, range(len(moms)))
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_fit_worker, initargs=initargs)
        fitted = executor.map(fit_momentum, range(len(moms)), chunksize=1)
    successful = 0
    try:
        with h5py.File(partial, "r+") as output:
            output.create_dataset("metadata/c3_inputs", data=json.dumps(collected["manifests"]))
            output.create_dataset("metadata/vev_verified_inputs", data=json.dumps(collected["vev_manifests"]))
            group = output["selections/baseline"]
            for number, result in enumerate(fitted, start=1):
                a, b = result["start"], result["stop"]
                group["plateau_jk"][:, a:b] = result["plateau"]
                for field in ("fit_ok", "chi2", "dof"):
                    group[field][a:b] = result[field]
                output["K"][a:b] = result["K"]
                output["Q2"][a:b] = result["Q2"]
                output["kinematic_null"][a:b] = result["kinematic_null"]
                output["status/fit_error"][a:b] = result["errors"]
                output["status/fitter_warning_count"][a:b] = result["warning_count"]
                output["status/completed_momenta"][result["momentum_index"]] = True
                successful += int(result["fit_ok"].sum())
                if number % args.progress_every == 0 or number == len(moms):
                    output.flush()
                    print(json.dumps(dict(phase="plateau_fits", momenta_done=number, momenta=len(moms), fit_ok=successful,
                                          elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
            output.attrs.update(complete=True, successful_rows=successful, elapsed_seconds=time.monotonic()-started)
            output.flush()
    finally:
        if executor is not None:
            executor.shutdown()
        global FIT_CONTEXT
        FIT_CONTEXT = None
    os.rename(partial, args.output)
    cache.unlink()
    print(json.dumps(dict(phase="complete", file=str(args.output), momenta=len(moms), rows=len(rows),
                          fit_ok=successful, elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
    return 0 if successful else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cfg-list", type=Path, default=core.DEFAULT_CFG_LIST)
    parser.add_argument("--c2", type=Path, default=DEFAULT_C2)
    parser.add_argument("--vev", type=Path, default=DEFAULT_VEV)
    parser.add_argument("--mass-cache", type=Path, default=DEFAULT_MASS)
    parser.add_argument("--loop-root", type=Path, default=core.DATA_ROOT/"cB211.072.64/data_post_hold")
    parser.add_argument("--reference-script", type=Path, default=core.DEFAULT_REFERENCE)
    parser.add_argument("--util", type=Path, default=LEGACY/"util.py")
    parser.add_argument("--kinematics", type=Path, default=LEGACY/"util_moments.py")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-momenta", type=int, help="small pilot only; requires distinct --output")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.workers, args.progress_every) < 1 or (args.limit_momenta is not None and args.limit_momenta < 1):
        parser.error("Workers, progress interval and optional momentum limit must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
