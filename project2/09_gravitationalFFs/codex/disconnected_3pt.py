"""Bounded-memory 1DV disconnected C3 production, one HDF5 file per cfg.

Scientific geometry/projectors are loaded from the established 05_moments
script without executing its CLI. See disconnected_3pt.md for the data layout.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time

try:
    import resource
except ImportError:  # Lightweight checks also run in the local Windows checkout.
    resource = None

import h5py
import numpy as np
from sympy.combinatorics import Permutation
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE.parents[1] / "05_moments/dataPrepare/cB211.072.64/post2avgmore.py"
DATA_ROOT = Path("/p/project1/ngff/li47/code/projectData/02_discNJN_1D")
OUTPUT_ROOT = Path("/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs")
INSERTS = ("tt", "tx", "ty", "tz", "xx", "xy", "xz", "yy", "yz", "zz")
SCHEMA = "gff-disconnected-1dv-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(path, ensemble="cB211.072.64", tfs=(2, 22), currents=(), stouts=()):
    """Reuse pure legacy helpers, excluding input choices and the decorated CLI."""
    source = Path(path).read_text()
    prefix = ast.parse(source.split("# -------------------------", 1)[0])
    nodes = [n for n in prefix.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {"extract2pt", "extractLoop", "src2ints", "get_phase", "correlate", "avgmore"}
    nodes += [n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(np=np, h5py=h5py, os=os, re=re, Permutation=Permutation)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    ns.update(case="1DV", inserts=list(INSERTS), flags={"g5H": True},
              ens=ensemble, lat_L={"cB211.072.64": 64, "cC211.060.80": 80,
                                 "cD211.054.96": 96, "cE211.044.112": 112}[ensemble],
              tfs=list(tfs), jqs=list(currents), stouts=list(stouts))
    return ns


def vectors(cap):
    extent = math.isqrt(cap)
    return [v for v in itertools.product(range(-extent, extent+1), repeat=3)
            if sum(x*x for x in v) <= cap]


def momentum_table(ref, sink_cap, transfer_cap, initial_cap=None):
    seen, representatives = set(), []
    for p in vectors(sink_cap):
        for q in vectors(transfer_cap):
            mom = p+q
            if mom in seen:
                continue
            if initial_cap is not None and sum((x+y)**2 for x, y in zip(p, q)) > initial_cap:
                continue
            orbit = ref["mom2moms"](list(mom))
            seen.update(map(tuple, orbit))
            representatives.append(min(orbit, key=ref["getSortkey_mom3pt"]))
    return np.asarray(sorted(representatives, key=ref["getSortkey_mom3pt"]), dtype=np.int32)


@dataclass
class OrbitBlock:
    raw: np.ndarray
    rotations: np.ndarray
    projs: np.ndarray
    inserts: np.ndarray
    signs: np.ndarray


def orbit_block(ref, moms):
    raw = [v for mom in moms for v in ref["mom2moms"](list(mom))]
    lookup = {tuple(v): i for i, v in enumerate(raw)}
    if len(lookup) != len(raw):
        raise ValueError("Momentum representatives have overlapping cubic orbits")
    rotations, projs, inserts, signs = [], [], [], []
    for e in ref["elements_rot48"]:
        sx, sy, sz, xyz = e
        determinant = sx*sy*sz*(-1)**sum(xyz[i] > xyz[j] for i in range(3) for j in range(i+1, 3))
        pi = np.asarray([0, *(i+1 for i in xyz)])
        ps = np.asarray([1, sx*determinant, sy*determinant, sz*determinant])
        xyzt = "xyzt"
        mapped = dict(zip(xyzt, [xyzt[i] for i in xyz]+["t"]))
        spatial_sign = dict(zip(xyzt, [sx, sy, sz, 1]))
        ii, iss = [], []
        for a, b in INSERTS:
            transformed = mapped[a]+mapped[b]
            ii.append(INSERTS.index(transformed if transformed in INSERTS else transformed[::-1]))
            iss.append(spatial_sign[a]*spatial_sign[b])
        ii = np.asarray(ii)
        # Legacy applies signs BEFORE the component permutations.
        signs.append(ps[pi, None]*np.asarray(iss)[None, ii])
        projs.append(pi)
        inserts.append(ii)
        rotations.append([lookup[tuple(ref["rotate_mom"](e, list(m)))] for m in moms])
    return OrbitBlock(np.asarray(raw, dtype=np.int32), np.asarray(rotations).T,
                      np.asarray(projs), np.asarray(inserts), np.asarray(signs))


def cubic_average(raw, block):
    """raw(time,raw_momentum,proj,insert) -> (time,representative,proj,insert)."""
    values = raw[:, block.rotations[:, :, None, None],
                 block.projs[None, :, :, None], block.inserts[None, :, None, :]]
    return np.mean(values*block.signs[None, None], axis=2)


def source_coordinates(label):
    match = re.fullmatch(r"sx(\d+)sy(\d+)sz(\d+)st(\d+)", label)
    if match is None:
        raise ValueError(f"Unrecognized source label {label}")
    return tuple(map(int, match.groups()))


def file_manifest(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def input_inventory(c2_dir, loop_dir, pvec, qvec, tfs, channels, stats=None):
    c2_files = sorted(p for p in c2_dir.iterdir() if p.is_file() and p.name.startswith("N.h5"))
    if not c2_files:
        raise ValueError(f"No N.h5* source-resolved inputs in {c2_dir}")
    records, manifests = [], []
    largest_c2_read, largest_loop_read = 0, 0
    for path in c2_files:
        with h5py.File(path, "r") as f:
            present = set(map(tuple, f["moms"][:].tolist()))
            if not set(map(tuple, pvec)).issubset(present):
                raise ValueError(f"Required sink momentum is absent from {path}")
            if set(f["data"]) != set(f["data_bw"]):
                raise ValueError(f"Forward/backward source lists differ in {path}")
            labels = list(f["data"])
            for label in labels:
                for group in ("data", "data_bw"):
                    for nucleon in ("N1_N1", "N2_N2"):
                        dataset = f[f"{group}/{label}/{nucleon}"]
                        shape = dataset.shape
                        largest_c2_read = max(largest_c2_read, len(tfs)*math.prod(shape[1:])*dataset.dtype.itemsize)
                        if shape[0] <= max(tfs) or shape[-1] != 4:
                            raise ValueError(f"Insufficient time/spin coverage in {path}/{group}/{label}")
                records.append((str(path), label, source_coordinates(label)))
        manifests.append(file_manifest(path))
    temporal = None
    for filename, selected in (("j.h5", [j for j in channels if not j.startswith("jg;")]),
                               ("jg.h5", [j for j in channels if j.startswith("jg;")])):
        if not selected:
            continue
        path = loop_dir/filename
        with h5py.File(path, "r") as f:
            if not set(map(tuple, qvec)).issubset(set(map(tuple, f["moms"][:].tolist()))):
                raise ValueError(f"Required insertion momentum is absent from {path}")
            required = selected if filename == "jg.h5" else [j+";d"+str(mu) for j in selected for mu in range(4)]
            for j in required:
                if j not in f["data"]:
                    raise ValueError(f"Missing {path}/data/{j}")
                t = f["data"][j].shape[0]
                dataset = f["data"][j]
                largest_loop_read = max(largest_loop_read, math.prod(dataset.shape)*dataset.dtype.itemsize)
                temporal = t if temporal is None else temporal
                if temporal != t:
                    raise ValueError("Loop temporal extents differ")
        manifests.append(file_manifest(path))
    if not records or temporal is None or max(tfs) >= temporal:
        raise ValueError("Empty input selection or invalid temporal extent")
    if stats is not None:
        stats.update(input_read_reserve_bytes=6*largest_c2_read+8*largest_loop_read)
    return c2_files, records, manifests, temporal


def read_c2(paths, records, pvec, tfs, ref):
    """Read each source once; retain only the requested times and momenta."""
    shape = (len(records), len(tfs), len(pvec), 4)
    forward, backward = np.empty(shape, complex), np.empty(shape, complex)
    offset = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            lookup = {tuple(m): i for i, m in enumerate(f["moms"][:])}
            indices = np.asarray([lookup[tuple(m)] for m in pvec])
            for label in f["data"]:
                for group, destination, projector in (
                        ("data", forward, ref["dirac2proj"]),
                        ("data_bw", backward, ref["dirac2proj_bw"])):
                    first = f[f"{group}/{label}/N1_N1"]
                    times = np.asarray(tfs) if group == "data" else first.shape[0]-np.asarray(tfs)
                    order = np.argsort(times)
                    inverse = np.argsort(order)
                    a = first[times[order]][:, indices]
                    b = f[f"{group}/{label}/N2_N2"][times[order]][:, indices]
                    # Preserve the reference complex64 N1/N2 addition, before projection.
                    averaged = (a+b)/2
                    selected = averaged[inverse]
                    destination[offset] = np.einsum("pd,fmd->fmp", projector, selected)
                offset += 1
    return forward, backward


def read_loops(loop_dir, qvec, channels):
    loops = {}
    for filename in ("j.h5", "jg.h5"):
        selected = [j for j in channels if (j.startswith("jg;") == (filename == "jg.h5"))]
        if not selected:
            continue
        with h5py.File(loop_dir/filename, "r") as f:
            lookup = {tuple(m): i for i, m in enumerate(f["moms"][:])}
            qi = [lookup[tuple(q)] for q in qvec]
            inserts = [x.decode() for x in f["inserts"][:]]
            for j in selected:
                if filename == "jg.h5":
                    a = f["data"][j][:][:, qi]
                    loops[j] = np.asarray(a[:, :, [inserts.index(x) for x in INSERTS]], dtype=np.complex128)
                    continue
                gamma = [inserts.index(x) for x in ("gt", "gx", "gy", "gz")]
                # Axes: gamma, derivative, absolute_time, insertion_momentum.
                a = np.stack([f["data"][j+";d"+str(mu)][:][:, qi][:, :, gamma]
                              for mu in (3, 0, 1, 2)]).transpose(3, 0, 1, 2)
                a = (a+a.transpose(1, 0, 2, 3))/2
                a = a-np.eye(4)[:, :, None, None]*np.trace(a, axis1=0, axis2=1)[None, None]/4
                txyz = "txyz"
                a = np.stack([a[txyz.index(x), txyz.index(y)] for x, y in INSERTS], axis=-1)
                negative = [list(map(tuple, qvec)).index(tuple(-q)) for q in qvec]
                loops[j] = (a+np.conj(a[:, negative]))/2
    return loops


def time_weights(c2f, c2b, coordinates, block, pvec, temporal, length, source_batch=16):
    lookup = {tuple(p): i for i, p in enumerate(pvec)}
    pf = np.asarray([lookup[tuple(m[:3])] for m in block.raw])
    pb = np.asarray([lookup[tuple(-m[:3])] for m in block.raw])
    shape = (temporal, c2f.shape[1], len(block.raw), 4)
    forward, backward = np.zeros(shape, complex), np.zeros(shape, complex)
    for start in range(0, len(coordinates), source_batch):
        stop = min(start+source_batch, len(coordinates))
        coords = coordinates[start:stop]
        phase = np.exp(1j*(2*np.pi/length)*(coords[:, :3]@block.raw[:, 3:].T))
        np.add.at(forward, coords[:, 3] % temporal,
                  c2f[start:stop][:, :, pf]*phase[:, None, :, None])
        np.add.at(backward, coords[:, 3] % temporal,
                  c2b[start:stop][:, :, pb]*np.conj(phase)[:, None, :, None])
    return forward, backward


def contract(weights_f, weights_b, loop, raw_moms, qvec, tf_index, tf, source_count,
             kernel="direct"):
    """Contract one current/separation, including PT but before spatial average."""
    temporal = len(loop)
    lookup = {tuple(q): i for i, q in enumerate(qvec)}
    groups = {}
    for i, m in enumerate(raw_moms):
        groups.setdefault(tuple(m[3:]), []).append(i)
    result = np.empty((tf+1, len(raw_moms), 4, 10), complex)
    times = np.arange(temporal)[:, None]
    tau = np.arange(tf+1)[None, :]
    for q, indices in groups.items():
        qi, qn = lookup[q], lookup[tuple(-np.asarray(q))]
        wf = weights_f[:, tf_index][:, indices]
        wb = weights_b[:, tf_index][:, indices]
        if kernel == "direct":
            lf = loop[(times+tau) % temporal, qi].reshape(temporal, -1)
            lb = loop[(times-tau) % temporal, qn].reshape(temporal, -1)
            fw = (wf.transpose(1, 2, 0).reshape(-1, temporal)@lf).reshape(len(indices), 4, tf+1, 10).transpose(2, 0, 1, 3)
            bw = (wb.transpose(1, 2, 0).reshape(-1, temporal)@lb).reshape(len(indices), 4, tf+1, 10).transpose(2, 0, 1, 3)
        elif kernel == "fft":
            fw = np.fft.ifft(wf[:, :, :, None]*loop[:, qi, None, None, :], axis=0)[:tf+1]
            lb = loop[(-np.arange(temporal)) % temporal, qn]
            bw = np.fft.ifft(wb[:, :, :, None]*lb[:, None, None, :], axis=0)[:tf+1]
        else:
            raise ValueError(kernel)
        result[:, indices] = (fw+bw*np.asarray([-1, 1, 1, 1])[None, None, :, None])/(2*source_count)
    return result


def selections(text):
    return [int(x) for x in text.split(",") if x]


def channel_name(channel):
    return channel if channel.startswith("jg;") else channel+";disc"


def parse_tfs(text):
    if ":" in text:
        low, high = map(int, text.split(":"))
        return list(range(low, high+1))
    return selections(text)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def peak_mib():
    if resource is None:
        import psutil
        return psutil.Process().memory_info().peak_wset/1024**2
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**2 if sys.platform == "darwin" else 1024)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--cfg", required=True)
    p.add_argument("--ensemble", default="cB211.072.64")
    p.add_argument("--sink-n2-max", type=int, default=16)
    p.add_argument("--transfer-n2-max", type=int, default=16)
    p.add_argument("--initial-n2-max", type=int)
    p.add_argument("--stouts", default="5,7,10,13,15,20,25,30,35,40")
    p.add_argument("--currents", default="j+,js,jc", help="comma-separated; empty selects gluon only")
    p.add_argument("--tfs", default="2:22", help="inclusive low:high or comma-separated list")
    p.add_argument("--c2-root", type=Path)
    p.add_argument("--loop-root", type=Path)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--run-label", default="K16_stout10")
    p.add_argument("--reference-script", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--momentum-block", type=int, default=4)
    p.add_argument("--memory-mib", type=int, default=1024, help="conservative working-array budget, checked before reading payloads")
    p.add_argument("--threads", type=int, default=1, help="BLAS threads per configuration process")
    p.add_argument("--kernel", choices=("direct", "fft"), default="direct")
    p.add_argument("--compression", choices=("none", "lzf", "gzip"), default="gzip")
    p.add_argument("--save-avgsrc", action="store_true")
    p.add_argument("--momenta-json", type=Path, help="optional selected canonical momenta for validation")
    p.add_argument("--limit-momenta", type=int, help="small pilot only; use a distinct run-label")
    return p


def validate_names(args):
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", args.cfg)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_label)
            or args.run_label in (".", "..")):
        raise ValueError("cfg/run-label must be plain names, not paths")


@contextmanager
def configuration_lock(path):
    """Keep this persistent inode: removing a flock file creates a race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def produce(args):
    validate_names(args)
    lock = args.output_root/args.ensemble/args.run_label/"locks"/(args.cfg+".lock")
    with configuration_lock(lock):
        _produce(args)


def _produce(args):
    started = time.monotonic()
    validate_names(args)
    if min(args.sink_n2_max, args.transfer_n2_max) < 0 or min(args.momentum_block, args.threads) < 1:
        raise ValueError("Invalid momentum cut, block size or thread count")
    tfs, stouts = parse_tfs(args.tfs), selections(args.stouts)
    currents = [j for j in args.currents.split(",") if j]
    if not tfs or min(tfs) < 1 or len(set(tfs)) != len(tfs) or any(j not in ("j+", "js", "jc") for j in currents):
        raise ValueError("Select distinct positive tfs and currents from j+,js,jc")
    channels = currents+[f"jg;stout{s}" for s in stouts]
    if not channels or len(set(channels)) != len(channels) or any(s < 0 for s in stouts):
        raise ValueError("Empty/duplicate channel selection or negative stout step")
    ref = load_reference(args.reference_script, args.ensemble, tfs, currents, stouts)
    moms = momentum_table(ref, args.sink_n2_max, args.transfer_n2_max, args.initial_n2_max)
    if args.momenta_json:
        chosen = np.asarray(json.loads(args.momenta_json.read_text()), dtype=np.int32)
        chosen = np.asarray([ref["mom3pt2standard"](list(m)) for m in chosen], dtype=np.int32)
        allowed = set(map(tuple, moms))
        if any(tuple(m) not in allowed for m in chosen) or len(set(map(tuple, chosen))) != len(chosen):
            raise ValueError("Selected momenta are duplicated or outside the specified cuts")
        moms = chosen
    if args.limit_momenta is not None:
        if args.limit_momenta < 1:
            raise ValueError("limit-momenta must be positive")
        moms = moms[:args.limit_momenta]
    if not len(moms):
        raise ValueError("No momentum representatives selected")
    all_raw = {tuple(v) for m in moms for v in ref["mom2moms"](list(m))}
    pvec = np.asarray(sorted({m[:3] for m in all_raw}), dtype=np.int32)
    qvec = np.asarray(sorted({m[3:] for m in all_raw}), dtype=np.int32)
    c2_root = args.c2_root or DATA_ROOT/args.ensemble/"data_post_hold_2pt_moreMom"
    loop_root = args.loop_root or DATA_ROOT/args.ensemble/"data_post_hold"
    inventory_stats = {}
    files, records, manifests, temporal = input_inventory(c2_root/args.cfg, loop_root/args.cfg, pvec, qvec, tfs, channels, inventory_stats)
    # Conservative bound for caches, two time-bucket buffers, source tiles,
    # contraction/rotation temporaries and interpreter/HDF5 overhead.
    cache_bytes = 2*len(records)*len(tfs)*len(pvec)*4*16
    loop_bytes = temporal*len(qvec)*10*len(channels)*16
    pair_bytes = (2*temporal*len(tfs)*4*16 + 2*16*len(tfs)*4*16
                  + max(max(tfs)+1, temporal if args.kernel == "fft" else 0)*4*10*16*4)
    overhead = 192*1024**2+inventory_stats["input_read_reserve_bytes"]
    budget = args.memory_mib*1024**2
    multiplier = 2 if args.kernel == "fft" else 1
    block_size = min(args.momentum_block, max(0, int((budget-cache_bytes-multiplier*loop_bytes-overhead)/(48*pair_bytes*multiplier))))
    if block_size < 1:
        raise MemoryError("Requested working-memory budget is too small; reduce cuts/tfs or raise --memory-mib")
    block_count = math.ceil(len(moms)/block_size)
    settings = dict(ensemble=args.ensemble, cfg=args.cfg, sink_n2_max=args.sink_n2_max,
                    transfer_n2_max=args.transfer_n2_max, initial_n2_max=args.initial_n2_max,
                    tfs=tfs, stouts=stouts, currents=currents, block_size=block_size,
                    threads=args.threads, working_memory_mib=args.memory_mib,
                    kernel=args.kernel, compression=args.compression, save_avgsrc=args.save_avgsrc,
                    dtype="complex128", g5H=True, spatial_average="48 group elements; no source/sink exchange")
    identity = dict(schema=SCHEMA, settings=settings, implementation_sha256=digest(__file__),
                    reference_sha256=digest(args.reference_script), inputs=manifests,
                    source_records=[(path, label, xyz) for path, label, xyz in records],
                    moms=moms.tolist(), numpy=np.__version__, h5py=h5py.__version__)
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    run_hash = hashlib.sha256(identity_json.encode()).hexdigest()
    directory = args.output_root/args.ensemble/args.run_label/"data_avgmore"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory/(args.cfg+".h5")
    partial = directory/(args.cfg+".partial.h5")
    if final.exists():
        with h5py.File(final, "r") as f:
            if f.attrs.get("run_hash") != run_hash or not f.attrs.get("complete", False):
                raise ValueError(f"Existing output has a different run identity: {final}")
        log(f"cfg={args.cfg} already complete: {final}")
        return
    mode = "r+" if partial.exists() else "x"
    # HDF5's exclusive writer lock prevents simultaneous processes for one cfg.
    with h5py.File(partial, mode, rdcc_nbytes=2*1024**2) as output:
        if mode == "r+":
            if output.attrs.get("run_hash") != run_hash:
                raise ValueError(f"Partial output has different settings/inputs/code: {partial}")
            if not output.attrs.get("initialized", False):
                raise ValueError(f"Partial output initialization was interrupted; use a fresh run-label: {partial}")
        else:
            output.attrs.update(schema=SCHEMA, run_hash=run_hash, complete=False)
            output.create_dataset("metadata/run_identity", data=identity_json)
            output.create_dataset("metadata/reference_source", data=args.reference_script.read_text())
            output.create_dataset("metadata/producer_source", data=Path(__file__).read_text())
            output.create_dataset("metadata/command_line", data=json.dumps(sys.argv))
            output.create_dataset("moms", data=moms)
            output.create_dataset("inserts", data=np.asarray(INSERTS, dtype="S2"))
            output.create_dataset("projs", data=np.asarray(["P0", "Px", "Py", "Pz"], dtype="S2"))
            output.create_dataset("tfs", data=tfs)
            output.create_dataset("channels", data=np.asarray([channel_name(ch) for ch in channels], dtype="S"))
            output.create_dataset("status/completed_blocks", data=np.zeros(block_count, dtype=bool))
            output.create_dataset("notes", data=np.asarray(["time,mom,proj,insert", "mom=[sink,ins]; sink+ins=src", "source/PT/cubic average; before ratios"], dtype="S"))
            options = dict(compression=None if args.compression == "none" else args.compression,
                           shuffle=args.compression != "none", fletcher32=True)
            if args.compression == "gzip":
                options["compression_opts"] = 1
            for ch in channels:
                for tf in tfs:
                    output.create_dataset(f"data/{channel_name(ch)}_{tf}", (tf+1, len(moms), 4, 10), dtype="complex128",
                                          chunks=(tf+1, min(block_size, len(moms)), 4, 10), **options)
            if args.save_avgsrc:
                raw_table = np.asarray([v for m in moms for v in ref["mom2moms"](list(m))], dtype=np.int32)
                offsets = np.cumsum([0]+[len(ref["mom2moms"](list(m))) for m in moms])
                output.create_dataset("avgsrc/moms", data=raw_table)
                output.create_dataset("avgsrc/orbit_offsets", data=offsets)
                for ch in channels:
                    for tf in tfs:
                        output.create_dataset(f"avgsrc/data/{channel_name(ch)}_{tf}", (tf+1, len(raw_table), 4, 10), dtype="complex128",
                                              chunks=(tf+1, min(48, len(raw_table)), 4, 10), **options)
            output.attrs["initialized"] = True
            output.flush()
        done = output["status/completed_blocks"][:]
        log(f"cfg={args.cfg} sources={len(records)} mom={len(moms)} raw_pairs={len(all_raw)} block={block_size} cache_MiB={cache_bytes/1024**2:.1f} threads={args.threads} resumed={int(done.sum())}/{block_count}")
        coordinates = np.asarray([r[2] for r in records])
        c2f, c2b = read_c2(files, records, pvec, tfs, ref)
        loops = read_loops(loop_root/args.cfg, qvec, channels)
        if args.kernel == "fft":
            loops = {ch: np.fft.fft(a, axis=0) for ch, a in loops.items()}
        log(f"cfg={args.cfg} inputs_loaded_s={time.monotonic()-started:.2f} peak_RSS_MiB={peak_mib():.1f}")
        for bi, start in enumerate(range(0, len(moms), block_size)):
            stop = min(start+block_size, len(moms))
            if done[bi]:
                continue
            block = orbit_block(ref, moms[start:stop])
            wf, wb = time_weights(c2f, c2b, coordinates, block, pvec, temporal, ref["lat_L"])
            if args.kernel == "fft":
                wf = np.fft.ifft(wf, axis=0)*temporal
                wb = np.fft.fft(wb, axis=0)
            for ch in channels:
                for fi, tf in enumerate(tfs):
                    raw = contract(wf, wb, loops[ch], block.raw, qvec, fi, tf, len(records), args.kernel)
                    output[f"data/{channel_name(ch)}_{tf}"][:, start:stop] = cubic_average(raw, block)
                    if args.save_avgsrc:
                        a, b = output["avgsrc/orbit_offsets"][[start, stop]]
                        output[f"avgsrc/data/{channel_name(ch)}_{tf}"][:, a:b] = raw
            output.flush()
            output["status/completed_blocks"][bi] = True
            output.flush()
            log(f"cfg={args.cfg} block={bi+1}/{block_count} mom={stop}/{len(moms)} elapsed_s={time.monotonic()-started:.2f} peak_RSS_MiB={peak_mib():.1f}")
            del wf, wb
        # Detect inputs replaced while this configuration was being processed.
        if manifests != [file_manifest(Path(m["path"])) for m in manifests]:
            raise RuntimeError("An input file changed during production; output remains partial")
        output.attrs.update(complete=True, elapsed_seconds=time.monotonic()-started, peak_rss_mib=peak_mib(),
                            threadpools=json.dumps(threadpool_info(), sort_keys=True))
        output.flush()
    os.rename(partial, final)
    log(f"flag_cfg_done: {args.cfg} elapsed_s={time.monotonic()-started:.2f} peak_RSS_MiB={peak_mib():.1f} file={final}")


def main():
    args = build_parser().parse_args()
    with threadpool_limits(limits=args.threads):
        produce(args)


if __name__ == "__main__":
    main()
