"""Small synthetic numerical checks against the unmodified legacy functions.

Run: python -m unittest discover -s <this directory> -p test_disconnected_3pt.py
The scientific environment must provide h5py, NumPy, SymPy and threadpoolctl.
No production data or HPC jobs are used.
"""

from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import copy
import io
import json
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

import disconnected_3pt as core


class LegacyNumericalEquivalence(unittest.TestCase):
    temporal = 12
    tfs = (2, 5, 7)
    currents = ("j+", "js", "jc")
    stouts = (0, 40)
    channels = currents + tuple(f"jg;stout{s}" for s in stouts)

    @classmethod
    def setUpClass(cls):
        cls.ref = core.load_reference(core.DEFAULT_REFERENCE, tfs=cls.tfs,
                                      currents=cls.currents, stouts=cls.stouts)
        # Include singleton/high-stabilizer, collinear and generic 48-element
        # spatial orbits. Source/sink exchange is deliberately absent.
        examples = [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1),
                    (0, 0, 1, 0, 0, 0), (0, 0, 1, 0, 1, 0),
                    (1, 1, 0, 1, 0, 1)]
        cls.moms = np.asarray([cls.ref["mom3pt2standard"](list(m)) for m in examples])
        cls.block = core.orbit_block(cls.ref, cls.moms)
        cls.pvec = np.asarray(core.vectors(2), dtype=np.int32)
        cls.qvec = cls.pvec.copy()

    def setUp(self):
        scratch = (Path(__file__).resolve().parent/"_ignore"/"unittest_disconnected_3pt").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temporary.name).resolve()
        self.assertEqual(self.directory.parent, scratch)
        self.addCleanup(self.temporary.cleanup)
        self.make_inputs(np.complex64)

    def make_inputs(self, dtype):
        rng = np.random.default_rng(2601026)
        def values(shape):
            return (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(dtype)
        # An identical source label in distinct C2 files must remain two
        # observations with distinct numerical C2 values and the same phase.
        labels = [
            ("sx001sy002sz003st000", "sx004sy001sz002st011", "sx003sy005sz001st003"),
            ("sx001sy002sz003st000", "sx000sy001sz005st011", "sx002sy003sz001st010"),
        ]
        self.paths = []
        for number, sources in enumerate(labels):
            path = self.directory/f"N.h5_{number}"
            self.paths.append(path)
            order = rng.permutation(len(self.pvec))
            with h5py.File(path, "w") as f:
                f["moms"] = self.pvec[order]
                for group in ("data", "data_bw"):
                    for source in sources:
                        for nucleon in ("N1_N1", "N2_N2"):
                            f[f"{group}/{source}/{nucleon}"] = values((self.temporal, len(order), 4))
        gamma = ("gy", "gt", "gz", "gx")
        order = rng.permutation(len(self.qvec))
        with h5py.File(self.directory/"j.h5", "w") as f:
            f["moms"] = self.qvec[order]
            f["inserts"] = np.asarray(gamma, dtype="S2")
            for current in self.currents:
                for derivative in range(4):
                    f[f"data/{current};d{derivative}"] = values((self.temporal, len(order), 4))
        order = rng.permutation(len(self.qvec))
        with h5py.File(self.directory/"jg.h5", "w") as f:
            f["moms"] = self.qvec[order]
            f["inserts"] = np.asarray(core.INSERTS, dtype="S2")
            for stout in self.stouts:
                f[f"data/jg;stout{stout}"] = values((self.temporal, len(order), 10))
        _, self.records, _, temporal = core.input_inventory(
            self.directory, self.directory, self.pvec, self.qvec, self.tfs, self.channels)
        self.assertEqual(temporal, self.temporal)
        self.coordinates = np.asarray([record[2] for record in self.records])

    def oracle(self, mom):
        labels, fwd, bwd, difference, difference_bw = self.ref["extract2pt"](self.paths, list(mom))
        loops = self.ref["extractLoop"](str(self.directory), list(mom))
        raw = self.ref["correlate"](labels, fwd, bwd, difference, difference_bw, loops, list(mom))
        averaged = self.ref["avgmore"](raw, list(mom))
        return labels, fwd, bwd, loops, raw, averaged

    def production_args(self, label, kernel="direct"):
        selected = self.directory/"selected_momenta.json"
        selected.write_text(json.dumps(self.moms.tolist()), encoding="utf-8")
        return core.build_parser().parse_args([
            "--cfg", self.directory.name, "--c2-root", str(self.directory.parent),
            "--loop-root", str(self.directory.parent), "--output-root", str(self.directory/"outputs"),
            "--run-label", label, "--sink-n2-max", "2", "--transfer-n2-max", "2",
            "--currents", ",".join(self.currents), "--stouts", "0,40",
            "--tfs", "7,2,5", "--momentum-block", "2", "--memory-mib", "512",
            "--compression", "none", "--save-avgsrc", "--momenta-json", str(selected),
            "--kernel", kernel,
        ])

    def output_path(self, args, partial=False):
        suffix = ".partial.h5" if partial else ".h5"
        return args.output_root/args.ensemble/args.run_label/"data_avgmore"/(args.cfg+suffix)

    def assert_compact_matches_oracle(self, path):
        oracles = [self.oracle(mom) for mom in self.moms]
        expected_channels = [j+";disc" for j in self.currents]+[f"jg;stout{s}" for s in self.stouts]
        expected_keys = {f"{ch}_{tf}" for ch in expected_channels for tf in self.tfs}
        with h5py.File(path, "r") as output:
            self.assertTrue(output.attrs["complete"])
            self.assertTrue(output.attrs["initialized"])
            self.assertTrue(np.all(output["status/completed_blocks"][:]))
            np.testing.assert_array_equal(output["moms"][:], self.moms)
            np.testing.assert_array_equal(output["tfs"][:], [7, 2, 5])
            self.assertEqual([ch.decode() for ch in output["channels"][:]], expected_channels)
            self.assertEqual(set(output["data"]), expected_keys)
            self.assertEqual(set(output["avgsrc/data"]), expected_keys)
            np.testing.assert_array_equal(output["avgsrc/moms"][:], self.block.raw)
            self.assertIn("def correlate", output["metadata/reference_source"][()].decode())
            for key in expected_keys:
                expected_raw = np.concatenate([o[4][key] for o in oracles], axis=1)
                expected_avg = np.concatenate([o[5][key] for o in oracles], axis=1)
                self.assertEqual(output[f"data/{key}"].dtype, np.dtype(np.complex128))
                np.testing.assert_allclose(output[f"data/{key}"][:], expected_avg, rtol=2e-12, atol=2e-13)
                np.testing.assert_allclose(output[f"avgsrc/data/{key}"][:], expected_raw, rtol=2e-12, atol=2e-13)

    def test_source_identity_time_buckets_and_complex64_c2_precision(self):
        self.assertEqual(len(self.records), 6)
        self.assertEqual(len({r[1] for r in self.records}), 5)
        forward, backward = core.read_c2(self.paths, self.records, self.pvec, self.tfs, self.ref)
        lookup = {tuple(m): i for i, m in enumerate(self.pvec)}
        for mom in self.moms:
            labels, old_f, old_b, _, _, _ = self.oracle(mom)
            self.assertEqual(labels, [r[1] for r in self.records])
            indices = [lookup[tuple(m[:3])] for m in self.ref["mom2moms"](list(mom))]
            np.testing.assert_allclose(forward[:, :, indices], old_f[:, self.tfs], rtol=1e-14, atol=1e-14)
            old_times = [-t for t in self.tfs]
            np.testing.assert_allclose(backward[:, :, indices], old_b[:, old_times], rtol=1e-14, atol=1e-14)

        actual_f, actual_b = core.time_weights(forward, backward, self.coordinates, self.block,
                                              self.pvec, self.temporal, 64, source_batch=2)
        expected_f, expected_b = np.zeros_like(actual_f), np.zeros_like(actual_b)
        for source, coordinate in enumerate(self.coordinates):
            time = coordinate[3] % self.temporal
            for pair, mom in enumerate(self.block.raw):
                phase = np.exp(1j*2*np.pi/64*np.dot(coordinate[:3], mom[3:]))
                expected_f[time, :, pair] += forward[source, :, lookup[tuple(mom[:3])]]*phase
                expected_b[time, :, pair] += backward[source, :, lookup[tuple(-mom[:3])]]*np.conj(phase)
        np.testing.assert_allclose(actual_f, expected_f, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(actual_b, expected_b, rtol=1e-14, atol=1e-14)

    def test_loop_symmetry_trace_and_precision_against_legacy(self):
        # complex64 raw inputs exercise the legacy float64-eye promotion at
        # trace subtraction, not only the algebra of the tensor components.
        lookup = {tuple(q): i for i, q in enumerate(self.qvec)}
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                self.make_inputs(dtype)
                loops = core.read_loops(self.directory, self.qvec, self.channels)
                for current in self.currents:
                    self.assertEqual(loops[current].dtype, np.dtype(np.complex128))
                for mom in self.moms:
                    old = self.ref["extractLoop"](str(self.directory), list(mom))
                    indices = [lookup[tuple(m[3:])] for m in self.ref["mom2moms"](list(mom))]
                    for current in self.channels:
                        key = current if current.startswith("jg;") else current+";disc"
                        np.testing.assert_allclose(loops[current][:, indices], old[key], rtol=1e-14, atol=1e-14)

    def test_all_48_cubic_elements_including_stabilizers(self):
        rng = np.random.default_rng(2648)
        raw = rng.normal(size=(3, len(self.block.raw), 4, 10)) + 1j*rng.normal(size=(3, len(self.block.raw), 4, 10))
        expected = []
        offset = 0
        orbit_sizes = []
        for mom in self.moms:
            count = len(self.ref["mom2moms"](list(mom)))
            orbit_sizes.append(count)
            expected.append(self.ref["avgmore"]({"x": raw[:, offset:offset+count]}, list(mom))["x"])
            offset += count
        self.assertIn(1, orbit_sizes)
        self.assertIn(6, orbit_sizes)
        self.assertIn(48, orbit_sizes)
        np.testing.assert_allclose(core.cubic_average(raw, self.block), np.concatenate(expected, axis=1),
                                   rtol=1e-14, atol=1e-14)

    def test_direct_and_fft_raw_and_averaged_match_legacy(self):
        forward, backward = core.read_c2(self.paths, self.records, self.pvec, self.tfs, self.ref)
        loops = core.read_loops(self.directory, self.qvec, self.channels)
        oracles = [self.oracle(mom) for mom in self.moms]
        # Different representative block sizes and source tiles must not change
        # the estimator. T=12 with st=0,10,11 exercises both temporal wraps.
        for block_size, source_batch in ((1, 1), (3, 2), (len(self.moms), 7)):
            for first in range(0, len(self.moms), block_size):
                last = min(first+block_size, len(self.moms))
                block = core.orbit_block(self.ref, self.moms[first:last])
                wf, wb = core.time_weights(forward, backward, self.coordinates, block,
                                            self.pvec, self.temporal, 64, source_batch=source_batch)
                wf_fft = np.fft.ifft(wf, axis=0)*self.temporal
                wb_fft = np.fft.fft(wb, axis=0)
                for current in self.channels:
                    old_key = current if current.startswith("jg;") else current+";disc"
                    loop_fft = np.fft.fft(loops[current], axis=0)
                    for fi, tf in enumerate(self.tfs):
                        with self.subTest(block_size=block_size, first=first, current=current, tf=tf):
                            expected_raw = np.concatenate([o[4][f"{old_key}_{tf}"] for o in oracles[first:last]], axis=1)
                            expected_avg = np.concatenate([o[5][f"{old_key}_{tf}"] for o in oracles[first:last]], axis=1)
                            direct = core.contract(wf, wb, loops[current], block.raw, self.qvec, fi, tf, len(self.records), "direct")
                            fft = core.contract(wf_fft, wb_fft, loop_fft, block.raw, self.qvec, fi, tf, len(self.records), "fft")
                            np.testing.assert_allclose(direct, expected_raw, rtol=2e-12, atol=2e-13)
                            np.testing.assert_allclose(fft, expected_raw, rtol=2e-12, atol=2e-13)
                            np.testing.assert_allclose(core.cubic_average(direct, block), expected_avg, rtol=2e-12, atol=2e-13)
                            np.testing.assert_allclose(core.cubic_average(fft, block), expected_avg, rtol=2e-12, atol=2e-13)

    def test_fused_channel_contraction_matches_single_channel(self):
        forward, backward = core.read_c2(self.paths, self.records, self.pvec, self.tfs, self.ref)
        available = core.read_loops(self.directory, self.qvec, self.channels)
        # A deliberately reversed order catches channel-axis reshaping mistakes.
        loops = {ch: available[ch] for ch in reversed(self.channels)}
        stacked = np.stack(list(loops.values()), axis=-1)
        for block_size in (1, 3, len(self.moms)):
            for first in range(0, len(self.moms), block_size):
                block = core.orbit_block(self.ref, self.moms[first:first+block_size])
                wf, wb = core.time_weights(forward, backward, self.coordinates, block,
                                            self.pvec, self.temporal, 64, source_batch=2)
                for fi, tf in enumerate(self.tfs):
                    with self.subTest(block_size=block_size, first=first, tf=tf):
                        expected = np.stack([core.contract(wf, wb, loop, block.raw, self.qvec,
                                                           fi, tf, len(self.records), "direct")
                                             for loop in loops.values()], axis=-1)
                        actual = core.contract_channels(wf, wb, loops, block.raw, self.qvec,
                                                        fi, tf, len(self.records), "direct", stacked)
                        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
                        without_cache = core.contract_channels(wf, wb, loops, block.raw, self.qvec,
                                                               fi, tf, len(self.records), "direct")
                        np.testing.assert_allclose(without_cache, expected, rtol=2e-12, atol=2e-13)
                        fft_loops = {ch: np.fft.fft(loop, axis=0) for ch, loop in loops.items()}
                        fft = core.contract_channels(np.fft.ifft(wf, axis=0)*self.temporal,
                                                     np.fft.fft(wb, axis=0), fft_loops,
                                                     block.raw, self.qvec, fi, tf, len(self.records), "fft")
                        np.testing.assert_allclose(fft, expected, rtol=2e-12, atol=2e-13)

    def test_compact_production_save_avgsrc_skip_and_identity(self):
        args = self.production_args("fresh")
        with redirect_stdout(io.StringIO()):
            core.produce(args)
        path = self.output_path(args)
        self.assertFalse(self.output_path(args, partial=True).exists())
        self.assert_compact_matches_oracle(path)
        timestamp = path.stat().st_mtime_ns
        with mock.patch.object(core, "read_c2", side_effect=AssertionError("Completed output reread payloads")):
            with redirect_stdout(io.StringIO()):
                core.produce(args)
        self.assertEqual(path.stat().st_mtime_ns, timestamp)
        changed = copy.copy(args)
        changed.stouts = "0"
        with self.assertRaisesRegex(ValueError, "different run identity"):
            with redirect_stdout(io.StringIO()):
                core.produce(changed)
        self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_resume_completed_blocks_fft_and_nonblocking_lock(self):
        fresh = self.production_args("fresh_direct")
        with redirect_stdout(io.StringIO()):
            core.produce(fresh)
        resumed = self.production_args("interrupted_fft", kernel="fft")
        original_contract = core.contract_channels
        calls = 0
        per_block = len(self.tfs)
        def stop_after_first_block(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == per_block+1:
                raise RuntimeError("Injected interruption after completed first block")
            return original_contract(*args, **kwargs)
        with mock.patch.object(core, "contract_channels", side_effect=stop_after_first_block):
            with self.assertRaisesRegex(RuntimeError, "Injected interruption"):
                with redirect_stdout(io.StringIO()):
                    core.produce(resumed)
        partial = self.output_path(resumed, partial=True)
        self.assertTrue(partial.exists())
        self.assertFalse(self.output_path(resumed).exists())
        with h5py.File(partial, "r") as output:
            np.testing.assert_array_equal(output["status/completed_blocks"][:], [True, False, False])
            self.assertFalse(output.attrs["complete"])
        with redirect_stdout(io.StringIO()) as captured:
            core.produce(resumed)
        self.assertIn("resumed=1/3", captured.getvalue())
        self.assertFalse(partial.exists())
        self.assert_compact_matches_oracle(self.output_path(resumed))
        with h5py.File(self.output_path(fresh), "r") as left, h5py.File(self.output_path(resumed), "r") as right:
            for group in ("data", "avgsrc/data"):
                self.assertEqual(set(left[group]), set(right[group]))
                for key in left[group]:
                    np.testing.assert_allclose(left[f"{group}/{key}"][:], right[f"{group}/{key}"][:],
                                               rtol=2e-12, atol=2e-13)
        lock = self.directory/"lock_check"/"cfg.lock"
        with core.configuration_lock(lock):
            with self.assertRaises(OSError):
                with core.configuration_lock(lock):
                    self.fail("A second writer acquired the same configuration lock")
        # Releasing the first writer must make the same persistent inode usable.
        with core.configuration_lock(lock):
            pass


if __name__ == "__main__":
    unittest.main()
