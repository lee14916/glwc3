# Disconnected 1DV three-point production

`disconnected_3pt.py` constructs source-, forward/backward- and cubic-averaged
disconnected nucleon three-point functions, before ratios or GFF extraction.
It reuses geometry and projector conventions from
`../../05_moments/dataPrepare/cB211.072.64/post2avgmore.py`. Only the `1DV`
operator is supported. This producer retains all ten symmetric tensor
components; downstream diagonal and off-diagonal analyses remain separate.

## Defaults and inputs

| Setting | Default |
| --- | --- |
| Ensemble | `cB211.072.64` |
| Sink and insertion momentum cuts | Each integer squared norm `<=16` |
| Initial momentum cut | None; `--initial-n2-max` adds one |
| Quark currents | `j+,js,jc` |
| Gluon stout steps | `5,7,10,13,15,20,25,30,35,40` |
| Source-sink separations | `--tfs 2:22`, inclusive |
| Kernel and precision | Source-time buckets with direct BLAS; complex128 |
| Run label | `K16_stout10` |
| Momentum block / working-array budget | At most 4 representatives / 1024 MiB |
| BLAS threads / compression | 1 / gzip level 1 |

Momenta are in units of `2*pi/L`. A stored row is
`[sink_x,sink_y,sink_z,ins_x,ins_y,ins_z]`, with **source = sink + insertion**
in the legacy convention. Cubic averaging preserves the oriented sink cut;
source/sink exchange is not included. Without an initial-momentum cut, source
squared norms can reach 64 at these cuts. `--initial-n2-max 25` restricts the
selection to initial momenta covered by the currently extracted two-point
files; otherwise that coverage must be supplied separately for later ratios.

For configuration `CFG`, the default input directories are:

```text
/p/project1/ngff/li47/code/projectData/02_discNJN_1D/ENSEMBLE/
    data_post_hold_2pt_moreMom/CFG/N.h5*
    data_post_hold/CFG/j.h5
    data_post_hold/CFG/jg.h5
```

`--c2-root` and `--loop-root` override the respective directories above `CFG`.
The producer checks selected sink/loop momentum and channel coverage before
reading payloads. `--currents ''` selects gluons only; `--stouts ''` selects
quarks only. `--kernel fft` is an alternative checked against the same legacy
estimator. Direct BLAS is the default selected by synthetic kernel studies;
real-data throughput and peak memory still require a representative pilot.

## Run and resume

Run computation in an allocated CPU job. For one configuration:

```bash
python disconnected_3pt.py --cfg CFG --threads 1 --run-label K16_stout10
```

`--memory-mib` is a conservative working-array estimate, not an operating-system
RSS limit; the producer may reduce the momentum block or refuse an insufficient
budget. Logs report loading time, block progress and peak RSS. Keep the same
inputs, code and settings when resuming. A changed source/reference hash,
input manifest or effective configuration rejects an existing output. Use a
new run label for changed selections or `--limit-momenta` pilots.

Output defaults to:

```text
/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs/
    ENSEMBLE/RUN_LABEL/data_avgmore/CFG.h5
    ENSEMBLE/RUN_LABEL/locks/CFG.lock
```

Each process holds a nonblocking per-configuration lock. **Keep the lock file**:
its persistent inode is intentional; the operating system releases the lock
when the process exits. Work is written to `CFG.partial.h5`, with flushed block
completion flags, then renamed to `CFG.h5` after completion. An interrupted
initialization is rejected and needs a fresh run label. A complete file with
the same identity is skipped. This is not a promise of recovery from HDF5
corruption or an unclean filesystem failure.

`run_disconnected_3pt.slurm` is an example, not a submitted job. It requests one
48-core CPU node with 96-GiB memory class, runs at most ten single-threaded
configuration processes with `xargs -P 10`, and sets a 1024-MiB working-array
budget per process. Adjust wall time after a real-data pilot. From the CPU
login, prepare a plain text list containing one unique configuration name per
line; blank lines and lines beginning `#` are ignored. For example:

```bash
CODE=/p/project1/ngff/li47/code/glwc3/project2/09_gravitationalFFs/codex
CFG_LIST=/absolute/path/to/cfgs.txt
RUN_DIR=/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs/cB211.072.64/K16_stout10
mkdir -p "$RUN_DIR/logs"
sbatch --chdir="$RUN_DIR/logs" "$CODE/run_disconnected_3pt.slurm" \
    "$CFG_LIST" cB211.072.64 K16_stout10
```

The example uses account `renormparton`, partition `batch`; a short pilot can
override partition/time with `sbatch --partition=devel --time=00:30:00 ...`.
JSC module environments differ between Booster and Cluster logins. The example
uses a login Bash shell and `--export=NONE`, then loads its modules on the
allocated compute node. Do not launch the ten-process command on a login node.

## Compact HDF5 layout

`data/CHANNEL_TF` has axes `(insertion_time, momentum, projector, insertion)`
and shape `(TF+1, Nmom, 4, 10)`, including insertion times `0..TF`.
Channel names remain `j+;disc`, `js;disc`, `jc;disc`, and `jg;stoutN`.
Tables `moms`, `tfs`, `channels`, `projs` (`P0,Px,Py,Pz`) and `inserts`
(`tt,tx,ty,tz,xx,xy,xz,yy,yz,zz`) define the axes. Metadata stores the run
identity and producer/reference sources; file attributes include schema,
completion status, elapsed time and peak RSS.

`--save-avgsrc` additionally stores source/PT-averaged, spatially unaveraged
rows under `avgsrc/`, with `avgsrc/moms` and `avgsrc/orbit_offsets`. These are
not source-resolved measurements. This larger output is disabled by default.
The compact per-configuration layout differs from the old per-momentum files;
a downstream reader adapter is required. These are correlators, not
vacuum-subtracted ratios or renormalized GFFs.

## Checks

With h5py, NumPy, SymPy and threadpoolctl available, run the small synthetic
legacy-equivalence checks from the repository root:

```bash
python -m unittest discover -s project2/09_gravitationalFFs/codex \
    -p test_disconnected_3pt.py
```

For existing real-data references, compare matching canonical momentum files:

```bash
python "$CODE/validate_disconnected_3pt.py" \
    --compact "$RUN_DIR/data_avgmore/CFG.h5" \
    --legacy-avgmore /absolute/path/to/legacy/avgmore/CFG \
    --currents j+,js,jc --stouts 5,10,40 --tfs 2,8,22 \
    --report "$RUN_DIR/logs/CFG.validation.json"
```

Add `--legacy-avgsrc /absolute/path/to/legacy/avgsrc/CFG` only when compact
`avgsrc/` output was requested. Select channels/times available in both files.
The validator checks matched rows against a per-momentum normalized-maximum
tolerance (default `1e-11`), reports comparison coverage, and exits nonzero on
failure. Its pass applies to reported overlap, not unobserved momentum rows;
also confirm `compact_marked_complete` in the report.
