# Analysis procedure registry

Each `runNNN_` prefix identifies a complete analysis procedure. A procedure
can contain several code files and datasets sharing its prefix; one ID does
not mean one code file. Record code and data dependencies explicitly.

Procedure `N` may reuse code or data assigned to IDs less than or equal to `N`.
Record existing unnumbered inputs and utilities as pre-ID dependencies.
Use three digits: `run001`, `run002`, `run003`, and so on, for code and run folders.
Add later procedures with their purpose, settings,
dependencies, output locations and execution status. Keep historical entries.

## run001: Preliminary gluon GFF signal study, stout level 10

**Purpose:** construct moving-sink disconnected gluon three-point functions,
then assess whether the available data support finite-momentum GFF extraction.

**Active code:** `run001_post2avgmore.py` in this directory.
Copyable single-configuration and original `xargs` launch commands are in its
header. The user will launch the complete configuration list.

### Fixed configuration cohort

- Ensemble: B64, `cB211.072.64`.
- Use the old common cohort of **725** configurations: the intersection of
  731 connected and 743 disconnected configurations.
- Authoritative ordered list, relative to the `glwc3` repository root:
  `project2/05_moments/dataPrepare/cB211.072.64/data_aux/cfgs_run`.
- Ordered-name SHA256, using LF separators without a final LF:
  `7b340fef3af0f7d12aa12b8b7fb63e0597bb00f35a6861b259e751f4b1ccdaf7`.
- Keep this fixed common cohort without a superjackknife extension.
  A larger path-availability list does not replace the cohort.

### Data construction settings

- Gluon channel only: `jg;stout10`; no quark-loop channels.
- One-derivative vector/EMT tensor conventions from the existing helpers.
- Sink and insertion cuts: integer momentum squared at most 16 for each.
  Momenta are `(2*pi/L)*n`, with `source = sink + insertion`.
- No additional initial/source-momentum cut.
- Source-sink separations: every integer from 2 through 22, inclusive.
- Preserve source, forward/backward and 48-element cubic averaging.
  Source/sink exchange is not performed during production.
- Store `complex128`, gzip-compressed `avgmore` only; no `avgsrc` output.
- Output directory:
  `/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs/cB211.072.64/run001_gluon_stout_level10/data_avgmore/`.
- Each configuration is stored as `<cfg>.h5`; its metadata records effective
  settings, input identities and producer/reference source identities.

### Dependencies and status

- Pre-ID scientific helpers:
  `project2/05_moments/dataPrepare/cB211.072.64/post2avgmore.py`.
- Pre-ID inputs: the selected configurations' source-resolved nucleon C2
  files and stout-level-10 gluon loops, as recorded in each output file.
- C3 construction code is implemented. A separate one-configuration check
  is required before the user's full-cohort launch; completion is not claimed.
- Ratios, matrix-element fits and the preliminary GFF signal assessment are
  planned subsequent steps of procedure `run001`.

The preserved `cB211.072.64/K16_stout10/data_avgmore/a0500.h5` is a pre-ID,
13-channel validation baseline. It remains historical and is not procedure
`run001` output. Temporary validation output belongs under the scratch root's
`_tests/` directory.
