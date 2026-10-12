# General Coding Guide

This guide contains reusable engineering practices. It does not define repository-specific ownership, file locations, data layouts, analysis selections, or project workflow. For N-sigma project context, consult `project2/07_Nsgm/AGENTS.md` when that project is in scope, and verify its time-sensitive details against current code and data.

## Before Editing

- Read the relevant code, tests, repository instructions, and working-tree state before changing behavior.
- Follow the local architecture and established helpers. Keep edits within the ownership boundaries and behavior implied by the request.
- Preserve user changes and unrelated work. Never revert edits you did not make. Ask how to proceed only when a conflicting change prevents safe completion.
- Obtain authorization before modifying files the user controls. Authorization for one change is not blanket permission for later changes.
- Prefer `rg`/`rg --files` for code search and structured parsers for structured formats.

## Implementation

- Make the smallest coherent change that solves the problem. Avoid unrelated refactors, abstractions without a clear payoff, and metadata churn.
- Prefer existing frameworks, libraries, and helper APIs. For domain problems with established engines or standards, use a proven implementation unless the user explicitly requests otherwise.
- Keep names descriptive and code understandable to the intended maintainer. Add comments only to explain non-obvious reasoning or invariants.
- Use ASCII by default unless the file's existing content or the domain requires other characters.
- Use patch-based edits for manual changes. Do not use shell redirection or ad hoc scripts as a substitute for deliberate file edits. Formatting tools and clearly mechanical transformations are exceptions.
- Never use destructive Git or filesystem commands without explicit authorization. Avoid force pushes and interactive Git workflows when a noninteractive alternative is available.

## Paper Figures

- Run exploratory tests not intended for publication in the code executor. If a file is needed, keep the one-off runner and its outputs in the project's ignored scratch area, not among maintained analysis sources.
- Once a result or figure is adopted for the paper, integrate its reproducible analysis and plotting into the canonical Git-tracked notebooks/scripts and reproduction inventory. A scratch script or saved figure alone is not a publication source. Keep scratch runners limited to orchestration of maintained code; do not leave essential analysis, fit selections, or plotting logic only in ignored files.
- Use the user's existing utility libraries, especially their fitting and resampling functions, for exploratory and publication analyses alike. Preserve their covariance, normalization, uncertainty, and cache conventions. Do not independently reimplement fitting logic or silently substitute a different fitter; discuss missing functionality before extending or replacing a user-controlled utility.
- Before changing a publication figure, inspect its current rendered version, plotting code, and manuscript caption. Extend the established style and helper APIs rather than designing a new presentation. Preserve color groups, marker meanings, open/filled conventions, reference bands, typography, and error-bar treatment unless a change is requested.
- Paper figures have no plot titles, explanatory footers, or grid lines by default. Put conventions and uncertainty explanations in the manuscript caption, not below the axes. Keep only the axis labels, concise method/group labels, legends, and annotations needed to identify the plotted results. Do not add numerical result columns unless requested. Deliberate group separators are not grid lines.
- Use the existing paper-style settings and check readability at the intended column or page width. Preview figures intended for a paper must already follow these publication conventions; general display or diagnostic plots may use a different presentation.
- When splitting a comparison, keep each analysis identifiable by the same marker across panels. Do not change a selected alternative to the default marker merely because it becomes the sole point in its row. Center a requested reference band on the selected result and use the specified uncertainty.
- Inner and outer error bars use the same line weight and cap length. Draw both behind opaque white-filled open markers so error-bar lines and caps do not appear inside the symbols.

## Verification and Handoff

- When plotting under repository-specific Matplotlib defaults, use the established reference-line helper or set both the line style and marker explicitly. Check the rendered figure: an inherited empty line style can leave only endpoint markers. Offset estimates sharing a plot row enough to distinguish their markers and error bars.

- Scale tests to the risk and reach of the change. Run relevant existing tests or checks; do not add new tests unless requested or necessary to validate risky behavior.
- Report what changed, why, what verification ran, and any important limitation. Do not claim tests or checks passed if they were not run.
- Keep generated outputs, caches, and diagnostics out of source directories unless the project explicitly requires them there.
- Check the final diff and working-tree status. Preserve unrelated changes and identify the files that remain modified.
