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

## Verification and Handoff

- Scale tests to the risk and reach of the change. Run relevant existing tests or checks; do not add new tests unless requested or necessary to validate risky behavior.
- Report what changed, why, what verification ran, and any important limitation. Do not claim tests or checks passed if they were not run.
- Keep generated outputs, caches, and diagnostics out of source directories unless the project explicitly requires them there.
- Check the final diff and working-tree status. Preserve unrelated changes and identify the files that remain modified.
