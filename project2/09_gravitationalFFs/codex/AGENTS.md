# Gravitational form factors: Codex working conventions

These instructions apply to `project2/09_gravitationalFFs/codex/` and its descendants in every checkout.
The HPC project directory is `/p/project1/ngff/li47/code/glwc3/project2/09_gravitationalFFs/`.

Follow the shared `glwc3/agents_glwc_coding.md` guide. For manuscript work, also consult `glwc3/agents_paper_writing.md`.

## File locations and ownership

- Put well-organized, reusable assistant-written code in `codex/` or suitable subdirectories.
- Put disposable scripts, one-off diagnostics, small exploratory tests, and their temporary outputs in `codex/_ignore/`.
- Maintained validation code can live with the reusable implementation; it is not automatically disposable.
- Preserve the existing lowercase directory names `codex` and `_ignore`.
- Files outside `codex/` are primarily the user's working area. Modify or move them only when the user's requested task calls for it; do not reorganize them as part of routine Codex work.
- The user may move adopted code outside `codex/`. Respect that organization and avoid creating competing maintained copies.

## Reuse and reproduction

- The user will refine which implementations and results should become maintained project sources as the work develops.
- When code is adopted for reuse or is essential to reproduce a key result, keep its scientific logic in maintained sources under `codex/`, with the entry point, inputs, relevant settings, and output locations documented.
- Scratch runners in `_ignore/` may orchestrate maintained code. Essential reproduction logic must not remain only in disposable scratch files.
- Reuse the user's established utilities and scientific conventions, including the earlier `project2/05_moments/` implementation where relevant.
- Run large data-production tasks on the HPC and keep bulk datasets there. Keep temporary diagnostics and logs in `_ignore/` unless a task specifies another location.
- Record confirmed project conventions here as they develop. Keep proposed momentum cuts, fit choices, and other unsettled scientific decisions separate from established rules.

## Local and HPC work

- Synchronize maintained project sources through the existing `glwc3` GitHub repository so the local and HPC checkouts can both be used.
- Use local work for source editing and lightweight diagnostics. Keep bulk production data and large computations on the HPC.
- Check both working-tree states before synchronizing and preserve unrelated edits. Stage and publish only changes belonging to the current task.

## Existing kinematic preview

- The earlier `project2/05_moments/general.ipynb` contains `getRank`, `getRanks`, and `ranks2fillstyle`, followed by the `momDistribution` plot.
- That preview distinguishes diagonal (`equal`), off-diagonal (`unequal`), and combined (`all`) tensor-component rank. It is a kinematic rank diagnostic; measured-data combinations also need the appropriate renormalization and mixing for each tensor sector.
