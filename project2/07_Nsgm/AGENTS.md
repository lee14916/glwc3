# N-sigma Project Guide

This file contains project-specific context and decisions for the N-sigma paper and analysis. Repository-level `agents_paper_writing.md` and `agents_glwc_coding.md` contain reusable writing and coding conventions. Apply those general conventions here, and use this file only for N-sigma-specific scope, notation, data contracts, analysis selections, and ownership.

## Project Authority and Ownership

- The Overleaf project is the authoritative manuscript: `https://www.overleaf.com/project/6ab365d9b158ed72c19cdfd8`. Local clone: `C:/Users/yan14/Documents/ChatGPT/Nsgm/Nsgm-paper`. The Git remote is Overleaf, not GitHub. Check `git status`, fetch, and compare before editing so collaborator updates are preserved.
- Every push to Overleaf requires the user's explicit permission for that push. Permission to clone, fetch, pull, edit, commit, compile, or sync `glwc3` with GitHub is not permission to push. Never automate Overleaf pushes. Overleaf warns that Git pushes may displace comments and tracked changes. Keep credentials out of source files and chat.
- The former OneDrive manuscript workflow is retired. Do not publish by copying from the OneDrive manuscript or using `paper_sync_guard_codex_ignore.ps1` for Overleaf changes. The OneDrive topology sources remain authoritative in their original directory and must not be duplicated into the repository.
- Independently editable analysis code must have `_codex` in its filename. Obtain explicit approval before changing user-controlled files such as `util.py`, `util_Nsgm.py`, `processData.ipynb`, or preparation code. Approval for one change is not standing permission.
- Generated N-sigma figures, caches, previews, logs, and diagnostics belong under `project2/07_Nsgm/__codex_ignore`. Never create root `fig`, `pkl`, `tmp`, `.ipython`, or `.jupyter` output directories. Preserve user inputs and `_wrong` diagnostic data.
- Check Git status before and after work. Preserve unrelated edits and never discard user changes.

## Maintained Analysis

- Maintained ensemble work is organized under A24 (`cA211.530.24`), A48 (`cA2.09.48`), and B64 (`cB211.072.64`). Only genuine cross-ensemble comparisons belong at project level. Do not restore `analysis_3pt_light_common_codex.py` or a combined ensemble fit.
- Read `reproduce_figures_codex.py` for the current publication inventory and execution order. It owns figure reproduction. Read current notebooks and data before relying on saved plot/fit outputs; historical caches and this guide do not override current selections.
- Shared analysis and plotting helpers live in `util_codex.py`. Keep common implementations there rather than duplicating them in notebooks. Maintained notebooks should explain the analysis in concise Markdown and follow the scientific workflow order.
- The three ensemble `processData_codex.ipynb` notebooks produce compact Appendix D inputs. These pickles are retained analysis inputs, not disposable caches. The processing notebooks are a separate raw-data stage and are not rerun for routine figure regeneration.
- Use `yu.jackme`, `yu.jackmec`, and `yu.jackme_un2str` for the established resampling/error conventions. Keep replica/configuration identity and ordering aligned for paired arithmetic. Matching sample counts alone does not establish matched configurations.
- Errors on sums, differences, products, and externally fitted quantities must be computed from aligned samples. Never jackknife an already jackknifed denominator.
- Preserve covariance choices, masks, priors, fit models, and input configurations unless explicitly changed. Numerical convergence and parameter resolution are distinct; report unresolved fits accurately.
- Some fit caches are keyed by labels rather than inputs. Check cache provenance, input hashes, ranges, model, and covariance before reuse. Do not silently substitute stale fits for failed current fits.

## Data Contracts

- Appendix D compact inputs contain full two-point/three-point matrices, matched denominators, separate BWZ/direct contributions, operators, configuration identity/order, retained/discarded configuration metadata, block size, separations, and provenance hashes. They do not contain fitted eigenvector weights.
- Production resampling: A24 has 2467 raw configurations, block size 4, 616 samples, and three unused trailing configurations; A48 has 1228 configurations, block size 2, 614 samples; B64 has 732 configurations, block size 1, 732 samples. Preserve original ordering and do not silently trim to another set.
- A24/A48 matched denominators are `real(c2[:,ts,0,0])`; B64 light connected analyses use the saved source-matched denominator. Disconnected strange/charm use the two-point matrix denominator.
- B64 `data_jsc.pkl` layout is `[c2, light_c3, matched_c2, strange_c3, charm_c3]`; `data_jsc_NsgmJNsgm.pkl` is `[c2, strange_c3, charm_c3]`. Both have 732 samples and are distinct from the independent 725-configuration standard-only check.
- In the full B64 strange/charm input, every integer separation 2..30 is available. The c2 and original three c3 entries must match `data_jsc.pkl` exactly; only the lower-right c3 entry is new.
- Full strange/charm scalar lower-right topologies include B2pt-j, W2pt-j, Z2pt-j, N-P-j, N-pi0f-j-pi0i, T-pi0f-j, and its reflected/conjugated partner. Remote raw completeness is not locally validated end-to-end; state the narrower source/synthetic checks accurately.
- BWZ means only B3pt/W3pt/Z3pt. “Direct” means N-jPi/N-jPf plus N-j-pi0i/N-pi0f-j. NJN-pi0i/f and T-j with its reverse partner are excluded from that approximation. These approximations change only off-diagonal three-point entries; keep NJN, denominators, weights, samples, and symmetrization fixed, and compute paired differences before errors.
- Reverse two-point terms use swapped operators. Three-point partners are conjugated and reflected in insertion time. Apply vacuum inclusion-exclusion and row/Hermitian averaging once. For symmetric fits, retain each reflected pair once; odd midpoint values average the two central insertions.
- Published ensemble masses and spacings are authoritative. A24 uses `a=0.09076 fm`, `m_pi=361.6 MeV`; A48 uses `a=0.0938 fm`, `m_pi=130.6(4) MeV`; B64 uses `a=0.07948 fm`. Do not mix A24's newer spacing with the older 346 MeV mass. A48's scalar conversion uses `(135/ens2mpi['a'])^2`; A24 retains the simulated mass.
- B64 central isosymmetric bare quark masses are light `0.0006669`, strange `0.018267`, charm `0.23134`. A48 leading matching to 135 MeV gives light mass `0.000962(6)` from `0.0009`; A24 retains `0.0053`.

## Manuscript Scope and Conventions

- Title: “Excited-State Effects in Nucleon Sigma Terms from a Variational Analysis.” B64 is the main-text focus; A24 and A48 are discussed in Appendix C. Call B64 and A48 physical-point ensembles while retaining their simulated pion masses and the stated isoQCD corrections.
- Use `N\sigma` for the interpolator basis and expectation-value notation for correlators. Do not use internal strings such as `NsgmJNsgm` in prose. For the additional matrix element use `(j,k)=(N\sigma,N\sigma)` after defining the matrix.
- Use `\mathcal J` for interpolators and `\mathcal O` for the scalar insertion. Use `m_\ell` for the degenerate light mass and `m_s,m_c` for strange/charm. Identify lattice quark masses as bare and use `\rm sim`/`\rm iso` superscripts.
- Call both `R_{\rm GEVP}^{d}` and `R_{\rm GEVP}` GEVP-improved ratios. Use “full” for `R_{\rm GEVP}` when useful. Avoid “reduced” in manuscript prose/captions and avoid bare “projection” for GEVP improvement. “GEVP projection” is acceptable for the mathematical operation. Only the W-weighted ratio has superscript `d`; prime variants are `R_{\rm GEVP}^{\prime}` and `R_{\rm GEVP}^{\prime\prime}`.
- All two-point energies/gaps after GEVP improvement carry tildes. Standard nucleon inputs remain untilded. Use `C_{\rm GEVP}^{\rm 2pt}`, `[v^{-1}]_{N0}`, `t_{\rm ins}^{\rm cut}`, and `\delta/a` for the projected denominator, inverse component, insertion cut, and displacement.
- `R_{\rm LG}` is the Laplace-filtered W construction. Use `\Delta E_1^{\rm 3pt}` and `\Delta E_1^{\rm 3pt,Lap}` for the standard and Laplace gaps. A prime does not denote Laplace filtering.
- Use zero momentum; do not add “spatial” unnecessarily. Finite-volume noninteracting levels are the explicit nonzero-momentum exception.
- Figures/captions must match current analysis and use first-appearance conventions. Cite the N-pi GEVP paper for topology-computation details. In Fig. 1, red squares are source nucleons and red lines are point propagators. Fig. 2 adds the crossed-circle current; Fig. 3 refers back without re-explaining sigma contractions.
- Standard light ratio fit cases are I–V. B64 selects standard IV at lower 12/cut 2, about 45.9(3.0) MeV with transition gap 405(35) MeV. Do not present this as a final light sigma-term determination.
- The unfiltered light `R_{\rm GEVP}^{d}` is not fitted with a constant or single-excited-state model because filtering indicates competing residual contributions. B64 Laplace IV uses (10,4), gap about 850(190) MeV. The light filter gap is a different quantity.
- The light Laplace response at the adopted discretization and central gaps 850 and 360 MeV is -4.76 for the stated ground-to-excited response ratio. It implies sign reversal and about fivefold amplification. Interpret this as support for at least two partially cancelling effective excited contributions, not a unique state assignment.
- Strange standard comparisons show I, IV, VI. VI fixes the transition gap to the selected light-IV `\Delta E_1^{\rm 3pt}`, not the Laplace gap. Strange uses AIC averaging over six constant windows; do not mark a selected/highest-weight window. The cited window spread is not a continuum/finite-volume error budget.
- Strange full and W-weighted GEVP-improved ratios are compatible; the full ratio is noisier. Retain `R_{\rm GEVP}^{d}` for strange extraction. Filtering the strange ratio is a noisy diagnostic, not evidence that higher states are absent. Do not quote a charm sigma-term result or add charm fits.
- Appendix C retains two combined A24/A48 light-current ratio/midpoint figures: standard versus GEVP-improved, then GEVP-improved versus Laplace-filtered. Do not restore removed A-ensemble two-point/eigenvector/overlap/fit-scan/energy-summary figures. Discuss A24 as qualitative support and A48 as suggestive but less decisive, not as establishing a common excitation energy.
- The strange literature comparison is included in Conclusions; light and charm literature plots remain previews. Preserve source error decomposition and do not form a cross-publication average. Inner and outer uncertainty bars use the same line weight; their lengths distinguish statistical and total uncertainty.
- Appendix order: ChPT pion-mass corrections; Laplace discretization stability; A24/A48 light-current GEVP analysis; off-diagonal contraction approximations. Use standard two-column flow and ordinary REVTeX floats.

## Bibliography Compatibility

- The manuscript uses REVTeX 4.2's default APS bibliography style with `longbibliography` to display article titles.
- Four INSPIRE BibTeX exports have an approved compatibility exception: `journal = {}` was added to `Borsanyi:2020bpd`, `Liang:2025adz`, `Wang:2025nsd`, and `Alexandrou:2026oks`, because the default `apsrev4-2` style otherwise fails on those journal-less entries. Do not extend this exception without user approval.

## Verification and Handoff

- For code changes, inspect upstream data contracts and downstream selections. Check dimensions, symmetry, normalization, insertion support, cache provenance, and numerical/visual baselines as relevant.
- Execute edited notebooks fully. Do not rerun discarded A-ensemble matrix-element, ratio-variant, double-filter, or displacement diagnostics during routine figure reproduction.
- Read `figure_report.json` after reproduction. A stale PDF does not satisfy a missing output. Inspect changed pages and report unresolved fits, warnings, changed results, and figure renumbering.
- The September 2026 processing audit compared 72/120/168 three-point diagram matrices on A24/A48/B64 plus two-point diagrams and full production matrices. Reproduction with raw file access blocked gave identical midpoint means/errors. This did not validate the large remote strange/charm raw input end-to-end.
- For current selections and file contracts, prefer maintained code/notebook outputs over fixed numerical notes in this guide. Update this guide when a scientific choice changes; remove obsolete guidance rather than appending contradictory history.
