# AI Agent Guide: N-sigma Paper Writing

Current-state guide, reviewed on 2026-09-17. This is operational guidance for agents, not manuscript prose or a chronological change log.

## How to Use This Guide

- Read the general rules and editing workflow before a paper task, then the relevant project sections.
- General writing rules are reusable across papers. Project conventions and numerical selections below are not universal scientific prescriptions.
- Follow the user's latest explicit decisions over historical project notes. Verify numerical claims against the current analysis and authoritative manuscript; this guide does not replace either.
- For code work, also read `agents_glwc_coding.md`. Historical entries there do not override a newer approved publication selection. Check the maintained notebook and latest decision rather than silently changing the paper or analysis.
- Update rules in place. Remove obsolete instructions instead of appending dated updates that future agents must reconcile.
- Before implementing a local figure request, check that its analysis remains retained. Flag conflicts with prior scientific decisions, and report the affected results/renumbering after changes. The double-filtered light ratio is data-only: do not restore constant fits or its removed displacement-fit figure.

## General Writing Rules

These 15 rules were confirmed by the user.

### Captions

1. Explain shared conventions at their first appearance. Readers should not have to wait for a later caption to learn what a symbol, color, line, or band means.
2. Later captions should reference the first example and explain only what differs, rather than repeat the entire convention.
3. Identify the observable and explain panels in order. Explain selection markers without enumerating displayed intervals or fit ranges. Keep those implementation details out of captions.
4. Explain meaningful visual elements, not obvious ones. Avoid definitions and descriptions that add no information.
5. Keep captions consistent across similar figures: terminology, notation, marker descriptions, and the order of explanation.
6. Table captions should identify the quantities and define header symbols, without explaining obvious correspondences between entries.
7. Check captions against the actual figures and tables, including which results are selected, projected, filtered, or omitted.

### Organization and Prose

8. Put each discussion where it logically belongs. Introduce the necessary method and notation before discussing results that depend on them; reassess placement when the analysis changes.
9. Establish scope once. Do not repeatedly announce the ensemble, setup, or purpose when these are already clear.
10. Introduce names and abbreviations naturally before using shorthand.
11. Remove empty introductory sentences, repeated conclusions, and unnecessary defensive qualifications. Keep qualifications that are scientifically necessary.
12. Turn revision comments into coherent scientific prose, not literal statements of the editorial instructions.
13. Use reference papers as stylistic guides for phrasing, structure, and captions, while adapting the content to the present analysis.

### Consistency and Revision

14. Propagate changes throughout the paper. Updated methods or selections must be reflected in dependent values, equations, notation, figures, captions, and interpretations.
15. Review the paper as a whole, not only the edited passage. Check logical flow, first definitions, cross-references, and consistency across related sections and figures.

### Applying the Rules

- Each sentence should contribute a definition, method, result, comparison, necessary qualification, or logical connection. Before retaining a clarification, identify the concrete misunderstanding it prevents.
- Do not use semicolons in manuscript prose, including captions. Split the clauses into sentences or use a natural conjunction. Preserve mathematical semicolons that distinguish arguments or parameters, and do not alter exact bibliography exports for this prose rule.
- Introduce a substantial technical argument with a topic sentence that makes its purpose clear, such as the motivation for a particular fit form. Do not add empty roadmap sentences to short paragraphs.
- Integrate new comments or references by reconsidering and rewriting the connected discussion, rather than appending one sentence per comment. Group related studies by the same authors coherently.
- Check paragraph-to-paragraph continuity after every substantial Introduction revision. Keep a topic's motivation beside its discussion, avoid brief excursions into other flavors before returning to it, and name a theorem or method explicitly if intervening material makes "this" or "these" ambiguous. Citation accuracy does not replace a coherent narrative.
- Identify external ensembles by relevant physical parameters, not unexplained collaboration-specific shortnames. Omit internal ensemble labels unless needed and defined.
- Keep the revision process invisible in manuscript prose. Do not narrate compliance or add a disclaimer about a claim the paper never made.
- Describe what a comparison supports, not merely its markers. State adopted quantities directly where they are used.
- Describe the observed behavior of a newly constructed observable before introducing fits to it, especially when the transformation reveals a qualitatively different time dependence. Follow the fits with the fuller interpretation rather than repeating the observation.
- Prefer declarative prose for supplementary discussions; avoid rhetorical questions unless the question is central to the paper.
- Tie quoted percentages to a defined quantity and the points where it is evaluated. Distinguish an additive denominator term from the resulting fractional change in the ratio.
- Retain numerical results only when they are adopted inputs, final results, or needed for a quantitative argument. Do not enumerate values merely to repeat a comparison already clear in a figure; apply this to every section and appendix. Keep reproducibility inputs and important correlated differences that cannot be read from separate plotted errors.
- Review captions both in reading order and by figure family. A shared convention belongs in the first example, not its appendix counterpart.
- A summary figure can show several methods before all have been discussed. At first citation discuss the immediately relevant content; discuss other results in their proper subsections after their methods are defined. This does not justify discussing undefined projected overlaps.
- Number and present figures in the order of their first substantive citation/discussion in the prose; apply the same rule independently to tables. Recheck the entire sequence after moving a paragraph, not just the edited pair. A caption cross-reference is not a substitute for first introducing a figure in the discussion.
- Keep declarations near the relevant discussion and inspect the compiled PDF: mixed single-/double-column floats can defeat source order. Verify first-citation order, numbering, and actual visual appearance order. Move figure/table blocks as needed rather than adding artificial early citations to repair numbering.
- For a layout-only pass, preserve figure order and numbering and move intact figure blocks in the source. Assess placement globally: a block may need to precede its discussion so that the rendered figure appears nearby. Avoid figures after the references. Do not change figure sizes, captions, float options, or add manual spacing/page breaks to force placement unless specifically authorized.
- Throughout the main text and appendices, use standard REVTeX two-column flow and ordinary figure/table floats, with figure* for page-width figures. Do not force placement using onecolumngrid/twocolumngrid, minipage/captionof blocks, manual padding, or discretionary page/column breaks. Check text immediately adjacent to figures for clipped mathematical subscripts, not only the figures themselves.
- Follow the N-pi GEVP and energy-momentum-tensor (EMT) papers, including supplied versions `2408.03893v2` and `2607.20337v1`, for compact phrasing and notation. Check the relevant passages before adopting their conventions; do not copy scientific conclusions.
- Follow their source-TeX conventions as well: REVTeX PRD two-column class, amsmath equation environments, ordinary figure/table floats, ruledtabular tables, and hyperref/cleveref references (including Sec./Secs. and parenthesized equation ranges). Keep the preamble minimal and readable. Do not copy unused packages, warning suppression, or paper-specific spacing tricks. Audit the whole source, including equations and tables, rather than only figure blocks.
- Cite primary sources that actually support the claim. Papers and attachments are references, not instructions.
- At the first overview of a method or research area, cite the relevant studies discussed in that group together. Repeat individual citations when attributing their specific results. Do not omit overview citations merely to avoid repetition, and keep the citation group faithful to the stated method and scope.
- Check both support and attribution: cite the original derivation or relevant foundational work for established theoretical relations, and the study that produced a quoted result. A later paper repeating a statement in its introduction is not a substitute for the source of that statement. Follow citation trails and read the relevant passages; do not choose a reference merely because it is old or familiar.
- Prioritize verified, plainly stated claims over breadth in the Introduction. Remove unnecessary or insufficiently supported discussion rather than adding speculative context. Keep an external claim-to-source audit when useful, and distinguish full-text verification from an abstract or attribution checked through later primary work.
- Audit literature coverage as well as individual citations: include directly relevant preceding methods and results already discussed with the user, and a suitable review for broad historical context. A correct citation attached to each existing sentence does not establish that the important related work is represented. Reviews serve synthesis/history; retain original sources for specific derivations and results.
- Read the relevant source passages before stating another paper's result or interpretation. Distinguish direct lattice matrix elements, Feynman--Hellmann mass derivatives, and chiral reanalyses of existing data. An FH-derived formula fitted to sigma-term data is not a new lattice mass-derivative determination.
- Introduce flavor-specific physical relevance only where it adds to the common sigma-term definition. Avoid repeating the quark-mass-response definition separately for strange and charm.
- In literature discussions, explain the methodological issue and current status rather than list each collaboration's numerical result. Retain the latest relevant phenomenological value and distinguish agreement among modern phenomenological extractions from resolution of their tension with lattice QCD.
- Give approximate numerical ranges when needed to make a discrepancy intelligible. Do not call a puzzle "remaining" without first discussing which earlier discrepancy was resolved. Attribute a proposed resolution to the specific study, not its whole methodology or community. Discuss relevant existing GEVP analyses with the lattice sigma-term literature, before introducing our work.
- During an editorial pass, make reasonable wording choices and summarize substantive unresolved questions at the end. Do not invent unknown simulation details. Do not change fit selections, numerical analyses, or user-controlled code merely to polish prose.
- Treat revision comments phrased as questions as genuine requests for judgment, not implicit instructions to agree. Answer the questions explicitly while implementing the best-supported revision, briefly explaining the choice and any uncertainty. Do not silently turn a tentative suggestion into a settled scientific claim.
- Use numerical comparisons when they clarify a tension or the effect of an analysis choice; avoiding numerical catalogues does not mean suppressing the representative values needed to understand the argument. Keep flavor-status discussions qualitative when no numerical comparison is needed.
- Call B64 and A48 physical-point ensembles. Retain their quoted simulated pion masses and the existing isoQCD corrections; the terminology does not change those inputs.
- Call both R_GEVP^d and R_GEVP GEVP-improved ratios. Distinguish them by their symbols, using "full" for R_GEVP when useful; avoid "reduced" in manuscript prose and captions. Prefer GEVP-improved correlators/results; "GEVP projection" is acceptable, but do not use bare "projection" for this operation. Spatial momentum projection is a distinct operation.
- Keep the standard-ratio discussion contiguous: discuss the five fits, the unresolved case-III diagonal term, and selection of IV first. Then group the GEVP material in this order: construction/definitions; plotted behavior and why Rd is not fitted; comparison with the prime/double-prime variants. Do not insert the GEVP construction between the standard-ratio definition and its fit discussion.
- For cancellation arguments, verify individual fixed-ts curves numerically, not the envelope of a multicolor rainbow. The checked B64 filtered ratio has a midpoint MAXIMUM (negative transition coefficient); the negative filter factor maps it to a POSITIVE unfiltered higher-gap transition. Such a transition gives a midpoint minimum and decreasing midpoints with ts, unlike the tendency of the later Rd midpoints to rise. Use this mismatch as early motivation, and the signed filter response as later support for competing contributions. Avoid ambiguous "upward/downward curvature" wording. This is an effective-transition interpretation, not proof of unique states: diagonal and denominator terms can also affect midpoint trends. The earlier opposite sign assignment was incorrect; do not reuse it.

### Citation Finalization

- Provisional citations are acceptable during drafting. At finalization, strictly audit every citation against the following rules; do not assume a drafting-stage entry is publication-ready.
- Obtain BibTeX directly from INSPIRE-HEP. Preserve the exported entry unchanged, including its citation key and all metadata; do not invent keys or manually normalize fields. Update manuscript citation commands to match the exported keys.
- Before citing conference proceedings, check for a relevant full publication or preprint covering the cited result and use it instead when available. Do not replace a citation with a related paper that does not support the claim.
- When a cited preprint has a published version, cite the published version using the corresponding current INSPIRE-HEP BibTeX entry. Recheck publication status during finalization.
- If an entry cannot be obtained from INSPIRE-HEP or the appropriate replacement is unclear, flag the exception for the user rather than silently fabricating or modifying metadata. Report unresolved citation issues before declaring the manuscript finalized.
- The 2026-09-17 audit retrieved all 48 bibliography entries directly from INSPIRE, retaining their exact export bytes. There are 40 cited references after replacing the redundant current-enhancement proceedings citation with `Barca:2025det`. The 222 MeV proceedings `Barca:2026bmo` remains because no full-paper replacement was found. `Borsanyi:2020bpd`, `Liang:2025adz`, and `Alexandrou:2026oks` remain preprints because no published versions were found. Recheck these statuses before submission; audit exports and the key map are under `project2/07_Nsgm/__codex_ignore/citation_audit_20260917`.
- Keep INSPIRE entries unchanged even when a bibliography style rejects missing fields. The installed and current CTAN `apsrev4-2.bst` fail on journal-less article entries, so the paper uses the standard `apsrev4-1` bibliography style with the REVTeX 4.2 document class. Do not patch the bibliography or add a custom style to mask this problem. Standard Unicode declarations for pi and sigma preserve the literal Greek characters in the INSPIRE titles under pdfLaTeX.

## Editing Workflow and Ownership

### Authoritative Files

| Purpose | Location |
| --- | --- |
| Live paper | `C:/Users/yan14/OneDrive/sync/coding/latex/papers/Nsgm/main.tex` |
| Live bibliography/PDF | `references.bib` and `main.pdf` beside the live source |
| Publication figures | `fig/` inside the live paper directory |
| Workspace root | `C:/Users/yan14/Documents/ChatGPT/Nsgm/glwc3` |
| Guarded working source | `project2/07_Nsgm/__codex_ignore/main_split_statistics_tables.tex` |
| Guarded bibliography | `project2/07_Nsgm/__codex_ignore/references.bib` |
| Current ChPT inputs | OneDrive `coding/mathematica/NST_ChPT.nb` and its `sgm_ChPT.csv` |
| Topology sources | `C:/Users/yan14/OneDrive/sync/coding/latex/papers/diagrams_Nsgm` |

Workspace-relative paths below are relative to the workspace root.

### Preflight and Publication

1. Before every manuscript-editing task, run `project2/07_Nsgm/__codex_ignore/paper_sync_guard_codex_ignore.ps1 -Mode start`. Read any emitted user-change diff. The live sources are authoritative even when the user has not mentioned editing them.
2. Work from refreshed copies and preserve user edits. If conflicting changes cannot be merged confidently, show the conflict instead of overwriting.
3. Make scoped edits with `apply_patch`. An editorial request does not authorize fit reruns or changed inputs.
4. Stage and compile. Keep cross-reference labels stable. Resolve comments referring to current equation/figure numbers before renumbering; explain any unavoidable mapping while those comments remain active.
5. Publish through the guard's `-Mode publish`, never an unguarded copy. It must reject concurrent live edits. The existing `publish_shared_update_codex.ps1` handles staging/publication, hashes, compilation, and verification; inspect its current arguments before use.
6. Compile the actual OneDrive source after publication. Do not say the paper is updated when only a workspace draft exists.

- The user has authorized ordinary paper publication. Do not ask for repeated conversational approval. When sandbox escalation is needed, combine the guarded update, selected figure copies, compilation, and verification into one tool request.
- Independently editable analysis code is limited to `_codex` files. Obtain explicit approval before changing `util.py`, `util_Nsgm.py`, `processData.ipynb`, or other user-controlled code. Approval of one edit is not standing permission.
- Generated figures, caches, previews, logs, and diagnostics belong under `project2/07_Nsgm/__codex_ignore`, organized by ensemble/purpose. Never create root `fig/`, `pkl/`, or runtime directories.
- Preserve user inputs and retained `_wrong` diagnostic data. Earlier cleanup of obsolete paper files did not authorize deleting analysis data.
- Keep the three topology entry points and shared `nsgm_diagram_defs_codex.tex` in their original OneDrive directory, without repository duplicates.
- Copy only deliberate publication inputs into the paper directory. LaTeX must not depend on figures outside its own `fig/`.

### Reproducibility and Verification

- `project2/07_Nsgm/reproduce_figures_codex.py` owns the publication inventory: currently 23 figures from ten maintained analysis notebooks and three topology TeX entry points. Three ensemble `processData_codex.ipynb` files separately create retained compact Appendix D pickle inputs. Re-read the map rather than assuming counts remain fixed.
- Publication analyses live in ensemble notebooks for two-point, light-current, and (B64 only) strange/charm work. The parent topology notebook is a genuine cross-ensemble comparison, not a combined fit. Do not restore a parent-level combined three-ensemble analysis.
- Regenerate from maintained non-ignored sources, not old ignored PDFs with matching names. Code cleanup must preserve data, normalization, fit settings, and plotting results and be verified by execution.
- Derive errors on sums, differences, and products from aligned samples, not separately quoted errors.
- Verify cache provenance. Some scans are keyed by label, not requested bounds; new windows require explicit computation. Changed data or GEVP weights require valid input-dependent cache identification.
- Compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Check undefined references, missing citations, overfull boxes, and float warnings. Inspect affected pages rather than dismissing existing warnings.
- Render changed equations, captions, figures, and section transitions. Check readability, overlap, and placement at final paper size.
- Verify live source/figure hashes and the compiled PDF against the reviewed draft. Check `git status` before and after and preserve unrelated changes.

## Current Structure and Scope

- Title: "Excited-State Effects in Nucleon Sigma Terms from a Variational Analysis".
- Distinguish residual time dependence in a correlator from the choice of fitted energies. For the nucleon, discuss the small effect of projection first, then motivate retaining the standard two-state fit through compatible energies and no precision gain.

- Section II introduces the ensembles naturally, establishes B64 as the main-text focus, and briefly locates A24/A48 in Appendix C without enumerating its contents.
- Section III starts directly with III.A, Multi-State Fits. Do not restore the overview paragraph or the sentence that one-state fits illustrate a plateau.
- III.A covers the standard spectral decomposition, effective mass, and fits. Omit prose specifying correlated versus uncorrelated fits.
- III.B defines and discusses projected N first, then projected N-sigma. Introduce W after both channels, then discuss their overlap ratios. The eigenvector/W figure precedes the overlap figure, matching their first discussion; both remain in III.B.
- III.C develops the Laplace method and two-point comparisons. Displacement checks belong in Appendix B unless essential to selecting the central analysis.
- Section IV covers standard/reduced-GEVP light ratios, then Laplace filtering. Keep the schematic cancellation argument with the filtering observations that motivate it, not as a conclusion from flatness alone.
- Section V has a shared introduction, then separate strange/charm subsections. Strange retains full/reduced and standard/reduced figures; charm combines standard, reduced, and full ratios in one three-panel rainbow figure without midpoints.
- The abstract is concise and physics-focused. The Introduction proceeds from general and flavor-specific importance to light phenomenology, lattice results, the sigma-term puzzle and proposed resolutions, then strange/charm status, and finally this work. Describe recent proposed resolutions without claiming the discrepancy is settled. The Conclusions use three short flavor-specific items, not a two-point summary or a list of technical checks. Keep acknowledgments empty.
- Distinguish the light-current diagnostic (no improved final sigma_piN extraction), the single-spacing strange result, and the lack of a charm precision gain. Do not present the strange fit-window error as a continuum/finite-volume systematic budget.
- The literature sources and comparisons are maintained in analysis_sigma_literature_codex.ipynb and sigma_literature_codex.json. The strange comparison is cited in the Conclusions and included as a single-column manuscript figure; light and charm remain separate previews. Group by extraction method with consistent marker shapes, and use open/filled lattice symbols for finite-spacing/continuum results. Select recent informative determinations rather than superseded results or very noisy entries. Preserve excluded records, source versions, error components, asymmetric intervals, and mass conventions in the data file. Related analyses are not independent determinations, and no world average is formed.
- The strange literature figure has no legend or gridlines. Retain ETMC's same-ensemble B64 value 56(12) MeV alongside its continuum result, with open/filled symbols explained in the caption. Exclude the Kou-Chen photoproduction extractions from the figure because their theoretical uncertainties are not fully quantified; retain their source records and a brief qualified mention in the Introduction. They use the same data and their quoted spreads exclude model uncertainties; do not describe them as independent model-free experimental measurements.
- Literature comparisons show inner statistical and outer total error bars wherever the source separately quotes the statistical component. Do not infer a statistical error from an undifferentiated total. Include all source-quoted systematic/theoretical components in the total. Strange colors from top to bottom: this work red, other direct results green, BMW blue, Copeland orange. Group Copeland under Feynman--Hellmann (chiral fit to published baryon masses), with an open symbol because it has no continuum extrapolation. Its Table I gives sigma_s=50(6)(1)(10) MeV; do not omit the third theoretical error. Liang instead fits existing sigma-term data and remains a distinct chiral reanalysis in the light preview.
- Draw inner and outer literature error bars with identical line thickness, cap size, and cap thickness. Distinguish statistical and total uncertainty by bar length, not by heavier inner bars.
- Appendices: ChPT Pion-Mass Corrections; Laplace-Discretization Stability; Light-Current GEVP Analysis on A24 and A48; Approximating the Off-Diagonal Contractions.
- All appendix prose stays in the standard two-column flow. Use ordinary figure/table floats and figure* only for genuinely wide figures. Do not force placement with onecolumngrid/twocolumngrid switches, nested minipages, captionof, manual vertical padding, or column breaks. Appendix B discusses two-point, single-filter light, then strange-current checks; the double-filter fit comparison is removed.
- Appendix C is a compact light-current comparison with two combined A24/A48 figures: standard versus reduced GEVP, then reduced GEVP versus Laplace-filtered GEVP, each showing rainbows and midpoints. Retain these as separate figures with the published marker conventions; the combined three-column preview was not adopted. Do not restore the separate two-point, eigenvector, overlap, fit-scan, or energy-summary figures. The full ensemble analyses remain available in code. Discuss A24 as qualitative support for competing residual contributions and A48 as suggestive but less decisive, not as identifying a common extra excitation energy.
- Put Appendix C method/separation legends and ensemble labels inside the axes without covering data. A48 uses the same 0..80 MeV range as B64. Cite Barca's variational scalar studies for similar improvement, accurately identifying their heavier pion masses (429 and 222 MeV); the earlier study also supports the strange-current comparison in the main text.
- Do not repeat B64 in main-text result descriptions/captions. Keep ensemble names when distinguishing inputs, appendix results, or cross-ensemble comparisons.

## Notation and Source Style

- Define analysis-specific symbols once at first use. Prefer words when a redundant identity adds no information.
- Call the ratio a GEVP-improved ratio, not a projected ratio, throughout the manuscript including the abstract and conclusions. Projection remains appropriate for the mathematical operation and projected correlators/interpolators.
- When a method or observable returns after a substantial gap, refresh the reader with a reference to its defining equation or section. Prefer precise definition references over vague references to earlier discussion, but do not repeat them in adjacent sentences. Section V.A should explicitly connect the reduced and full GEVP ratios to their respective definitions.
- Keep source/equations compact. Short formulas should fit one row where legible; use `align` for actual alignment or related equations.
- Use `\cref` consistently, preserve labels, and use `sec:` for subsection labels.
- Avoid semicolons in prose/captions; mathematical argument lists may use them.
- Use parenthetical errors. Correction/pion-mass uncertainties generally have one significant digit, reported sigma-term uncertainties two. Preserve deliberately quoted source precision.
- Say "zero momentum." Suppress unnecessary momentum arguments and do not introduce U rotations. Finite-volume noninteracting levels are the explicit nonzero-momentum exception.
- Use `\mathcal J` for interpolators, `\mathcal O` for the scalar insertion, and J in topology labels. Do not introduce S as another current symbol.
- Use `m_\ell` for the degenerate light mass and `m_s,m_c` for strange/charm. Identify quoted lattice quark masses as bare, with superscripts `\rm sim` and `\rm iso`.
- Define N/sigma interpolators together, with consistent color indices and position/time arguments and suppressed Dirac indices. Keep the sigma bilinear `[ubar^a u^a + dbar^a d^a](x,t)` on one row. Do not put sigma-term names beside the scalar-operator formula as case labels.
- Use expectation-value notation for correlators, not internal strings such as NJN or NsgmJNsgm. After defining the matrix, use `(j,k)=(N\sigma,N\sigma)` for its additional entry.
- Use `N\sigma` for this basis and `Z_{k,n}` for overlaps. Projected overlaps have channel labels such as `Z_{\widetilde{N\sigma},n}`; never assign them to the bare interpolator.
- All projected two-point energies/gaps carry tildes, including Laplace-on-GEVP and downstream references. Standard N inputs remain untilded. Generic axes may cover both if captions distinguish them.
- Use `C_{\rm GEVP}^{\rm 2pt}`, `[v^{-1}]_{N0}`, `t_{\rm ins}^{\rm cut}`, and `\delta/a` for the projected denominator, inverse component, insertion cut, and displacement.
- Only the W-weighted reduced ratio is `R_{\rm GEVP}^{d}`. Full projection is `R_{\rm GEVP}`. Diagnostic variants are `R_{\rm GEVP}^{\prime}` and `R_{\rm GEVP}^{\prime\prime}`, without d. Attach both scripts directly to R; apostrophes before subscripts can shift the latter in Matplotlib.
- `R_{\rm LG}` filters the reduced W construction. Use explicit excitation indices: `\Delta E_1^{\rm 3pt}` and `\Delta E_1^{\rm 3pt,Lap}`. A prime does not encode the Laplace method.
- Use `N^*` for `N^*(1440)`, defined in its caption; do not call the Roper R.

## Simulation Details and Topologies

- Introduce `\texttt{cA211.53.24} (A24)`, `\texttt{cA2.09.48} (A48)`, and `\texttt{cB211.072.64} (B64)`. A24's repository directory `cA211.530.24` is not its manuscript spelling.
- Use published ensemble parameters. A24: a=0.09076(54) fm, m_pi=362(2) MeV; do not combine this spacing with the old 346 MeV mass. A48: a=0.0938(3) fm, L=4.50 fm, m_pi=130.6(4) MeV.
- Table I reports total available configurations A24/A48/B64 = 2467/1228/732, not internal resampling bins.
- Include Gaussian-smearing formulas and settings for all ensembles: B64 (N_G,alpha_G)=(125,0.2); A24/A48 (50,4.0), as in the N-pi paper. All use APE (n_APE,alpha_APE)=(50,0.5).
- Vacuum-subtract both the sigma interpolator and scalar current. State this at the end of Simulation Details and assume it thereafter.
- Strange/charm include the full N-sigma diagonal three-point entry; light does not. Its seven topologies each contain a separate heavy-current loop.
- Fig. 1 defines shared notation: source left/sink right; squares for N, circles for sigma; red squares identify the nucleon source and red lines point-source propagators. Hyphens separate quark-disconnected pieces. Cite the N-pi paper for computational details here.
- Fig. 2 only adds the crossed-circle current. Fig. 3 refers back. Do not add J-sigma or Sigma-contraction explanations to either caption.
- Distinguish `N-Jsigma_i/f` (current/meson connected) from `N-J-sigma_i` or `N-sigma_f-J` (separate loops). Use `Trf-J-sigma_i` with room below its label.
- Do not describe Trf reconstruction in manuscript prose; refer to the N-pi computation. Do not claim all nucleon correlators were reused from earlier papers, since additional two-point statistics are included.

### Statistics Tables

- Keep sampled point-source and fully projected statistics in separate one-column tables. Arrange source-position statistics vertically by ensemble with adjacent ts/a and N_src, and projected statistics horizontally by ensemble.
- Captions define header symbols without explaining that multiple entries correspond in order.
- N two-point sources: A24/A48/B64 = 121/272/369. B64 combines 349+4+16.
- Standard N three-point sources: A24 16 at ts/a=10,12,14; A48 9 at 10,12,14 and 32 at 16,18; B64 1,2,5,10,32,112,128 at 8,10,12,14,16,18,20.
- Label N_src, N_stoc, N_oet, N_def, and n_vec separately. B64 projected-table N_stoc is 25 for sigma and 12 for B/W. Preserve flavor-dependent inversion counts in the current source.
- Current rows are J (light), J (strange), J (charm), without "loop."
- Do not discuss full-statistics versus source-matched denominators in manuscript prose/captions. This does not remove the statistics tables.

## Two-Point Selections

Windows are in lattice units. These are current publication inputs, not independently hard-coded replacements for the notebooks.

| Ensemble | Standard N two-state lower bound | Projected N-sigma two-state lower bound | d=(t-t0)/a | Fitted later t/a | Adopted W |
| --- | --- | --- | --- | --- | --- |
| B64 | 8 | 4 | 2 | 9..14 | 0.00860(78), reconstructed |
| A24 | 5 | 3 | 4 | 11..14 | 0.0473(67), directly fitted |
| A48 | 5 | 3 | 2 | 8..12 | 0.0163(27), directly fitted |

- Plot eigenvectors against later t and stability against t_low, with t0=t-d. Do not relabel the earlier coordinate without shifting it. Both times must be sufficiently large; cite Luscher/Blossier. Avoid irrelevant normalization conventions.
- B64 component ratios: v0Ns/v0N=0.01563(99), v1N/v1Ns=-0.550(17). Direct W=0.00851(77) is a compatible check.
- B64 N one/two/three-state comparison windows are 20/8/3, with masses 944.6(6.3),944.0(4.4),944.7(5.7) MeV. Only the two-state result is adopted.
- B64 selected standard N: m_N=944.0(4.4) MeV, E1=1.475(59) GeV, gap=531(55) MeV. Projected N-sigma: E0=1.318(25), E1=2.30(10) GeV; external gap to N=374(26) MeV, internal gap=982(83) MeV.
- B64 overlap ratios: 0.803(27)/0.710(44) for standard N two/three states, 2.80(24) for projected N-sigma two states. The schematic product is 2.25(21), using the projected basis.
- A24 projected N-sigma E0/E1=1.945(29)/3.72(39) GeV, external gap=736(29) MeV. A48 E0/E1=1.256(32)/2.38(15) GeV, external gap=322(32) MeV.
- In the retained code diagnostics, the A24 standard N-sigma energy trends toward N: show it, but do not run/display unprojected one-/two-state fits. Both effective-energy series extend through available t/a=14. These diagnostics are no longer paper figures.
- GEVP display support: A24 t/a=5..14, A48 3..12. Approved scales can clip uninformative late errors; never change fit data or inflate errors to suggest convergence.
- Implementation only: ordinary one-state effective-mass comparisons are uncorrelated for both channels/all ensembles; other objectives remain as implemented. Do not announce these choices in prose or change them during editing.
- B64 N projection changes the mass by -1.4(2.0) MeV and slightly increases late errors; retain standard N. Projected N-sigma shifts by +17.5(2.3) MeV, consistent with removing a lower admixture. Selected-window errors are similar, not improved.
- N-sigma fitted energies are resolved scales, not necessarily the exact common-channel ground state or unique resonances. Eigenvector components alone do not quantify contamination in a normalized correlator.

## Laplace Construction

- Introduce filtering neutrally as an alternative with `Portelli:2025lop`, not as proven superior.
- Present continuous `-partial_t^2+E^2`, its action on the established spectrum, then the centered discrete `-nabla_delta^2+lambda^2(E,delta)`. The second difference includes 1/delta^2 and `lambda=sqrt(exp(E delta)+exp(-E delta)-2)/delta`. Preserve this dimensionful convention in all denominators.
- Filtering changes weights, not energies, and exactly removes a chosen level. Then identify the actual inputs: standard N and projected N-sigma.
- The two-point fit determines m and E1, one fewer parameter than the two-state effective-mass fit. Retain the explicit trial-dependent chi-square definition: transformed data and covariance both depend on E1. Do not call it maximum likelihood, add GMM terminology, or create a statistical appendix.
- Jackknife is assumed; do not introduce repeated jackknife explanations into the manuscript. Preserve resampling in code.
- Do not explain horizontal marker offsets, display-time shifts, or earliest-original-timeslice plotting conventions in manuscript prose or captions. Preserve the actual matched-support implementation in code.
- Explain briefly in III.C that smaller displacements give noisier but less-correlated filtered data and that the fitted energies have weak displacement dependence in the appendix study, then state the central delta/a=2 choice once. Keep numerical displacement settings and stability details in Appendix B, not repeated in main-text captions or result paragraphs. Appendix diagnostic captions may identify their displacement axes/legend.
- Define the positive-sign second-difference operator nabla_delta^2 directly, keeping the filter as -nabla_delta^2+lambda^2. Use E_n for the removed generic level. State explicitly that not determining overlaps leaves one fewer fit parameter.
- The B64 two-point displacement diagnostic retains fixed original endpoints N=7..28 and projected N-sigma=4..16. Compute and show delta/a=1,2,3,4 for both channels. Keep the previously tested wider scan only as an internal diagnostic, without discussing it in the paper. These scans do not replace the adopted central analysis.
- Central delta/a=2. B64 N Laplace gives m=945.6(4.4) MeV and gap=558(62) MeV; projected N-sigma gives 1.315(27),2.29(12) GeV. These are checks, not replacements for selected two-state inputs.
- Three-point filtering acts on insertion time at fixed ts, unlike the cited midpoint/ts example. Since the denominator is insertion-time independent, filtering numerator and denominator applies L/lambda^2 to the ratio.
- Displacement comparisons use matched original support. Do not fit the double-filtered light ratio or show its old appendix fit comparison.

## Light Ratios and Energy Interpretation

- Cases: I shares both gaps with the two-point input; II fits a common transition/diagonal gap; III fits the transition gap and fixes the diagonal gap to the two-point value; IV sets only r11=0 and retains the two-point denominator; V sets r1=r11=0. Use r01/r11, not ra01/ra11.
- Comparable diagonal matrix elements motivate testing III, but its diagonal contribution is unresolved. Adopt IV at (ts_low/a,tins_cut/a)=(12,2), about 0.95 fm: sigma_piN=45.9(3.0) MeV and Delta E1^3pt=405(35) MeV. Do not add a redundant sentence saying III is retained as an omitted-term check. Explain that unresolved residual contamination may reflect multiple transitions inadequately represented by one effective gap, not necessarily the diagonal matrix element; imposing a diagonal exponential may misdescribe it. An unresolved parameter does not by itself prove overfitting. Do not present the light sigma term as a final determination.
- I is lower, II/III nearly identical, IV near III, and V higher. GEVP-improved midpoints agree with the free-gap fits and are more precise before fitting, but retain time dependence. Motivate the absence of constant/two-state fits from the insertion-time/midpoint mismatch, then refer forward to the effective multi-state interpretation supported by filtering. Do not infer an exact state count from near-flatness alone.
- Full projection replaces the reduced numerator by sum Ijk; it does not merely add the missing diagonal to the W numerator. Compare with identical other inputs and aligned samples.
- Prime variants show small effects of W reweighting and the diagonal projected two-point denominator term. They do not prove a small diagonal three-point matrix element or exclude other states.
- Light Laplace window (10,4) at delta/a=2 matches unfiltered cut-2 support. It gives gap=360(25) MeV and sigma_piN=48.6(3.2) MeV.
- Compare IV and V for R_LG, retaining IV(10,4) for all downstream comparisons. Its standard two-point denominator inputs give sigma_piN=49.5(5.6) MeV and gap=850(190) MeV. Keep the gap as an adopted input; do not enumerate IV/V values merely to repeat their visual agreement. II/III are poorly constrained; do not conflate large errors and numerical failure.
- Explain matched original support concretely once and explicitly connect it to delta/a=2: each filtered value uses neighbors at t_ins +/- 2a, so filtered centers 4a..ts-4a reach the same original range as unfiltered insertions 2a..ts-2a.
- Filtering reveals stronger dependence in nearly flat R_GEVP^d. Restrict the response argument to ground-to-excited transitions, Delta E_n=E_n-m_N: `1-lambda^2(Delta E_n,delta)/lambda^2(Delta E,delta)`. With central gaps 850 and 360 MeV and the adopted discretization, the response is -4.76: sign reversal and almost fivefold amplification. Connect this quantitatively to the interpretation of at least two partially cancelling excited contributions, without claiming unique state assignments. Do not fit the unfiltered light reduced ratio with one excitation.
- The overlap model uses {J_N,tilde J_Nsigma} and effective X/Y states. Its inherited squared overlap 2.25(21) belongs schematically to Y at tilde E1^Nsigma=2.30(10) GeV, not the fitted E1^LG=1.79(19) GeV. Explain the mismatch by referring to the two-/three-point spectral weights and current enhancement of unresolved levels; the schematic overlap is not a quantitative prediction of the R_LG excitation.
- Close the light analysis by distinguishing precision of the nearly flat reduced ratio from precision of the extraction. GEVP reduces visible contamination without degrading ratio precision, but residual and inherited higher contributions prevent a simple extraction. R_LG IV/V fits are noisier, so this 2x2 basis provides a cross-check, not an improved light sigma-term determination.
- E1^3pt=m_N+Delta E1^3pt, E1^Lap=m_N+Delta E1^3pt,Lap, E1^LG=m_N+Delta E1^LG. B64 values: 1.349(34),1.304(26),1.79(19) GeV; the last now uses R_LG IV. Keep these in the figure, not repeated as a list in the discussion.
- The upper two-point-only comparison is `m_N+Delta E1^2pt+Delta tilde E0^Nsigma=1.849(59)` GeV. Keep it distinct from the double-filter target `Delta E_Sigma=Delta E1^2pt+Delta E1^3pt,Lap`.
- External `Delta tilde E0^Nsigma=tilde E0^Nsigma-m_N` differs from internal `Delta tilde E1^Nsigma=tilde E1^Nsigma-tilde E0^Nsigma`.
- Noninteracting levels use each ensemble's masses/volume, with momentum index k denoting squared momentum k(2pi/L)^2. Physical N*,N-sigma,N*-sigma ranges are orientation with half-width bars, not statistical errors. Omit them on A24.
- Double-filter delta1/a=2,delta2/a=3: only ratios/midpoints, no fits/selections in that figure. Keep the discussion brief: too noisy to conclude anything about remaining contamination. Do not discuss insertion cuts as fit settings in this comparison.
- Retained A24/A48 analysis code uses the same I--V standard-ratio and IV/V filtered-ratio comparisons as B64, with the same shared plotting styles and standard two-point inputs in case IV. Those fit scans and energy comparisons are not displayed in Appendix C. Do not silently substitute V for an unresolved IV fit. I--III require at least two separations; identifiable IV/V/Laplace fits may use one.
- A48 R_LG windows 12/18 fail numerically; 14 approaches zero gap with unresolved sigma. Omit all three, but do not call all failures. Remaining gaps are also uncertain.

## Strange and Charm Results

- Integer separations 2..30 are available; publication uses through 22. Rainbows show even ts; midpoints and strange fits include odd/even. Odd midpoints average the two central insertions.
- Full/reduced ratios agree and full is noisier; retain R_GEVP^d for strange extraction.
- Standard strange comparisons are I,IV,VI. VI is III with transition gap fixed to the selected light-IV Delta E1^3pt, not the Laplace gap; denominator/diagonal inputs remain two-point quantities. At lower 10, sigma_s values are 37.3(4.4),40.8(7.1),45.3(5.4) MeV. Read the saved standard_ratio_selected object and key the VI cache by the external samples.
- Strange II/III and free-gap Laplace fits are imprecise/window-sensitive, not generically nonconvergent. Laplace errors are about 50 percent. Filtering the reduced ratio is noisy even with the light filter energy; failure to resolve light-like enhancement does not prove absent excited states.
- Adopt the AIC average of six constant windows: lower 8,10,12,14,16,18; upper 22; cut 1. Result `sigma_s=43.9(2.6)` MeV, not an individual selected fit.
- The notebook uses yu.modelAvg(..., fullOutputQ=True) with central minimized chi2, weights proportional to exp(-chi2_i/2+nu_i), nu=Ndata-Npar, and mixture variance. The window-spread contribution is 0.01695 MeV; this is not a continuum/finite-volume error budget. In the paper omit the detailed AIC formulas and cite Jay:2020jkz and Neil:2022joj, the same references as the N-pi and EMT papers. Fig.15 caption simply calls the orange band the constant-fit model average.
- Orange average band has no center line or selection/highest-weight star; ordinary constant points are filled diamonds.
- Brown summation fits are a separate check, not averaged in. Sum all noncontact insertions and fit a line over odd/even windows through 22. Lower 12a (about 0.95 fm) gives 44.3(9.4) MeV.
- Charm: full/reduced agree, both much noisier than standard. No charm fit or quoted sigma_c. Track a future standard-analysis citation in a source comment, not a fabricated bibliography entry.
- Keep the separate 725-configuration Srijit diagnostic distinct from matrix-data analysis. Do not pair their samples or import diagnostic results into the paper.

## Pion-Mass Corrections and Appendix A

- II.A motivates leading replacement via sigma_piN=m_l*g_S, constant g_S at leading order, and m_pi squared proportional to m_l. It requires no ChPT LEC input. Higher-order changes are estimated below 1 MeV for A48/B64 (central additive p2-to-p4 changes about 0.6/0.8 MeV), not strictly bounded.
- Leading replacement and additive/multiplicative ChPT corrections are alternatives, never successive corrections.
- B64 simulated bare masses: (0.00072,0.0186,0.249); isoQCD: (0.0006669(28),0.018267(53),0.23134(52)). A48 leading matching to 135.0 MeV gives 0.000962(6) from 0.0009. Leave A24 at 0.0053.
- II.A includes phenomenology 59.0(3.5) MeV; published downward p2/p3/p4 shifts 3.7(1),3.2(1),3.1(1)(4) MeV. Adopt the p4 correction with the same two-error notation throughout, giving 55.9(3.5) MeV, citing `Hoferichter:2023ptl`. Use `sigma_piN^{pi^pm}` for the charged-pion convention.
- Appendix A begins with the ChPT/Roy-Steiner references followed. Keep formulas, order-specific LECs, correlation inputs, and correction table here. Explain direct p2/p3 evaluations before calibrated p4.
- Quote gA=1.27641(56) and Fpi=92.215(143) MeV simply as values quoted by PDG, without PERKEO. Do not change the gA input to a PDG average or display an intermediate sqrt(2) conversion for Fpi.
- Give lbar3=3.41(41) with citation, without a digression. Use m_N^(0)=869.6 MeV and calibrate e1=12(6) GeV^-3 from sigma_piN(139.57 MeV)=59.0(3.5) MeV. Retain c1/c2/c3 correlations and state that other joint input correlations are unavailable.
- Keep `eq:sigma-chpt-B` on one row if it fits. Use the published B64 Mi=140.2(2) MeV (`ExtendedTwistedMass:2022jpw`) in both Table I and Appendix A; A48 Mi=130.6(4) MeV. Do not add a conversion formula or discussion of agreement with a mass-ratio reconstruction or linear approximation.
- Define sigma_add=sigma_sim+Delta sigma and sigma_mult=sigma_sim*r before defining the ChPT difference/ratio. State generally that p2 r=(Mf/Mi)^2 reproduces the leading prescription, without ensemble-specific elaboration.
- Table target is 135.0 MeV. A48 shifts p2/p3/p4: +3.5(3),+3.0(5),+2.9(7) MeV; factors 1.069(7),1.056(9),1.05(1). B64 shifts: -4.2(2),-3.6(3),-3.5(7) MeV; factors 0.927(3),0.941(4),0.942(7). Recomputed with the existing OneDrive Mathematica definitions and uncertainty propagation, changing only the B64 pion-mass input; the user's notebook and CSV remain untouched.
- Keep the third phenomenology table block. Its p4 shift to 135 MeV is -3.0(6) MeV, distinct from the published neutral-pion-target -3.1(1)(4) MeV. In comparison, quote published order-by-order errors: first from LECs, second at p4 from the phenomenological sigma input. Do not conflate targets.

## Plotting Rules and Exceptions

Use shared renderers in `project2/07_Nsgm/util_codex.py` with notebook-supplied ensemble settings. Consult the coding guide for implementation; do not duplicate plotting code.

### Shared Appearance

- Match light axes/ticks/error bars of the N-pi paper. Vertical error stems and horizontal caps have equal weight.
- Use consistent nonzero panel spacing; shared-y plots retain inward y ticks on every panel, though numerical y labels may appear only at left.
- Comparable ratio/stability panels use consistent x limits/ticks with margins beyond outer points. Shorter filtered support alone does not justify different axes; genuinely extended separation sets can.
- Match y ranges for comparisons, except explicit scale/clipping exceptions below. Never silently remove fitted data for appearance.
- Reference lines are grey dashed without endpoint markers. Prevent default markers leaking onto band/line endpoints.
- Keep legends clear of points, error bars, and inward ticks. Shorten labels and move details to captions; no fitted numerical values in legends.
- Preserve the shared rainbow color/marker map. Display both mirrored halves of symmetrized ratios; fit each independent insertion once.
- Direct comparisons use opaque white-faced open baseline markers and slightly shifted filled transformed markers. Representative ratio legends use black circles via `ratio_legend_handles`; fit legends match plotted markers.
- Use ordinary stars only where specified, drawn once with their error bar, not overlaid on another marker. Keep sizes comparable to neighboring points.
- Do not mention two- or three-point fit upper bounds in manuscript prose or captions unless the user explicitly requests them. Keep exact ranges in code. Captions should not enumerate displayed intervals or selected lower bounds either; describe the relevant selection markers.
- Retained displacement scans use open stars for delta/a=2 in each two-point category and for the single light filter. There is no double-filter fit selection.
- Retained code-only A24 N-sigma effective-energy diagnostics use a tighter range, allowing the very noisy late projected points to extend beyond the axis. The code-only energy comparison removes the missing PDG rows and their vertical gaps, rather than only hiding the black markers.
- Choose column width for readability at final paper size, not just space saving. The Appendix C comparisons are genuine page-width two-row figures. The off-diagonal midpoint-difference comparison is a genuine single-column figure with a compact two-panel canvas and a shared legend above the panels.

### Figure Families

- Two-point energies: open standard, filled projected; red squares/green circles/blue diamonds for one/two/three states. Selected two-state fit gets a dashed vertical line at the shifted datum coordinate. No selection star or special fill.
- Laplace N: open orange triangles, "Laplace on standard"; N-sigma: filled orange triangles, "Laplace on GEVP". Bands identify selected two-state energies, not Laplace replacements.
- Overlaps: ordinary points filled; selected two-state results white-faced open stars. Omit projected N-sigma points with absolute overlap uncertainty greater than two, explaining this at the first caption.
- Eigenvectors: rows v0Ns/v0N,v1N/v1Ns,W; left data, right lower-bound scans. Explain direct/reconstructed bands and selections at first appearance. Do not generalize B64's reconstructed-W choice to A24/A48.
- Light standard/reduced ratios: four-panel layout, labels R_std I/II/III/IV/V, case-IV purple down-triangles and case-V orange diamonds. Red/gray gap bands denote standard N gap and projected N-sigma external gap.
- Light Laplace summary: left Rstd/RLap; middle standard V, Laplace, R_LG IV and V fits; right R_GEVP^d/R_LG. No selection stars. Put outer legends near the top when clear. Do not restore the redundant separate R_GEVP^d/R_LG fit figure. A-ensemble summaries also show both IV and V.
- Energy colors: red E1^2pt; green E1^3pt,E1^Lap,tilde E0^Nsigma; blue tilde E1^Nsigma,two-point-only sum,E1^LG. The E1^2pt/N* group is at the bottom; each black reference is at the bottom of its own group, not all together.
- Full/reduced strange comparisons split legends between outer ratio panels and leave midpoint panels clear. Strange overlaid standard/reduced uses a combined legend. Charm places its method legends inside each of the three ratio panels.
- Strange uses 0..80 MeV, matching light. Approved clipping of late points does not remove them from the scan. Use a compact three-column fit legend: Rstd I/IV/VI, constant, summation.
- The combined charm comparison uses a common +/-2500 MeV scale in all three panels. Do not propagate that scale elsewhere.
- Double-filter light has ratio/midpoint panels only, with no fits or displacement-fit scan. Single-filter displacement scans are separate appendix diagnostics.

## Off-Diagonal Approximation Appendix

- Use the two-panel, three-ensemble midpoint-difference figure in physical units with distinct symbols/offsets and no headings above panels.
- Left: complete reduced ratio minus omission of B/W/Z from off-diagonals. Right: complete minus retaining only N-Jsigma_i and N-J-sigma_i with sink counterparts in off-diagonals.
- Keep standard N three-point, denominator, eigenvectors, and W fixed. Distinguish current-connected and separate-loop direct topologies.
- A24 shifts are 12..15 MeV, or 6..8 percent of its 185..190 MeV midpoint signal. A48/B64 are generally a few MeV or less, noisier at late separations.
- Explain relative and absolute size. These are midpoint tests, not a uniform approximation at all insertion times or a new asymptotic sigma-term extraction.

## Maintaining This Guide

- Keep the 15 general rules separate from project selections and numerical reference values.
- When a choice changes, update its owning section and check other mentions/examples. Do not leave a second competing instruction elsewhere.
- Prefer labels and source paths to fragile equation/figure numbers. Check retained numbers against the current PDF.
- Remove superseded selections from active guidance. Diagnostic history belongs in existing analysis records.
- If code, guide, and manuscript disagree and the latest decision does not resolve it, identify the discrepancy and ask rather than inventing a resolution.
- A guide-only task does not require changing, regenerating, or publishing the paper.


- The drafting reference for Barca et al.'s preliminary study at m_pi=222 MeV is arXiv:2609.05989 (LATTICE2026); distinguish it from the 429 MeV study and our physical-point analysis. At finalization, check for a full publication or preprint supporting this result and replace the proceedings citation when available, following Citation Finalization above.
- Fig.16 is a true single-column figure environment, not a narrow image inside a page-width figure*. It has two panels: overlaid R_GEVP^d/R_LG rainbows on the left and both midpoint series on the right. Use the sample-aligned light Laplace energy360(25) MeV and delta3: originalcut2 becomes filteredcut5. Unfiltered midpoints cover8..22 and filtered midpoints10..22, including odd/even; rainbows show even separations only. Filtering is too noisy to constrain residual strange excited states; do not use it as support for the constant fit or evidence that higher states are absent. The extended flat unfiltered data and stable window dependence support the constant fit. The delta2 and delta6 displays are historical diagnostics, not the publication choice.


- Explain the small R_GEVP prime/double-prime changes as consistent with leading diagonal contamination from the standard two-point excitation rather than the N-sigma channel, not proof of this state assignment. Introduce Appendix D in its own paragraph explaining the cost-saving contractions tested and the midpoint comparison used to assess their accuracy.
