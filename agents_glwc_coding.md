# AI Agent Guide: N-sigma Analysis Code

Current operational rules. This is not a chronological change log. Read agents_paper_writing.md for manuscript conventions. The pre-consolidation guide is retained in project2/07_Nsgm/__codex_ignore/processing_audit_20260916/agents_glwc_coding_before.md for historical retrieval only.

## Precedence and Required Reporting

- Follow the latest explicit user decision. Historical cached outputs and older instructions do not authorize reinstating a removed fit, figure, or selection.
- Before changing a figure, check whether its analysis is still retained. In particular, the light double-filtered ratio is a data-only comparison: NO constant fits, no selected fit, and no displacement-fit figure. The former Fig. 21 was removed for this reason.
- After changes, explicitly report contradictions discovered, results affected by bug fixes, unresolvable fits, and consequential figure renumbering. Do not silently follow a local formatting request that conflicts with an earlier scientific decision.
- Change current rules in place. Do not append a second, contradictory set of selections or figure counts.
- Review actual code, data, and current selections, not filenames alone. A review cannot establish absence of all bugs; describe exactly what was checked and what remains unverified.

## Ownership and Paths

- Independently editable code must have _codex in its filename. Obtain authorization before changing util.py, util_Nsgm.py, processData.ipynb, preparation code, or other user-controlled files. Previous approval is not standing permission.
- Preserve unrelated Git changes. Check git status before/after. Never revert user edits or delete input/_wrong data.
- Repository: C:/Users/yan14/Documents/ChatGPT/Nsgm/glwc3.
- Project: project2/07_Nsgm. Ensemble directories: cA211.530.24 (a24), cA2.09.48 (a), cB211.072.64 (b).
- Keep maintained notebooks in their ensemble directory. Only genuine cross-ensemble comparisons belong at project level. Do not restore analysis_3pt_light_common_codex.py or a combined ensemble fit.
- All generated N-sigma outputs and internal diagnostics belong under project2/07_Nsgm/__codex_ignore. Never create repository-root fig, pkl, tmp, .ipython, or .jupyter directories.
- Normal notebook outputs: __codex_ignore/<ensemble>/{fig,pkl}/<notebook>/{internal_ignore,reg_ignore}. Original pkl/processData inputs stay in their ensemble directory.
- Launch from the ensemble directory, record its resolved HERE, then change to the ignored output directory BEFORE yu.setpath. Use absolute input paths.
- Maintained deliverables are publication sources and user-requested files. Internal scripts, audits, reports, and previews stay ignored; do not add one-off ignore rules.
- Topology TeX sources stay in OneDrive/coding/latex/papers/diagrams_Nsgm. Never create duplicate maintained diagram sources under glwc3.
- The resolved xcheck_Srijit.ipynb and reproduce_strange_typeII_fit_codex.ipynb notebooks have been removed at the user's request. Do not restore them.

## Maintained Pipeline

1. Ensemble processData.ipynb files own the existing production inputs. Read them as specifications; do not modify them without approval.
2. Each ensemble's processData_codex.ipynb creates the additional compact Appendix D input from its local HDF5 export.
3. analysis_2pt_codex.ipynb computes the current two-point selections and paired eigenvector weights.
4. analysis_3pt_light_codex.ipynb reads those selections and computes light-current results. A notebooks must not independently refit the selected nucleon input.
5. B64 analysis_3pt_strange_charm_codex.ipynb reads the two-point and selected light outputs.
6. Parent analysis_3pt_topologies_codex.ipynb uses compact pickles and current eigenvector pickles only.
7. reproduce_figures_codex.py owns the exact manuscript inventory and runs eight analysis notebooks in dependency order. Processing notebooks are a separate, one-time raw-data stage, not rerun when making figures. The inventory currently has 35 figures; verify the map against the current manuscript rather than hard-coding this count elsewhere.

## Code Style and Execution

- Read the original analysis and relevant utilities before implementing. Import util as yu from the project directory.
- Use existing jackknife, fitting, plotting, and serialization helpers. Prefer suitable doFits_* methods. Use jackfit only where the high-level helpers cannot express the operation.
- util_codex.py owns shared processing, analysis safeguards, and publication renderers, including the rest-frame scalar contractions used by all three processing notebooks. Keep these helpers together; do not create a separate processing utility or copy their bodies into ensemble notebooks.
- Keep notebooks real .ipynb files organized by physics purpose, with short Markdown explanations and the analysis sequence before plotting. Avoid repeated one-line wrappers, semicolon-heavy code, and excessive argument-per-line formatting.
- Keep independent ensembles' settings explicit, but use the same renderer and model definitions. Do not introduce unnecessary abstractions or standalone restyling scripts.
- Use .venv; do not create another environment. Redirect Jupyter/IPython runtime state under the ignored run folder.
- Execute edited notebooks fully. Do not rerun discarded A-ensemble matrix-element, ratio-variant, double-filter, or displacement diagnostics during publication reproduction. Historical caches remain diagnostic only.
- Bound exploratory fit work; preserve warning counts and avoid automatic retries of known failed windows. Numerical convergence is distinct from a poorly constrained result.
- Replay captured warnings only after leaving the capture context. Replaying inside it can append indefinitely.
- Never use pseudo jackknife samples for production fit uncertainties or silently alter covariance treatment, masks, priors, or model parameters.

## Compact Appendix D Data

- Files:
  - A24: Desktop/NST_f_cA211.53.24_Nsgm.h5.
  - A48: Desktop/NST_f_cA2.09.48_Nsgm_tf=10-18.h5.
  - B64: Desktop/Nsgm_4src_16,18,20_cB211.072.64.h5.
- Outputs: __codex_ignore/<ensemble>/pkl/processData_codex/reg_ignore/data_topologies.pkl. These are retained analysis INPUTS, not disposable fit caches. The user may remove the Desktop exports after validation; do not remove those exports yourself.
- Each compact pickle contains full c2/c3 matrices, matched c2 denominators, separate BWZ/direct matrices, operators, configuration IDs/order, used/discarded configurations, block size, separations, and raw/reference/processor hashes. It contains no fitted eigenvector weights.
- A24/A48 use production data.pkl; B64 uses data_cd.pkl, including its saved connected nucleon contribution. Full reconstructed matrices must match the production pickle sample by sample before saving.
- Read only rest-frame N/N-sigma row-diagonal scalar j+ data, including both l1/l2. Preserve original raw precision, blocking, subtraction, and summation order for exact production reproduction. Independent higher-precision raw-statistics tests are different analyses.
- Reconstruct reverse two-point terms using swapped operators: get2pt_diag(opb, opa, diag). Three-point partners also conjugate and reflect insertion time. Apply vacuum inclusion-exclusion and row/Hermitian averaging exactly once.
- BWZ means ONLY B3pt/W3pt/Z3pt, not all loopless-current contractions.
- Direct means N-jPi/N-jPf (connected current-sigma loop) plus N-j-pi0i/N-pi0f-j (separate current and sigma loops). NJN-pi0i/f and T-j with its reverse partner are not in this approximation.
- Both differences change ONLY off-diagonal three-point entries. Keep NJN, denominators, weights, samples, and symmetrization fixed. Compute paired differences before errors.
- Validate missing diagram sets, dimensions, finite data, reconstruction, and production hashes. Do not execute definitions extracted from a fixed cell number during maintained processing/analysis.
- The September 15 audit compared 72/120/168 individual three-point diagram matrices on A24/A48/B64, plus two-point diagrams and full production matrices. Reproduction with h5py.File blocked gave exactly identical midpoint means/errors.
- Current compact sizes are about 4.73/8.92/14.84 MiB; they replace raw inputs of about 8.9/1.4/2.6 GB for this purpose. They do not replace the raw data for other channels or alternative raw resampling.

## Data Contracts and Statistics

- Production jackknife counts: A24 2467 raw configurations, block 4, 616 samples, three unused trailing configurations; A48 1228/block2/614; B64 732/block1/732. Preserve original order; never silently trim to another configuration set.
- These are leave-one-block-out samples, not raw measurements. Never jackknife an already jackknifed denominator.
- A24/A48 matched denominators are real(c2[:,ts,0,0]); B64 light connected analyses use the saved source-matched denominator.
- For disconnected strange/charm use the two-point matrix denominator, not the light connected source-matched denominator. Do not discuss this statistics detail in manuscript text/captions.
- B64 data_jsc.pkl layout: [c2, light_c3, matched_c2, strange_c3, charm_c3]. data_jsc_NsgmJNsgm.pkl layout: [c2, strange_c3, charm_c3]. Both are 732-sample matrix inputs, not the independent 725-configuration standard-only cross-check.
- The full strange/charm input has every integer separation 2..30. Its entire c2 and old three c3 entries must exactly equal data_jsc.pkl; only the lower-right c3 entry is new.
- Full scalar lower-right topologies include B2pt-j, W2pt-j, Z2pt-j, N-P-j, N-pi0f-j-pi0i, T-pi0f-j, and the reflected/conjugated partner. The user confirms generation of all preparation-list diagrams; remote raw strange/charm completeness is an assumption backed by source/synthetic checks, not a local end-to-end raw check.
- Full projection replaces the reduced numerator with C00+v(C01+C10)+v^2*C11. It does not merely add v^2*C11 to the W-weighted numerator.
- The reduced numerator is (1-W^2)C00+v(1+W)(C01+C10), with projected c2 in the denominator.
- Symmetrize before symmetric fitting, retaining each reflected pair once: insertion cut..floor(ts/2). Even midpoints are single points; odd midpoints average the central pair. Sum fits retain both members with their proper weights.
- Use aligned sample arithmetic for differences, sums, products, and external fitted parameters. Matching sample counts alone does not prove matching configurations.
- Meson/baryon raw configuration ordering must be verified before sample-wise combinations. Existing meson-file count checks alone are insufficient.
- Use yu.jackme/jackmec/jackme_un2str. Parenthetical uncertainty formatting is standard.
- B64 central isosymmetric bare masses: light .0006669, strange .018267, charm .23134; spacing .07948 fm. A48 scalar conversion uses central factor (135/ens2mpi['a'])^2; A24 retains simulated mass. Display factors do not automatically propagate external mass uncertainties.
- Use published ensemble masses/spacings, not auxiliary meson fits. A24 a=.09076 fm, mpi=361.6 MeV; A48 a=.0938 fm, mpi=130.6 MeV. Never mix A24's newer spacing with the older 346 MeV mass.
- Meson effective energies use the periodic cosh-ratio helper c2pt2meff_pion. Auxiliary meson fits are not publication mass inputs.

## Cache Safety

- util decorators often cache by label ONLY. A matching filename is not proof of matching data, range, or model.
- fit_meff_comparison now keys multistate caches by actual effective-mass samples, ranges, starting values, and covariance choice. A legacy cache is migrated only after reproducing all its stored objectives with the current inputs and exact fit windows. One-state comparison caches already include actual samples.
- All publication notebooks call guard_fit_cache before legacy cached fits. The guard records reviewed numerical inputs and rejects later changes. Do not delete/reset it merely to make a changed-input run succeed; first rekey or regenerate every affected cache.
- The audit found 13 stale A48 projected-nucleon two-/three-state fits after the eigenvector update. They were recomputed. Standard N remains the adopted nucleon input, so this bug affected comparison fits, not the adopted mass.
- Before a changed-input analysis, check downstream dependency hashes and extend missing ranges explicitly. Preserve old caches for diagnosis; never silently replace a failed current fit with an old result.
- RLG case-IV cache labels hash ratio and external two-point samples. Historical labels containing IV sometimes mean the old case V; identify the functional form, not its label.
- Existing chi2_jk entries are residuals against CENTRAL data, evaluated at each fitted sample parameter. They are neither minimized central chi2 nor each sample's own residual. Do not misdescribe notebook mean-chi2 p-values as exact central-fit goodness of fit.
- Use actual central minimized chi2 for AIC weights. Do not migrate cached fits on the basis of rounded results alone.

## Two-Point Selections

- All ordinary one-state effective-mass comparison fits are uncorrelated, using fit_meff_comparison. Keep multistate and Laplace covariance choices unchanged. No one-/three-state selection.
- B64 adopts standard N two-state lower8 and projected N-sigma lower4. SELECTED_METHOD and FIT_SELECTION own these choices.
- B64 eigenvectors use later t with t-t0=2a. Component fits are uncorrelated; selected indices7..12 correspond to later t/a=9..14 inclusive. W used downstream is reconstructed as -v*s.
- A24 uses t-t0=4a and later t/a=11..14. A48 uses 2a and later8..12. Their component and direct-W plateau fits are correlated. Preserve this approved difference from B64.
- A24/A48 adopt standard N lower5 and projected N-sigma lower3. two_point_references.pkl owns paired energies, weights, and GEVP metadata. Downstream notebooks read it.
- A24 has NO unprojected N-sigma fits. Display standard/projected effective energies through available t/a=14, with ylim1..2.9 GeV; noisy late points may extend outside.
- A48 eigenvector display stops before t/a=13,14. Plot all eigenvector scans versus later t, never relabel t0 as t without adjusting the pair.
- B64 standard_two_state_selected.pkl holds unprojected fits for standard-N inputs; two_state_selected.pkl holds the adopted mixed selection. Projected N-sigma overlaps and energies must use the latter.
- The nucleon three-state fit may be shown when resolved. Do not run the unused B64 N-sigma three-state scan.
- Use the current precision-based upper-time criterion; preserve support unless the user requests a change. A plateau requires stability and usable errors, not just a handful of visually convenient points.

## Light Ratios and Laplace Filters

- Standard cases:
  I = 2st2step_SYMshare.
  II = 2st2step_SYM.
  III = 2st2step_SYM_share11.
  IV = 2st2step_SYM_0ra11, retaining the two-point denominator.
  V = 2st2step_SYM_0rc1_0ra11, also dropping denominator contamination.
- B64 selected standard IV is (lower,cut)=(12,2), sigma_piN about45.9(3.0) MeV, gap405(35) MeV. Save standard_ratio_selected with case, window, fit, and paired gap; strange VI reads it.
- Standard light reduced R_GEVP^d is not fitted with a constant or single-excited-state model: filtering indicates competing excited contributions.
- B64 RLG summary shows IV and V; the retained RLG energy uses IV(10,4), gap about850(190) MeV. It is not the light Laplace gap used to construct the filter.
- A24/A48 use all standard cases I--V and RLG IV/V. Standard-IV energy references use lower12/14 respectively; RLG-IV energy references use (10,4). Both use the selected standard two-point denominator.
- A48 RLG IV/V at lower12/14/18 are unresolved and omitted. Preserve warning counts and distinguish poor parameter resolution from nonconvergence. Surviving gaps do not establish a stable precise excitation.
- Ordinary single-filter comparison: original insertion cut2, displacement2, filtered cut4. Equality of ORIGINAL input support is what matters.
- Two-point displacement study holds original inclusive endpoints7..28 (N) and4..16 (projected N-sigma), accounting for the adjacent-time mass. For displacement d use trimmed effective-mass indices range(first,last-2*d), with at least three masses. Show only d=1..4. Fixed endpoints do not imply equal numbers of filtered points.
- Double-filtered light ratios are displayed with (delta1,delta2)=(2,3), original cut1 and filtered cut6. Do not fit them or restore the removed displacement-fit plot.
- A-ensemble removed diagnostics are no longer executed by maintained publication notebooks. Their retained figure types are eigenvectors/W, N two-point, N-sigma two-point, overlaps, standard/reduced light ratio, Laplace summary, energy comparison.
- A24 energy comparison omits physical-resonance markers AND their rows/spacing. A48/B64 retain the PDG-based references.

## Strange and Charm

- Publication data use matched 732-sample matrices, all integer separations8..22; rainbows show even separations, midpoints/fits both odd and even.
- Strange standard plots show I, IV, VI. VI has the case-III functional form with its transition gap fixed to selected light IV, with paired external uncertainties.
- Cases II/III and free-gap Laplace fits are noisy/window-sensitive, not automatically numerically nonconvergent.
- Strange reduced-ratio extraction uses AIC averaging of six constant windows (lower8,10,12,14,16,18, upper22, cut1). No selected/highest-weight star. Orange band has total error and no center line.
- yu.modelAvg uses central minimized chi2 with exp(-chi2/2+Ndof) weights. Current result43.88282498 MeV, statistical mixture error2.64229267, window spread.01694749 MeV, hence43.9(2.6). Summation with only contact points removed is a separate check, not part of the average.
- jackMA with systematicQ=False averages paired samples; systematicQ=True calls modelAvg and returns pseudo samples. It is not equivalent to per-bin weight propagation. Historical AIC audit is under selection_aic_20260915.
- Strange filtered midpoint figure uses paired light Laplace gap from RLap_delta2(10,4), displacement3, originalcut2/filteredcut5. It is a noisy diagnostic; extended flat unfiltered data motivate the constant fit.
- Full and W-weighted strange/charm ratios are compatible; full is noisier. Charm W is also noisier than standard. No charm fits or quoted sigma_c result.
- Use light-like0..80 MeV strange axes. User permits late standard-point clipping. Charm plots may use their larger plot-specific limits.
- Never pair independent725-configuration data with732-sample light/matrix results. If testing external gaps on725, use the central input explicitly and treat its uncertainty independently.

## Shared Plot Conventions

- Use util_codex shared renderers and paper_style, plus existing yu plotting helpers. Do not duplicate renderer bodies in notebooks.
- Use the same models, colors, markers, labels, and meaningful panels across ensembles. Preserve genuinely ensemble-specific ranges and statistical choices.
- Standard fit styles: I red circle, II green up-triangle, III blue square, IV purple down-triangle, V orange diamond. Reduced filtered IV/V use the established shared-summary styles. Strange summation is a brown left-triangle.
- Only the W construction carries d: R_GEVP^d. Full is R_GEVP. Variants are R_GEVP^prime and R_GEVP^double-prime, with scripts attached directly to R. All projected two-point energies/gaps carry tildes.
- Standard/transformed representative legends use open/filled black circles. Separation data retain yu.colors16/fmts16 with common tfs_reference across subsets. Match legend labels/markers across plot families.
- Two-point selected two-state fits use dashed vertical lines aligned with the actual shifted selected marker. No stars for two-point energy selection.
- Other actual selections use one ordinary open star with opaque white face, replacing rather than overlaying the ordinary marker. No selection marks where no fit is adopted; strange AIC has no selection.
- B64 displacement plots mark delta2 with open stars. There is no double-filter displacement-fit selection.
- Use shared-y finish_shared_y_panels; retain inward ticks on both spines and hide repeated labels only. Reference lines are dashed, without accidental endpoint markers.
- Usual widths are7.1in for full page and3.4--3.55in for column figures. Keep line widths near.8pt axes and1pt errorbars, with visible caps.
- Never let legends cover data/error bars/ticks. Check rendered paper-size figures. Keep fit/display support separate; do not discard data from a fit to improve visual appearance.
- Numerical implementation ranges and horizontal display offsets belong in code, not captions. Follow paper-writing guide for what enters prose.
- Split strange/charm method legends between left/right panels for the three-panel comparisons. Keep the combined legend for the overlaid strange standard/W panel.
- No headers on Appendix D panels. Show paired midpoint differences in physical units with distinct ensemble symbols and small offsets.

## Cross-Checks Outside the Publication Pipeline

- The Srijit comparison is resolved and its two maintained cross-check notebooks are removed. Existing ignored diagnostic outputs are historical records, not publication inputs.
- His later clarified model is current case IV (no diagonal numerator), not the earlier four-parameter assumption.
- His quoted51.4(11.6)MeV,359(101)MeV,chi2/dof.82 are not exactly reproduced. Do not equate overlapping errors with full reproduction.
- Gamma studies are ignored diagnostics under xcheck_Srijit. pyerrors covariance uses zero-lag correlation coefficients scaled by Gamma errors; it is NOT a full lag-summed cross-covariance. State which estimator is used.
- Preserve replica IDs, configuration gaps, and common configuration subsets in statistical comparisons. The earlier720 standard/matrix lists differed and were not a paired comparison.
- Increasing block size changes covariance noise as well as autocorrelation treatment. Do not present large-chi2 delete8 conditional errors as improved precision.
- Keep bootstrap/Gamma/deleted-block tests separate from the publication jackknife analysis unless explicitly adopted.

## Verification and Paper Handoff

- For bug fixes, test the relevant symmetry, dimensions, normalization, independent insertion support, cache provenance, and downstream selections. Preserve a numerical/visual baseline and compare changed figures.
- The September 15 audit scripts and reports are in __codex_ignore/processing_audit_20260916 and its parent (the directory name is retained). Two-point and ratio cache audits independently recompute stored objectives; processing audit compares original/new diagrams and blocks raw access for the final analysis.
- Known user-owned utility issue, not used by publication notebooks: doMA_3pt's chi2RThreshold branch compares the mean of parameters/chi2 instead of chi2/Ndof, and the returned model average is not recomputed after that filtering. Do not use that option until corrected with user authorization. A two-fit synthetic test retained the bad-chi2 window and rejected the good one. util.py was not modified in this audit.
- Existing regression tests cover swapped-operator reconstruction, symmetric insertion handling, covariance masks, convergence reporting, model limits, displacement support, and rainbow subset styling.
- Do not claim the large remote strange/charm raw file was validated locally. Source/synthetic checks and equality of old/new saved entries have narrower scope.
- reproduce_figures_codex.py --paper <paper> regenerates every included figure, checks maintained sources are not ignored, and renders pixel comparisons. --diagrams can point a staged build to the ORIGINAL OneDrive source directory. It does not publish prose.
- Read figure_report.json. Missing/new outputs may not be satisfied by stale PDFs. Do not rerun removed fits just because their cached results exist.
- Before manuscript edits run paper_sync_guard_codex_ignore.ps1 -Mode start. Publish through its guard using publish_shared_update_codex.ps1, then compile the LIVE OneDrive main.tex and compare it with staging.
- Combine necessary sandbox escalation for authorized publication into one request. No repeated conversational approval for ordinary in-scope work.
- Inspect changed pages, section transitions, figure order, captions, legends, and reference warnings. Report what actually changed and any limitations.
