# General Paper-Writing Guide

This guide contains reusable writing and citation practices. It does not define project-specific scientific results, notation, fit choices, manuscript structure, file locations, or publication workflow. For N-sigma project context, consult `project2/07_Nsgm/AGENTS.md` only when that project is in scope, and verify it against current sources.

## Prose and Argument

- Read the relevant source material before stating another work's result or interpretation. Do not infer conclusions from titles, abstracts, or a later paper's summary when the original discussion is available.
- Organize a paragraph around one point. State its purpose early, make the reasoning explicit, and connect it naturally to the preceding and following paragraphs.
- Put each explanation in the section where its method or result belongs. Introduce a method before interpreting results that depend on it.
- Define specialized terms and abbreviations before relying on them. Distinguish concepts or methods with similar names when their data, assumptions, or interpretation differ.
- Make comparisons quantitative enough to establish the claimed scale of an effect, but do not catalogue values whose comparison is already clear from a figure.
- When substantive rewriting is authorized, integrate revisions into the affected argument rather than appending isolated sentences. Propagate consequential changes to equations, notation, figures, captions, and conclusions.
- Remove repetition, empty roadmap language, unsupported qualifications, and defensive claims. Keep caveats that are necessary for scientific accuracy.
- Treat editorial comments phrased as questions as requests for judgment. Explain the recommendation and uncertainty, then revise consistently with the best-supported interpretation.
- Prefer direct, precise wording. Avoid rhetorical questions and semicolons in prose unless there is a clear technical or grammatical need.
- Use the conventions of the target journal and relevant field papers as style references, not as substitutes for checking the science.
- Do not make manuscript-wide punctuation or typographic normalizations during a focused revision. Preserve the source style and review history unless the user explicitly requests a separate copy-editing pass; fix the requested figure labels independently.
- Explain the operational meaning of a construction before using it: distinguish the formal definition from how its parameters are determined and how the resulting observable is analyzed.
- Connect observations to decisions in a natural sequence: identify the comparison, describe the relevant behavior, explain its physical meaning, and state the resulting analysis choice. Prefer explicit causal connections over compressed lists of outcomes.
- Discuss literature comparisons as scientific analyses, not as plot rows. Identify the cited work and its method, then explain the physical reason for the comparison; use marker descriptions only to help locate the corresponding results.
- Match the organization of related comparison figures to the scientific discussion. In an integrated discussion, introduce both figures together, distinguish their roles, and explain each work's adopted result alongside its alternative methods. A separate discussion may instead introduce them sequentially. Follow the agreed structure, cite the appropriate figure for each claim, and avoid repeating the selection discussion.
- When a concluding summary is useful, synthesize the setup, main findings, and physical implications before the outlook. Include only numerical anchors that serve this synthesis, and avoid repeating the same results in both a summary paragraph and a list.

## Collaborative Revision

- Distinguish agreement to implement proposed wording from approval to remove its review markings. Keep new edits marked until the user explicitly accepts them into the review baseline; do not advance approval checkpoints merely because the user says to add a proposed sentence.
- Treat senior coauthors' revisions as style references. Read their changes in context and learn reusable patterns of explanation, paragraph organization, and scientific emphasis rather than automatically replacing their wording with the assistant's preferred phrasing.
- Compare the structure before and after an edit, not just individual sentences. Notice how definitions are placed before result comparisons, fit cases are explained through their explicit assumptions, and supporting algebra is shortened when it is not needed for the physical conclusion.
- After moving or splitting discussion, recheck every contextual explanation, comparison, and figure reference in its new location. Keep convention conversions attached to the results they concern, rather than carrying over an introduction that no longer matches the figure.
- When simplifying a technical explanation, retain the physical mechanism and the limits of the evidence. Prefer already defined quantities and standard terminology over new shorthand or extra notation. More equations are not automatically a clearer explanation.
- Keep coauthor comments and replies brief and focused on the remaining question or requested action. Correct an understood problem directly rather than adding a redundant reply. Preserve unresolved comments and ask the relevant coauthor to review the revised passage without repeating its detailed edit history.
- Preserve coauthor language and organization unless the user explicitly requests rewriting. During a correctness check, distinguish factual errors, inconsistent notation, broken references, and obvious typos from optional stylistic preferences. Make only the smallest authorized corrections. Do not add unsolicited caveats, shared-data notes, or explanations of routine plotting conventions during a correction pass. Do not turn internal discussion or the history of our decisions into manuscript justifications. State necessary analysis choices briefly, without repeating them in captions and discussion.
- Editorial authority does not replace scientific verification. Learn from the writing without adopting an accidental typo or an unsupported claim as a general rule, and flag substantive concerns separately.
- Record transferable lessons in this general guide, not project-specific results, numerical choices, or a history of individual edits. Reconcile new guidance with existing rules instead of adding contradictory instructions.

## Figures and Tables

- For result and method comparison plots, describe fit ranges in the corresponding discussion text, not in captions. Use the defined fit-bound symbol with an approximate physical value, such as `$t_s^{\rm low}\approx 1$ fm` or `$t_s^{\rm low}\approx 0.5$ fm`, without redundant decimals or detailed lattice-unit bounds, fit-window pairs, or insertion cuts. Keep precise technical definitions in the analysis section where needed. Keep captions focused on the displayed results, symbols, and methods. Do not explain statistical/systematic error conventions in plot captions unless the user explicitly requests it.
- Cite figures directly with their labels in the text, rather than calling them the first or second figure. When quoting literature results, preserve the published error components without routinely spelling out statistical/systematic labels; briefly explain an unusual final-value prescription when needed.
- For literature comparisons, use the authors' quoted result in the required convention. If separate statistical/systematic components are not explicitly identified for that result, show one quoted error bar rather than infer a decomposition from another table or convention.
- Explain shared visual conventions at their first appearance. Later captions can refer back and describe only what changes.
- Identify the plotted quantity and explain panels in reading order. Define meaningful symbols, colors, lines, and bands, but do not narrate obvious visual facts.
- Keep captions for related figures consistent in terminology, notation, and explanation order. Check captions against the rendered figure and the analysis it represents.
- Table captions should identify the quantities and define non-obvious headers or symbols.
- Discuss a figure near its first substantive use. Keep figure and table numbering in order of first discussion, and inspect the compiled document because float placement can differ from source order.

## Citations and Bibliography

- Cite primary sources that support the specific claim. Use reviews for broad context and historical synthesis, and original studies for particular derivations or results.
- At the first overview of a method or research area, cite the relevant group of studies. Repeat citations when attributing specific findings; do not omit the overview citations merely to avoid repetition.
- At bibliography finalization, obtain BibTeX from INSPIRE-HEP and preserve the exported citation key and metadata. Check whether a proceedings contribution has a relevant full paper and whether a preprint has a published version. Prefer the relevant full or published paper when it supports the same claim.
- If a claim's source or publication status is uncertain, verify it and flag unresolved cases rather than inventing or silently altering bibliographic metadata.
- Rebuild the bibliography and check for missing citations, unresolved references, and title/style issues after changes.

## Whole-Paper Review

- Review the full argument after substantial edits. Check terminology, definitions, attribution, cross-references, figure/table order, and consistency between prose and displayed results.
- Check the compiled document at its intended reading size for clipped text, overfull lines, poor float placement, and unreadable figures.
- Keep technical reproducibility details in the manuscript only when they serve the reader's argument or are required to reproduce the result; otherwise place them in the appropriate supplementary material.
