"""Three-current ratio comparison for the HPC proposal, without changing it."""

import ast
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "__codex_ignore" / "proposal_annex_20261011"
OUTPUT.mkdir(parents=True, exist_ok=True)


def ratio_cells(filename, indices):
    notebook = json.loads((HERE / filename).read_text(encoding="utf-8"))
    namespace = {"__name__": "proposal_ratio_inputs"}
    os.chdir(HERE)
    for index in indices:
        source = "".join(notebook["cells"][index]["source"])
        tree = ast.parse(source)
        # Only the ratio construction is needed; do not touch the fit-cache guard.
        tree.body = [statement for statement in tree.body if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "guard_fit_cache"
        )]
        exec(compile(tree, filename, "exec"), namespace)
    return namespace


light = ratio_cells("analysis_3pt_light_codex.ipynb", [0, 9, 11])
heavy = ratio_cells("analysis_3pt_strange_charm_codex.ipynb", [0, 2, 10])
yu, yuc = light["yu"], light["yuc"]
reference_tfs = light["TFS"]
spacing = light["XUNIT"]
channels = [
    ("light", light["ratio_standard"], light["ratio_gevp"],
     light["YUNIT"], reference_tfs, 1, r"$\sigma_{\pi N}$ [MeV]", (0, 80), [0, 20, 40, 60, 80]),
    ("strange", heavy["strange_standard"], heavy["strange_full"],
     heavy["YUNIT_STRANGE"], reference_tfs, 1, r"$\sigma_s$ [MeV]", (0, 80), [0, 20, 40, 60, 80]),
    ("charm", heavy["charm_standard"], heavy["charm_full"],
     heavy["YUNIT_CHARM"], [10, 18], 2, r"$\sigma_c$ [MeV]", (-50, 350), [0, 100, 200, 300]),
]

os.chdir(OUTPUT)
yuc.apply_paper_style({"font.size": 9, "legend.fontsize": 7.5,
                       "lines.markersize": 3.3, "errorbar.capsize": 2.2})
figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
for axis, (name, standard, gevp, unit, tfs, cut, ylabel, limits, ticks) in zip(axes, channels):
    for ratio, face, offset in [(standard, "white", -.010), (gevp, None, .010)]:
        displayed = {tf: yu.symmetrizeRatio(ratio)[tf] for tf in tfs}
        assert all(values.shape == (732, tf + 1) for tf, values in displayed.items())
        yu.plot_rainbow(axis, displayed, tcmin=cut, xunit=spacing, yunit=unit,
                        mfc=face, shift=offset / (.1 * spacing),
                        tfs_reference=reference_tfs)
    axis.set(xlabel=r"$t_{\rm ins}-t_s/2$ [fm]", ylabel=ylabel,
             xlim=(-.82, .82), xticks=[-.6, -.3, 0, .3, .6],
             ylim=limits, yticks=ticks)
    axis.grid(False)

labels = [r"$R_{\rm std}$", r"$R_{\rm GEVP}$"]
axes[0].legend(handles=yuc.ratio_legend_handles(labels), loc="upper center", ncols=2,
               handletextpad=.3, columnspacing=.9, borderpad=.3)
figure.subplots_adjust(left=.068, right=.994, bottom=.245, top=.965, wspace=.40)
figure.savefig(OUTPUT / "sigma_ratios_proposal_annex.pdf")
figure.savefig(OUTPUT / "sigma_ratios_proposal_annex.png", dpi=240)
plt.close(figure)

sources = [HERE / name for name in ["analysis_3pt_light_codex.ipynb",
           "analysis_3pt_strange_charm_codex.ipynb", "plot_proposal_annex_codex.py"]]
sources += [ROOT / "util.py", ROOT / "util_codex.py",
            ROOT / "__codex_ignore/cB211.072.64/pkl/analysis_2pt_codex/reg_ignore/evec_ratios.pkl"]
sources += [HERE / "pkl/processData/reg_ignore" / name for name in [
    "data.pkl", "data_jsc.pkl", "data_jsc_NsgmJNsgm.pkl"]]
provenance = {
    "sample_count": 732,
    "spacing_fm": spacing,
    "displayed_separations": {channel[0]: channel[4] for channel in channels},
    "insertion_cuts": {channel[0]: channel[5] for channel in channels},
    "ratio_source": {"light": "paper reduced-GEVP ratio", "strange": "full GEVP ratio",
                     "charm": "full GEVP ratio"},
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
}
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
print(OUTPUT / "sigma_ratios_proposal_annex.pdf")
