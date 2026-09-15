"""Shared processing, analysis, and plotting helpers for the N-sigma paper."""

import hashlib

import h5py
import matplotlib as mpl
import numpy as np
import util_Nsgm as yn


PAPER_STYLE = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
    "lines.markeredgewidth": 1.0,
    "lines.markersize": 3.5,
    "errorbar.capsize": 3,
}


def paper_style(overrides=None):
    return PAPER_STYLE | (overrides or {})


def apply_paper_style(overrides=None):
    mpl.rcParams.update(paper_style(overrides))


def ratio_legend_handles(labels):
    """Representative open/filled circles for a baseline/transformed ratio pair."""
    return [mpl.lines.Line2D([], [], color="black", marker="o", ls="",
                            mfc=face, label=label)
            for label, face in zip(labels, ["white", "black"])]


def compact_four_panels(figure, axes):
    """Ratio, midpoint, fit, and gap panels with space for the gap-axis label."""
    axes = np.atleast_2d(axes)
    figure.tight_layout(w_pad=0)
    left, right = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    gap, energy_gap = .012, .075
    weights = np.array([1.3, 1, 1, 1])
    unit = (right - left - 2 * gap - energy_gap) / weights.sum()
    x = left
    for column, (axis, weight) in enumerate(zip(axes[0], weights)):
        position = axis.get_position()
        if column:
            x += energy_gap if column == 3 else gap
        axis.set_position([x, position.y0, unit * weight, position.height])
        axis.tick_params(axis="y", left=True, right=True, labelleft=column in [0, 3])
        x += unit * weight


def errorbar_with_selected(
    axis, x, y, yerr, selected=None, *, color, fmt, mfc=None, label=None,
    star_size=5.5, **kwargs,
):
    """Draw selected entries once as open stars and all others with ``fmt``."""
    x, y, yerr = map(np.atleast_1d, (x, y, yerr))
    mask = np.zeros(x.size, dtype=bool)
    if selected is not None:
        indices = np.atleast_1d(selected)
        mask = indices.astype(bool) if indices.dtype == bool else mask
        if indices.dtype != bool:
            mask[indices.astype(int)] = True

    if np.any(~mask):
        axis.errorbar(x[~mask], y[~mask], yerr[~mask], color=color, fmt=fmt,
                      mfc=mfc, label=label, **kwargs)
    if np.any(mask):
        axis.errorbar(
            x[mask], y[mask], yerr[mask], color=color, fmt="*", mfc="white",
            mec=color, mew=PAPER_STYLE["lines.markeredgewidth"], ms=star_size,
            label=label if np.all(mask) else None, **kwargs,
        )


def finish_shared_y_panels(figure, axes, *, w_pad=0.5, wspace=0.06, rect=None):
    """Apply consistent spacing and retain y ticks on shared-y interior panels."""
    axes = np.atleast_2d(axes)
    for row in axes:
        for column, axis in enumerate(row):
            axis.tick_params(axis="y", which="both", left=True, right=True,
                             labelleft=column == 0)

    kwargs = {"w_pad": w_pad}
    if rect is not None:
        kwargs["rect"] = rect
    figure.tight_layout(**kwargs)
    figure.subplots_adjust(wspace=wspace)


import matplotlib.pyplot as plt
import util as yu


def fit_meff_comparison(meff, ranges, starts, *, corrQ=True, label, overwrite=False):
    """Keep multistate fits and use uncorrelated one-state comparison scans."""
    specification = np.asarray([repr([list(r) for r in ranges]), repr(list(starts)), str(corrQ)])
    cache_label = f"{label}_{sample_cache_tag(meff, specification)}"
    if not overwrite and yu.load_pkl_internal(cache_label) is None:
        legacy = yu.load_pkl_internal(label)
        if legacy is not None and _meff_cache_matches(legacy, meff, ranges, corrQ):
            yu.save_pkl_internal(cache_label, legacy)
        elif legacy is not None:
            print(f"Recompute stale effective-mass cache: {label}", flush=True)
    scans = list(yu.doFits_meff_nst(
        meff, ranges, starts.copy(), corrQ=corrQ, label=cache_label, overwrite=overwrite))
    scans[0] = yu.doFits_2pt(
        meff, ranges[0], yu.func_meff_1st, starts[:1], corrQ=False,
        label=f"{label}_one_state_uncorrelated_{sample_cache_tag(meff)}", overwrite=overwrite)
    return scans


def _meff_cache_matches(scans, meff, ranges, correlated):
    """Allow legacy-cache migration only when its objectives match current data."""
    if len(scans) != len(ranges):
        return False
    upper = yu.find_fitmax(meff)
    models = [yu.func_meff_1st, yu.func_meff_2st, yu.func_meff_3st]
    for fits, lowers, model in zip(scans, ranges, models):
        if [f[0] for f in fits] != [t for t in lowers if upper - t >= 2]:
            return False
        for lower, parameters, chi2, ndof in fits:
            times = np.arange(lower, upper)
            if parameters.shape[0] != len(meff) or ndof != len(times) - parameters.shape[1]:
                return False
            mean, _, covariance = yu.jackmec(meff[:, times])
            if not correlated:
                covariance = np.diag(np.diag(covariance))
            residual = np.array([model(times, *p) - mean for p in parameters])
            expected = np.sum(np.linalg.solve(np.linalg.cholesky(covariance), residual.T)**2, axis=0)
            if not np.allclose(expected, np.ravel(chi2), rtol=1e-7, atol=1e-7):
                return False
    return True

def sample_cache_tag(*arrays):
    """Identify the actual samples used by a cached analysis."""
    import hashlib
    digest = hashlib.sha256()
    for values in arrays:
        values = np.ascontiguousarray(values)
        digest.update(str((values.shape, values.dtype.str)).encode('ascii'))
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]

def guard_fit_cache(*inputs):
    """Stop legacy label-only cache reuse after an analysis input changes."""
    label = "verified_input_samples_codex"
    current = sample_cache_tag(*inputs)
    previous = yu.load_pkl_internal(label)
    if previous is not None and previous != current:
        raise ValueError("Analysis inputs changed. Review/rekey the affected fit caches before rerunning; "
                         "do not discard this guard and reuse old label-only fits.")
    if previous is None:
        yu.save_pkl_internal(label, current)


def plot_eigenvectors(times, components, scans, selected, spacing, config):
    """Component data and lower-bound scans, with the later GEVP time on both axes."""
    with mpl.rc_context(paper_style({"lines.markersize": 3.3, "errorbar.capsize": 2.5})):
        fig, axes = yu.getFigAxs(3, 2, sharex="col", sharey="row", Lrow=1.05, Lcol=1.75)
        lower, upper = config["window"]
        for row, name in enumerate(["v", "s", "w"]):
            mean, error = yu.jackme(components[name])
            display = (times >= config["display"][0]) & (times <= config["display"][1])
            fitted = (times >= lower) & (times <= upper)
            for mask, face in [(display & ~fitted, "red"), (display & fitted, "white")]:
                axes[row, 0].errorbar(times[mask] * spacing, mean[mask], error[mask],
                                      fmt="s", color="red", mfc=face)
            scan = scans[name]
            x = np.array([fit[0] for fit in scan]) * spacing
            values = np.array([yu.jackme(fit[1]) for fit in scan])
            selection = [fit[0] for fit in scan].index(lower)
            errorbar_with_selected(axes[row, 1], x, values[:, 0], values[:, 1], selection,
                                   color="red", fmt="s", mfc="red")
            mean, error = yu.jackme(selected[name])
            for axis in (axes[row] if row < 2 else [axes[row, 0]]):
                axis.axhspan(mean - error, mean + error, color="red", alpha=.18, linewidth=0)
        values = np.array([yu.jackme(-fv[1] * fs[1]) for fv, fs in zip(scans["v"], scans["s"])])
        mean, error = values[selection]
        axes[2, 1].axhspan(mean - error, mean + error, color="blue", alpha=.15, linewidth=0)
        errorbar_with_selected(axes[2, 1], x + .012, values[:, 0], values[:, 1], selection,
                               color="blue", fmt="d", mfc="blue")
        labels = [r"$v_{0,N\sigma}/v_{0,N}$", r"$v_{1,N}/v_{1,N\sigma}$", r"$W$"]
        for row, label in enumerate(labels):
            axes[row, 0].set(ylabel=label, **config["rows"][row])
            for axis in axes[row]:
                axis.tick_params(axis="y", left=True, right=True)
                axis.margins(x=.08)
        axes[2, 0].set(xlabel=r"$t$ [fm]", **config.get("left", {}))
        axes[2, 1].set(xlabel=r"$t_{\rm low}$ [fm]", **config.get("right", {}))
        for color, marker, label in [("red", "s", "direct fit"), ("blue", "d", "reconstructed")]:
            axes[2, 1].plot([], [], color=color, marker=marker, ls="", label=label)
        axes[2, 1].legend(loc="upper center", bbox_to_anchor=(.5, .94), fontsize=7,
                          handletextpad=.35, borderaxespad=0)
        fig.tight_layout(w_pad=0, h_pad=0)
        fig.subplots_adjust(wspace=0, hspace=0)
        yu.finalizePlot("GEVP_vw", tightQ=False)


def plot_correlator_comparison(channel, masses, fit_scans, selection, selected_samples,
                              laplace_data, spacing, energy_unit, config):
    A, AINV_GEV = spacing, energy_unit
    PLOT_STYLE = paper_style({"lines.markersize": 3.3, "errorbar.capsize": 2.5})
    include_three_state = config.get("include_three_state", any(len(scans) > 2 and len(scans[2]) for scans in fit_scans))
    with mpl.rc_context(PLOT_STYLE):
        fig, panels = plt.subplot_mosaic(
            [["effective", "effective"], ["ground", "excited"]], figsize=(7.1, 3.1),
            gridspec_kw={"height_ratios": [1, 1.05], "width_ratios": [1.6, 1]},
        )
        effective, ground, excited = panels.values()
        fit_colors = ["red", "green", "blue"]
        projection_shifts = [0, .012]
        state_shifts = [-.04, 0, .04]
        for method, masses, fits_by_state, shift, filled in zip(
            ["standard", "GEVP projected"], masses, fit_scans,
            projection_shifts, [False, True],
        ):
            mean, error = yu.jackme(masses)
            display_times = config["display_times"][int(filled)]
            effective.errorbar(display_times * A + shift, mean[display_times] * AINV_GEV,
                               error[display_times] * AINV_GEV, fmt="s", color="black",
                               mfc="black" if filled else "white", label=method)
            for state, (fits, marker, method_shift) in enumerate(zip(fits_by_state, ["s", "o", "d"], state_shifts)):
                if state == 2 and not include_three_state:
                    continue
                for lower, parameters, _, _ in fits:
                    value, uncertainty = yu.jackme(parameters[:, 0] * AINV_GEV)
                    x = lower * A + shift + method_shift
                    ground.errorbar(
                        x, value, uncertainty, color=fit_colors[state],
                        fmt=marker,
                        mfc=fit_colors[state] if filled else "white",
                        markersize=PLOT_STYLE["lines.markersize"],
                    )
                    if state:
                        value, uncertainty = yu.jackme(np.sum(parameters[:, :2], axis=1) * AINV_GEV)
                        excited.errorbar(
                            x, value, uncertainty, color=fit_colors[state],
                            fmt=marker,
                            mfc=fit_colors[state] if filled else "white",
                            markersize=PLOT_STYLE["lines.markersize"],
                        )

        for axis in [ground, excited]:
            selected_x = selection[1] * A + projection_shifts[selection[0]] + state_shifts[1]
            yu.addRefLine(axis, selected_x, hv="v")

        for axis, samples in [(effective, selected_samples[:, 0]), (ground, selected_samples[:, 0]),
                              (excited, np.sum(selected_samples[:, :2], axis=1))]:
            value, uncertainty = yu.jackme(samples * AINV_GEV)
            axis.axhspan(value - uncertainty, value + uncertainty,
                         color=fit_colors[1], alpha=.16, linewidth=0)

        laplace_face = "white" if selection[0] == 0 else "darkorange"
        laplace_label = "Laplace on standard" if selection[0] == 0 else "Laplace on GEVP"
        laplace_times, laplace_masses, laplace_scans = laplace_data
        mean, error = yu.jackme(laplace_masses * energy_unit)
        effective.errorbar(laplace_times * A + .024, mean, error, fmt="^", color="darkorange",
                           mfc=laplace_face, label=laplace_label)
        for axis, index in [(ground, 0), (excited, 1)]:
            x = np.array([fit[0] for fit in laplace_scans]) * A + .024
            values = np.array([yu.jackme((fit[1][:, 0] if index == 0 else fit[1][:, :2].sum(axis=1)) * energy_unit)
                               for fit in laplace_scans])
            if len(values):
                axis.errorbar(x, values[:, 0], values[:, 1], fmt="^", color="darkorange", mfc=laplace_face)

        effective.set(xlabel=r"$t$ [fm]", ylabel=r"$E^{\rm eff}$ [GeV]")
        ground.set(xlabel=r"$t_{\rm low}$ [fm]", ylabel=r"$E_0$ [GeV]")
        excited.set(xlabel=r"$t_{\rm low}$ [fm]", ylabel=r"$E_1$ [GeV]")
        for name, axis in panels.items():
            axis.set(**config.get(name, {}))
            if name not in config:
                low, high = axis.get_ylim()
                axis.set_ylim(low, high + .3 * (high - low))
                axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
                axis.margins(x=.06)
        effective.legend(**{"loc": "upper center", "bbox_to_anchor": (.5, .90),
                            "ncols": 3, "borderaxespad": 0, **config.get("effective_legend", {})})
        fit_handles = [
            mpl.lines.Line2D([], [], marker=marker, color=color, ls="", label=label)
            for marker, color, label in zip(["s", "o", "d"], fit_colors,
                                            ["one-state", "two-state", "three-state"])
            if include_three_state or label != "three-state"
        ]
        ground.legend(handles=fit_handles, loc="upper right", bbox_to_anchor=(.98, .90),
                      ncols=2, fontsize=7, borderaxespad=0)
        fig.tight_layout(w_pad=.25, h_pad=.15)
        yu.finalizePlot(f"C2pt_{channel}_GEVP_compare", tightQ=False)


def plot_overlap_ratios(scans, selections, spacing, config):
    A = spacing
    PLOT_STYLE = paper_style({"lines.markersize": 3.3, "errorbar.capsize": 2.5})
    with mpl.rc_context(PLOT_STYLE):
        fig, axis = plt.subplots(figsize=(3.4, 2.1))
        specifications = [
            ("N", 1, "blue", "o", r"$N$, two-state", -.012),
            ("N", 2, "red", "d", r"$N$, three-state", 0),
            ("Nsgm", 1, "darkorange", "s", r"$\widetilde{N\sigma}$, two-state", .012),
        ]
        for channel, state, color, marker, label, shift in specifications:
            fits = scans.get((channel, state), [])
            if not fits:
                continue
            if channel == "Nsgm":
                fits = [fit for fit in fits if yu.jackme(fit[1][:, 2])[1] < 2]
            x = np.array([fit[0] for fit in fits]) * A + shift
            values = np.array([yu.jackme(fit[1][:, 2]) for fit in fits])
            selected = next((i for i, fit in enumerate(fits) if state == 1 and fit[0] == selections[channel]), None)
            errorbar_with_selected(
                axis, x, values[:, 0], values[:, 1], selected, color=color, fmt=marker,
                mfc=color, label=label,
            )
        axis.set(xlabel=r"$t_{\rm low}$ [fm]", ylabel=r"$|Z_{k,1}/Z_{k,0}|^2$",
                 **config)
        if "ylim" not in config:
            axis.set_ylim(0, axis.get_ylim()[1] * 1.35)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, .92), ncols=3, fontsize=6.8,
                    columnspacing=.7, handletextpad=.25, borderaxespad=0)
        fig.tight_layout(pad=.25)
        yu.finalizePlot("C2pt_overlap_compare", tightQ=False)


def plot_scan(axis, fits, cut, color, marker, label, shift=0, energy=False, selection=None, open_symbol=False, *, xunit, yunit, energy_unit):
    scan = sorted((fit for fit in fits if fit[0][1] == cut), key=lambda fit: fit[0][0])
    if not scan:
        return
    x = (np.array([fit[0][0] for fit in scan]) + shift) * xunit
    samples = [fit[1][:, 1] * energy_unit if energy else fit[1][:, 0] * yunit for fit in scan]
    values = np.array([yu.jackme(data) for data in samples])
    selected = None if selection is None else next(
        index for index, fit in enumerate(scan) if fit[0] == selection
    )
    errorbar_with_selected(
        axis, x, values[:, 0], values[:, 1], selected, color=color, fmt=marker, label=label,
        mfc="white" if open_symbol else None,
    )


def draw_ratio_pair(axis, baseline, transformed, cuts, labels, midpoint_axis=None, tfmin=None, legend_loc="lower left", *, xunit, yunit, limits=None):
    ratios = [yu.symmetrizeRatio(baseline), yu.symmetrizeRatio(transformed)]
    tfs_reference = sorted(set(baseline) | set(transformed))
    for ratio, cut, face, shift in zip(ratios, cuts, ["white", None], [0, .1]):
        yu.plot_rainbow(axis, ratio, tfmin=tfmin, tcmin=cut, xunit=xunit, yunit=yunit,
                        mfc=face, shift=shift, mid_tfshift=3 * shift, ax_mid=midpoint_axis,
                        tfs_reference=tfs_reference)
    handles = ratio_legend_handles(labels)
    axis.legend(handles, labels, loc=legend_loc, ncols=2, fontsize=7,
                columnspacing=.7, handletextpad=.25, framealpha=1)
    if limits:
        axis.set(**limits)
        if midpoint_axis is not None:
            midpoint_axis.set(**limits)


def start_ratio_plot(baseline, transformed, cuts, labels, columns=3, *, xunit, yunit, limits):
    def options(ratio, cut, face, shift):
        return {"tf2ratio": yu.symmetrizeRatio(ratio),
                "rainbow:[tfmin,tfmax,tcmin,dt]": [None, None, cut, None],
                "xyunit": (xunit, yunit), "mfc:[global]": [face],
                "shift:[rainbow,midpoint,fit]": [shift, 3 * shift, 0]}
    figure, axes = yu.makePlot_3pt(options(baseline, cuts[0], "white", 0),
        shows=["rainbow", "midpoint"] + [None] * (columns - 2),
        Lrow=2.35, Lcol=7.1 / columns, sharey=False)
    yu.makePlot_3pt(options(transformed, cuts[1], None, .1),
        shows=["rainbow", "midpoint"] + [None] * (columns - 2), figAxs=(figure, axes), sharey=False)
    axes[0, 0].legend(handles=ratio_legend_handles(labels), loc="upper center",
                      bbox_to_anchor=(.5, .96), ncols=2, columnspacing=.8, handletextpad=.3, framealpha=1)
    for axis in axes[0, :2]:
        axis.set(**limits)
    axes[0, 0].set_ylabel(r"$\sigma_{\pi N}$ [MeV]")
    axes[0, 1].tick_params(labelleft=False)
    return figure, axes


def plot_standard_gevp(ratio_standard, ratio_gevp, fit_cases, nucleon_standard,
                       nsigma_selected, xunit, yunit, energy_unit, config, output_name):
    from functools import partial
    scan_plot = partial(plot_scan, xunit=xunit, yunit=yunit, energy_unit=energy_unit)
    STANDARD_CUT = config.get("cut", 2)
    AINV_GEV = energy_unit
    figure, axes = start_ratio_plot(ratio_standard, ratio_gevp, [1, 1],
                                    [r"$R_{\rm std}$", r"$R_{\rm GEVP}^{d}$"], 4, xunit=xunit, yunit=yunit, limits=config["limits"])
    fit_axis, gap_axis = axes[0, 2], axes[0, 3]
    styles = config.get("fit_styles", [
        ("red", "o", "I", -.4, False),
        ("green", "^", "II", -.2, True),
        ("blue", "s", "III", 0, True),
        ("purple", "v", "IV", .2, True),
        ("darkorange", "d", "V", .4, True),
    ])
    if len(fit_cases) != len(styles):
        raise ValueError("Each fit case needs one plotting style")
    for fits, (color, marker, label, shift, has_energy) in zip(fit_cases, styles):
        selection = config.get("selection") if label == config.get("selection_case", "III") else None
        scan_plot(fit_axis, fits, STANDARD_CUT, color, marker, rf"$R_{{\rm std}}$, {label}", shift,
                  selection=selection, open_symbol=True)
        if has_energy:
            scan_plot(gap_axis, fits, STANDARD_CUT, color, marker, "_nolegend_", shift,
                      energy=True, selection=selection, open_symbol=True)
    gap_mean, gap_error = yu.jackme(nucleon_standard[:, 1] * AINV_GEV)
    gap_axis.axhspan(gap_mean - gap_error, gap_mean + gap_error, color="red", alpha=.14,
                     label=r"$\Delta E_1^{\rm 2pt}$")
    nsigma_mean, nsigma_error = yu.jackme((nsigma_selected[:, 0] - nucleon_standard[:, 0]) * AINV_GEV)
    gap_axis.axhspan(nsigma_mean - nsigma_error, nsigma_mean + nsigma_error, color="grey", alpha=.2,
                     label=r"$\Delta\widetilde E_0^{N\sigma}$")
    fit_axis.set(**config["limits"], xlabel=r"$t_s^{\rm low}$ [fm]")
    gap_axis.set(ylabel=r"$\Delta E$ [GeV]", **config.get("gap", {}), xlabel=r"$t_s^{\rm low}$ [fm]")
    fit_axis.legend(ncols=2, fontsize=6, loc="lower left",
                    handlelength=1, handletextpad=.3, columnspacing=.5, borderpad=.3)
    gap_axis.legend(ncols=1, loc="upper right", fontsize=6.5,
                    bbox_to_anchor=(.96, .97), borderaxespad=0, handletextpad=.3)
    axes[0, 0].set(**config["rainbow"])
    axes[0, 1].set(**config["midpoint"])
    for axis in [fit_axis, gap_axis]:
        axis.set(**config["scan"])
    compact_four_panels(figure, axes)
    yu.finalizePlot(output_name, tightQ=False)


def plot_laplace_summary(ratio_standard, ratio_laplace, ratio_gevp, ratio_laplace_gevp,
                         fit_cases, xunit, yunit, energy_unit, config, output_name,
                         *, fits_laplace_gevp_iv=None):
    from functools import partial
    pair = partial(draw_ratio_pair, xunit=xunit, yunit=yunit, limits=config["limits"])
    scan_plot = partial(plot_scan, xunit=xunit, yunit=yunit, energy_unit=energy_unit)
    fits_standard_no_diagonal, laplace_fits, fits_laplace_gevp = fit_cases
    STANDARD_CUT, LAPLACE_CUT = config.get("cuts", (2, 4))
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=True,
                                gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
    pair(axes[0], ratio_standard, ratio_laplace, [STANDARD_CUT, LAPLACE_CUT],
                    [r"$R_{\rm std}$", r"$R_{\rm Lap}$"], legend_loc="upper center")
    pair(axes[2], ratio_gevp, ratio_laplace_gevp, [STANDARD_CUT, LAPLACE_CUT],
                    [r"$R_{\rm GEVP}^{d}$", r"$R_{\rm LG}$"], legend_loc="upper center")
    axes[0].set_xlabel(r"$t_{\rm ins}-t_s/2$ [fm]")
    axes[2].set_xlabel(r"$t_{\rm ins}-t_s/2$ [fm]")
    methods = [
        (fits_standard_no_diagonal, STANDARD_CUT, "red", "d", r"$R_{\rm std}$, V", -.2, True),
        (laplace_fits, LAPLACE_CUT, "darkorange", "s", r"$R_{\rm Lap}$", 0, False),
        (fits_laplace_gevp, LAPLACE_CUT, "green", "^", r"$R_{\rm LG}$, V", .2, False),
    ]
    if fits_laplace_gevp_iv is not None:
        methods[-1] = (*methods[-1][:5], .4, methods[-1][-1])
        methods.insert(2, (fits_laplace_gevp_iv, LAPLACE_CUT, "purple", "v",
                           r"$R_{\rm LG}$, IV", .2, False))
    for fits, cut, color, marker, label, shift, open_symbol in methods:
        scan_plot(axes[1], fits, cut, color, marker, label, shift, open_symbol=open_symbol)
    axes[0].set_ylabel(r"$\sigma_{\pi N}$ [MeV]")
    axes[1].set(xlabel=r"$t_s^{\rm low}$ [fm]", **config["limits"])
    axes[1].legend(loc="lower left", fontsize=6.5)
    for axis in [axes[0], axes[2]]:
        axis.set(**config["rainbow"])
    axes[1].set(**config["scan"])
    finish_shared_y_panels(figure, axes, w_pad=.5, wspace=.06)
    figure.subplots_adjust(left=.09)
    yu.finalizePlot(output_name, tightQ=False)


def filter_insertion_ratio(ratios, gap_samples, displacement):
    """Apply L/lambda^2 sample by sample; leave unsupported endpoints undefined."""
    eigenvalue = 2 * np.cosh(displacement * np.asarray(gap_samples)) - 2
    filtered = {}
    d = displacement
    for ts, values in ratios.items():
        if len(values) != len(eigenvalue):
            raise ValueError("Filter and ratio samples must be aligned")
        result = np.full_like(values, np.nan)
        difference = values[:, 2*d:] + values[:, :-2*d] - 2 * values[:, d:-d]
        result[:, d:-d] = values[:, d:-d] - difference / eigenvalue[:, None]
        filtered[ts] = result
    return filtered


def plot_laplace_midpoints(reduced, filtered_reduced, xunit, yunit, config, output_name):
    """Reduced/filtered rainbows and integer-separation midpoints at column width."""
    figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.95), sharey=True,
                               gridspec_kw={"width_ratios": [1.3, 1]})
    cuts = config.get("cuts", (2, 4))
    tfs = sorted(reduced)
    labels = [r"$R_{\rm GEVP}^{d}$", r"$R_{\rm LG}$"]
    handles = ratio_legend_handles(labels)
    for values, cut, face, shift in zip(
        [reduced, filtered_reduced], cuts, ["white", None], [0, .1]
    ):
        even = {ts: value for ts, value in yu.symmetrizeRatio(values).items() if ts % 2 == 0}
        yu.plot_rainbow(axes[0], even, tcmin=cut, xunit=xunit, yunit=yunit,
                        mfc=face, shift=shift, tfs_reference=tfs)
    axes[0].legend(handles=handles, loc="upper center", ncols=2,
                   fontsize=6.5, columnspacing=.5, handletextpad=.2, framealpha=1)
    axes[0].set(xlabel=r"$t_{\rm ins}-t_s/2$ [fm]", **config["rainbow"])
    for index, ts in enumerate(tfs):
        for values, cut, face, shift in zip([reduced, filtered_reduced], cuts, ["white", None], [0, .3]):
            if ts // 2 < cut:
                continue
            midpoint = .5 * (values[ts][:, ts // 2] + values[ts][:, (ts + 1) // 2])
            mean, error = yu.jackme(midpoint * yunit)
            yu.errorbar(axes[1], (ts + shift) * xunit, mean, error,
                        color=yu.colors16[index], fmt=yu.fmts16[index], mfc=face)
    for axis in axes:
        axis.set(**config["limits"])
    axes[0].set_ylabel(config["ylabel"])
    axes[1].set(xlabel=r"$t_s$ [fm]", **config["midpoint"])
    finish_shared_y_panels(figure, axes, w_pad=.5, wspace=.06)
    yu.finalizePlot(output_name, tightQ=False)


def plot_double_laplace(ratio_gevp, ratio_double_laplace, double_cut, xunit, yunit, config):
    figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True,
                                gridspec_kw={"width_ratios": [1.3, 1]})
    baseline = ratio_gevp
    filtered = {tf: ratio for tf, ratio in ratio_double_laplace.items() if tf >= config.get("filtered_min", 2 * double_cut)}
    draw_ratio_pair(axes[0], baseline, filtered, [1, double_cut],
                    [r"$R_{\rm GEVP}^{d}$", r"$R_{\rm L2G}^{\Sigma}$"], axes[1], xunit=xunit, yunit=yunit, limits=config.get("limits"),
                    legend_loc=config.get("legend_loc", "lower left"))
    axes[0].set_ylabel(r"$\sigma_{\pi N}$ [MeV]")
    axes[0].set_xlabel(r"$t_{\rm ins}-t_s/2$ [fm]")
    axes[1].set_xlabel(r"$t_s$ [fm]")
    axes[0].set(**config["rainbow"])
    axes[1].set(**config["midpoint"])
    if "limits" not in config:
        low, high = axes[0].get_ylim()
        axes[0].set_ylim(low, high + .45 * (high - low))
        axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    finish_shared_y_panels(figure, axes, w_pad=.5, wspace=.06)
    yu.finalizePlot("RLG_RL2G", tightQ=False)


def plot_energy_scales(ensemble, nucleon_standard, nsigma_selected, standard_gap,
                       laplace_gap, rlg_gap, energy_unit, config=None, *, show_resonances=True):
    ENS, AINV_GEV = ensemble, energy_unit
    mpi = yu.ens2mpi[ENS] / 1000
    minimum_momentum = 2 * np.pi * .1973269804 / (yu.ens2NL[ENS] * yu.ens2a[ENS])
    nucleon = nucleon_standard[:, 0] * AINV_GEV
    moving_nucleon = np.sqrt(nucleon**2 + minimum_momentum**2)
    moving_pion1 = np.sqrt(mpi**2 + minimum_momentum**2)
    moving_pion2 = np.sqrt(mpi**2 + 2 * minimum_momentum**2)

    references = [
        nucleon, nucleon + 2 * mpi, moving_nucleon + moving_pion1,
        moving_nucleon + mpi + moving_pion1, nucleon + 2 * moving_pion1,
        nucleon + 2 * moving_pion2,
    ]
    reference_labels = [
        r"$N_0$", r"$N_0\pi_0\pi_0$", r"$N_1\pi_1$",
        r"$N_1\pi_0\pi_1$", r"$N_0\pi_1\pi_1$", r"$N_0\pi_2\pi_2$",
    ]
    scales = [
        nucleon + standard_gap * AINV_GEV,
        nucleon + laplace_gap * AINV_GEV,
        nsigma_selected[:, 0] * AINV_GEV,
        np.sum(nucleon_standard[:, :2], axis=1) * AINV_GEV,
        nucleon + rlg_gap * AINV_GEV,
        (nucleon_standard[:, 1] + nsigma_selected[:, 0]) * AINV_GEV,
        np.sum(nsigma_selected[:, :2], axis=1) * AINV_GEV,
    ]
    scale_labels = [
        r"$E_1^{\rm 3pt}$", r"$E_1^{\rm Lap}$", r"$\widetilde E_0^{N\sigma}$",
        r"$E_1^{\rm 2pt}$", r"$E_1^{\rm LG}$",
        r"$m_N+\Delta E_1^{\rm 2pt}+\Delta\widetilde E_0^{N\sigma}$",
        r"$\widetilde E_1^{N\sigma}$",
    ]
    scale_y = [2.4, 2.9, 3.4, 1.0, 4.8, 5.3, 5.8]
    if not show_resonances:
        scale_y = [1.0, 1.5, 2.0, .5, 2.5, 3.0, 3.5]
    upper = 6.2 if show_resonances else 3.9

    figure, axis = plt.subplots(figsize=(7.1, 3.0 if show_resonances else 2.3))
    for y, samples, color, marker in zip(
        scale_y, scales, ["green"] * 3 + ["red"] + ["blue"] * 3, yu.fmts8[:7]
    ):
        value, error = yu.jackme(samples)
        yu.errorbar(axis, [value], [y], xerr=[error], color=color, fmt=marker)

    resonance_y = [1.9, .5, 4.3]
    resonance_labels = [r"$N\sigma$", r"$N^*$", r"$N^*\sigma$"]
    resonance = [[yu.jackme(nucleon)[0] + .5, .55 / 2], [1.440, .350 / 2], [1.940, .900 / 2]]
    if show_resonances:
        for y, (value, error), marker in zip(resonance_y, resonance, [yu.fmts8[i] for i in [0, 2, 3]]):
            yu.errorbar(axis, [value], [y], xerr=[error], color="black", fmt=marker, mfc="white")

    reference_values = np.array([yu.jackme(samples) for samples in references])
    for value in reference_values[:, 0]:
        yu.addRefLine(axis, value, hv="v", color="grey", ls="--", lw=.8, zorder=0)
    for (value, _), label, side in zip(reference_values, reference_labels, [1, -1, 1, 1, 1, 1]):
        axis.text(value + side * .012, upper - .12, label, rotation=90, rotation_mode="anchor",
                  ha="right", va="top" if side > 0 else "bottom", fontsize=7.5, clip_on=True)

    y_positions = [.5, 1.0, 1.9, 2.4, 2.9, 3.4, 4.3, 4.8, 5.3, 5.8]
    y_labels = [resonance_labels[1], scale_labels[3], resonance_labels[0],
                *scale_labels[:3], resonance_labels[2], *scale_labels[4:]]
    if not show_resonances:
        y_positions, y_labels = zip(*sorted(zip(scale_y, scale_labels)))
    axis.set(xlabel=r"$E$ [GeV]", ylim=(.2, upper),
             yticks=y_positions, yticklabels=y_labels)
    if config:
        axis.set(**config)
    else:
        axis.margins(x=.05)
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
    figure.tight_layout(pad=.25)
    yu.finalizePlot("energy_scales", tightQ=False)


# Rest-frame scalar contractions for the three Appendix D processing notebooks.
# Vacuum subtraction and source/sink reconstruction follow processData.ipynb.
OPS = ["g;0,0,0;G1g;a;l1;p", "g;0,0,0;G1g;N0sgm0,a;l1;p,sgm"]
BWZ = {"B3pt", "W3pt", "Z3pt"}
DIRECT = {"N-jPi", "N-jPf", "N-j-pi0i", "N-pi0f-j"}


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_scalar_contractions(path, block_size, separations):
    """Read only needed operators/current, preserving raw configuration order."""
    data = {"2pt": {}, "3pt": {}, "VEV": {"j": {}, "pi0f": {}}}
    with h5py.File(path) as source:
        cfgs = source["cfgs"].asstr()[:].tolist()
        if len(set(cfgs)) != len(cfgs):
            raise ValueError("Duplicate configuration IDs")
        for diagram, group in source["diags"].items():
            pairs = group["opabs"].asstr()[:]
            selected = []
            for index, pair in enumerate(pairs):
                sink, origin = [op.split(";") for op in pair.split("_")]
                if (all(op[0] == "g" and op[1] == "0,0,0" and op[2] == "G1g"
                        and op[3] in {"a", "N0sgm0,a"} for op in [sink, origin])
                        and sink[-2] == origin[-2] and sink[-2] in {"l1", "l2"}):
                    selected.append(index)
            if not selected:
                continue
            for flavor, dataset in group["data"].items():
                fields = flavor.split("_")
                three_point = "_deltat_" in flavor
                if three_point and (fields[1] != "j+" or int(fields[-1]) not in separations):
                    continue
                if dataset.shape[0] != len(cfgs):
                    raise ValueError(f"Configuration count mismatch: {dataset.name}")
                if three_point:
                    scalar = group["inserts"].asstr()[:].tolist().index("id")
                    samples = yu.jackknife(dataset[:, :, selected, scalar], d=block_size)
                    sink_flavor, current, source_flavor, _, tf = fields
                    if samples.shape[1] != int(tf) + 1:
                        raise ValueError(f"Insertion-time extent mismatch: {dataset.name}")
                else:
                    samples = yu.jackknife(dataset[:, :, selected], d=block_size)
                    sink_flavor, source_flavor = fields
                for index, pair in enumerate(pairs[selected]):
                    sink, origin = pair.split("_")
                    sink = sink.rsplit(";", 1)[0] + ";" + sink_flavor
                    origin = origin.rsplit(";", 1)[0] + ";" + source_flavor
                    target = data["3pt" if three_point else "2pt"].setdefault(sink + "_" + origin, {})
                    if three_point:
                        target = target.setdefault(f"id_{current}_{tf}", {})
                    if diagram in target:
                        raise ValueError(f"Duplicate operator/diagram: {pair}, {flavor}, {diagram}")
                    target[diagram] = samples[:, :, index]
        data["VEV"]["pi0f"]["sgm"] = yu.jackknife(source["VEV/pi0f/data/sgm"][:], d=block_size)
        # The original processing fixes the VEV gamma ordering with id first.
        data["VEV"]["j"]["id_j+"] = yu.jackknife(source["VEV/j/data/j+"][:, 0], d=block_size)
    return data, cfgs


class ScalarContractions:
    """Vacuum-subtracted rest-frame scalar matrices, including missing partners."""

    def __init__(self, data, sample_count, time_extent, nucleon_connected=None):
        self.data = data
        self.sample_count = sample_count
        self.time_extent = time_extent
        self.nucleon_connected = nucleon_connected

    @staticmethod
    def remove_sigma(operator):
        fields = operator.split(";")
        if fields[3] == "N0sgm0,a" and fields[-1] == "p,sgm":
            fields[3], fields[-1] = "a", "p"
            return ";".join(fields)
        raise ValueError(f"Not a rest-frame N-sigma operator: {operator}")

    def two_diagram(self, sink, origin, diagram):
        terms = self.data["2pt"].get(sink + "_" + origin, {})
        if diagram not in terms:
            return 0
        result = terms[diagram].copy()
        parts = diagram.split("-")
        vev = self.data["VEV"]["pi0f"]["sgm"][:, None]
        if "pi0f" in parts:
            reduced = "-".join(p for p in parts if p != "pi0f")
            result -= self.data["2pt"][self.remove_sigma(sink) + "_" + origin][reduced] * vev
        if "pi0i" in parts:
            reduced = "-".join(p for p in parts if p != "pi0i")
            result -= self.data["2pt"][sink + "_" + self.remove_sigma(origin)][reduced] * vev.conj()
        if "pi0f" in parts and "pi0i" in parts:
            reduced = "-".join(p for p in parts if p not in {"pi0f", "pi0i"})
            pair = self.remove_sigma(sink) + "_" + self.remove_sigma(origin)
            result += self.data["2pt"][pair][reduced] * vev * vev.conj()
        return result

    def two(self, sink, origin, diagrams):
        result = np.zeros((self.sample_count, self.time_extent), dtype=complex)
        result += np.sum([self.two_diagram(sink, origin, diagram)
                          for diagram in self.data["2pt"].get(sink + "_" + origin, {})
                          if diagram in diagrams], axis=0)
        result += np.conj(np.sum([self.two_diagram(origin, sink, diagram)
                                 for diagram in self.data["2pt"].get(origin + "_" + sink, {})
                                 if diagram in diagrams and diagram in {"T", "T-pi0f"}], axis=0))
        return result

    def two_matrix(self, diagrams=yn.diags_all):
        matrices = []
        for ops in [OPS, [yn.op_flipl(op) for op in OPS]]:
            matrix = np.transpose([[self.two(a, b, diagrams) for b in ops] for a in ops], (2, 3, 0, 1))
            matrices.append((matrix + matrix.swapaxes(2, 3).conj()) / 2)
        return ((matrices[0] + matrices[1].conj()) / 2).real

    def three_diagram(self, sink, origin, insertion, diagram):
        if (diagram == "NJN" and self.nucleon_connected is not None
                and sink.split(";")[3] == origin.split(";")[3] == "a"):
            return self.nucleon_connected[int(insertion.split("_")[-1])]
        terms = self.data["3pt"].get(sink + "_" + origin, {}).get(insertion, {})
        if diagram not in terms:
            return 0
        result = terms[diagram].copy()
        parts = diagram.split("-")
        vev = self.data["VEV"]["pi0f"]["sgm"][:, None]
        if "pi0f" in parts:
            reduced = "-".join(p for p in parts if p != "pi0f")
            pair = self.remove_sigma(sink) + "_" + origin
            result -= self.data["3pt"][pair][insertion][reduced] * vev
        if "pi0i" in parts:
            reduced = "-".join(p for p in parts if p != "pi0i")
            pair = sink + "_" + self.remove_sigma(origin)
            result -= self.data["3pt"][pair][insertion][reduced] * vev.conj()
        if "pi0f" in parts and "pi0i" in parts:
            reduced = "-".join(p for p in parts if p not in {"pi0f", "pi0i"})
            pair = self.remove_sigma(sink) + "_" + self.remove_sigma(origin)
            result += self.data["3pt"][pair][insertion][reduced] * vev * vev.conj()
        if "j" in parts:
            reduced = "-".join(p for p in parts if p != "j")
            tf = int(insertion.split("_")[-1])
            result -= (self.two_diagram(sink, origin, reduced)[:, tf]
                       * self.data["VEV"]["j"]["id_j+"])[:, None]
        return result

    def three(self, sink, origin, insertion, diagrams):
        result = np.zeros((self.sample_count, int(insertion.split("_")[-1]) + 1), dtype=complex)
        forward = self.data["3pt"].get(sink + "_" + origin, {}).get(insertion, {})
        reverse = self.data["3pt"].get(origin + "_" + sink, {}).get(insertion, {})
        partners = {"B3pt", "W3pt", "Z3pt", "T-j", "T-pi0f-j", "T-jPf",
                    "B3pt-pi0f", "W3pt-pi0f", "Z3pt-pi0f"}
        if "NJN-pi0f" not in forward:
            partners.add("NJN-pi0i")
        result += np.sum([self.three_diagram(sink, origin, insertion, diagram)
                          for diagram in forward if diagram in diagrams], axis=0)
        partner = np.zeros_like(result) + np.sum([
            self.three_diagram(origin, sink, insertion, diagram)
            for diagram in reverse if diagram in diagrams and diagram in partners], axis=0)
        result += partner[:, ::-1].conj()  # gtCj['id'] = +1
        return result

    def three_matrix(self, tf, diagrams=yn.diags_all):
        matrices = []
        for ops in [OPS, [yn.op_flipl(op) for op in OPS]]:
            matrix = np.transpose([[self.three(a, b, f"id_j+_{tf}", diagrams)
                                    for b in ops] for a in ops], (2, 3, 0, 1))
            matrices.append((matrix + matrix[:, ::-1].swapaxes(2, 3).conj()) / 2)
        # Both operators have the same row: row-sign product and fourCPTstar['id'] are +1.
        return (matrices[0] + matrices[1].conj()) / 2


def compact_topology_data(raw_path, reference_path, block_size, time_extent):
    """Validate against production matrices, then retain only reusable small arrays."""
    saved = yu.load_pkl(reference_path)
    if saved is None:
        raise FileNotFoundError(reference_path)
    c2, c3, matched = saved[:3]
    data, cfgs = read_scalar_contractions(raw_path, block_size, c3)
    if len(cfgs) // block_size != len(c2):
        raise ValueError("Blocking differs from production")
    connected = {tf: saved[3][tf][:, :, 0, 0].real for tf in c3} if len(saved) > 3 else None
    contractions = ScalarContractions(data, len(c2), time_extent, connected)
    np.testing.assert_allclose(contractions.two_matrix(), c2, rtol=1e-9, atol=2e-12)
    subsets = {"bwz": {}, "direct": {}}
    for tf, full in c3.items():
        np.testing.assert_allclose(contractions.three_matrix(tf), full, rtol=1e-8, atol=2e-12)
        for name, diagrams in [("bwz", BWZ), ("direct", DIRECT)]:
            present = set().union(*(set(v.get(f"id_j+_{tf}", {})) for v in data["3pt"].values()))
            if not diagrams <= present:
                raise ValueError(f"Missing {name} diagrams at ts={tf}: {diagrams - present}")
            subset = contractions.three_matrix(tf, diagrams)
            np.testing.assert_array_equal(subset[:, :, 0, 0], 0)
            np.testing.assert_allclose(subset[:, :, 0, 1], subset[:, ::-1, 1, 0].conj(), rtol=1e-10, atol=1e-12)
            subsets[name][tf] = subset
    metadata = {
        "schema": 1, "ensemble": reference_path.parents[3].name,
        "raw_filename": raw_path.name, "raw_sha256": file_sha256(raw_path),
        "reference_sha256": file_sha256(reference_path),
        "processor_sha256": file_sha256(__file__),
        "configurations": cfgs, "block_size": block_size,
        "used_configurations": len(c2) * block_size,
        "discarded_tail": cfgs[len(c2) * block_size:],
        "operators": OPS, "separations": sorted(c3),
        "diagram_sets": {"bwz": sorted(BWZ), "direct": sorted(DIRECT)},
    }
    return {"metadata": metadata, "c2": c2, "c3": c3,
            "matched_c2": matched, **subsets}
