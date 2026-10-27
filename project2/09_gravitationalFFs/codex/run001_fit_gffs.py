"""run001: fit bare A20, B20 and C20 directly to plateau matrix elements.

    python3 -u run001_fit_gffs.py --input /p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs/cB211.072.64/run001_gluon_plateau/matrix_elements.h5

The diagonal and off-diagonal sectors are fitted independently. Every retained
oriented row uses its actual Q2; there is no per-Q2 rank selection or exchange
average. Unchanged 05_moments utility functions perform the mean and paired
jackknife fits with fixed diagonal weights. This avoids a singular/large dense
matrix-element covariance while preserving correlations in propagated errors.
K and Q2 are central values; the input plateau samples already propagate C2 and
energy-fit errors. Results are bare: D=4*C20 is a convention conversion only.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import curve_fit, leastsq


HERE = Path(__file__).resolve().parent
UTILITY = HERE.parents[1]/"05_moments/util.py"
OUTPUT = Path("/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs/cB211.072.64/run001_gluon_plateau")
MODELS = {"zlinear": 1, "zquadratic": 2}
SECTORS = ("equal", "unequal")
COLORS = {"equal": "tab:blue", "unequal": "tab:orange"}
SECTOR_LABELS = {"equal": r"$\mu=\nu$", "unequal": r"$\mu\ne\nu$"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_array(dataset):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]])


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_utility(path):
    """Load the original helper bodies without notebook setup or filesystem writes."""
    source = Path(path).read_text()
    wanted = ("jackme", "jackmec", "jackmap", "jackfit", "doSimpleFit")
    found = {n.name: n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.FunctionDef) and n.name in wanted}
    if set(found) != set(wanted):
        raise ValueError("Missing required 05_moments fitting/resampling helper")
    module = ast.Module(body=[found[name] for name in wanted], type_ignores=[])
    namespace = dict(np=np, warnings=warnings, cholesky=cholesky,
                     curve_fit=curve_fit, leastsq=leastsq, flag_fast=False)
    exec(compile(module, str(path), "exec"), namespace)
    signature = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    return SimpleNamespace(**{name: namespace[name] for name in wanted}), signature


def z_basis(q2, order, tcut):
    root = np.sqrt(tcut+np.asarray(q2))
    z = (root-np.sqrt(tcut))/(root+np.sqrt(tcut))
    return z[:, None]**np.arange(order+1)[None, :]


def design_matrix(k, q2, order, tcut):
    return (k[:, :, None]*z_basis(q2, order, tcut)[:, None, :]).reshape(len(q2), -1)


def identified_statistics(samples, identified, utility):
    mean, error = np.full(samples.shape[1], np.nan), np.full(samples.shape[1], np.nan)
    if np.any(identified):
        mean[identified], error[identified] = utility.jackme(samples[:, identified])
    return mean, error


def fit_sector(y_jk, k, q2, order, tcut, utility, grid):
    mean, error = utility.jackme(y_jk)
    if np.any(~np.isfinite(y_jk)) or np.any(~np.isfinite(error)) or np.any(error <= 0):
        raise ValueError("Selected plateau rows must have finite samples and positive errors")
    design = design_matrix(k, q2, order, tcut)
    weighted = design/error[:, None]
    _, singular, vh = np.linalg.svd(weighted, full_matrices=len(mean) < weighted.shape[1])
    tolerance = (singular[0] if len(singular) else 0)*max(weighted.shape)*np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    npar = design.shape[1]
    null = vh[rank:]
    # Only rank-deficient models are reparameterized. Null directions are not
    # measured zeros: report/evaluate a functional only if it annihilates null.
    basis = np.eye(npar) if rank == npar else vh[:rank].T
    reduced = design@basis
    started = time.monotonic()
    if rank:
        xs = np.arange(len(mean), dtype=float)

        def model(_xs, *pars):
            return reduced@np.asarray(pars)

        def jacobian(_xs, *pars):
            return reduced

        def fit_one(values, start):
            return utility.doSimpleFit(model, xs, values, sigma=error, p0=start,
                                       jackQ=False, jac=jacobian, maxfev=2000,
                                       ftol=1e-10, xtol=1e-10)

        mean_fit = fit_one(mean, np.zeros(rank))
        central_beta = mean_fit[0]

        def fit_sample(values):
            result = fit_one(values, central_beta)
            return result[0], result[3]

        beta_jk, chi2_jk = utility.jackmap(fit_sample, y_jk)
        if np.any(~np.isfinite(beta_jk)):
            raise ValueError("Fitting utility returned nonfinite parameter samples")
        prediction = reduced@central_beta
    else:
        central_beta = np.empty(0)
        beta_jk = np.empty((len(y_jk), 0))
        prediction = np.zeros_like(mean)
        chi2_jk = np.sum((y_jk/error)**2, axis=1)
    parameters = beta_jk@basis.T
    identifiable = np.linalg.norm(null, axis=0) < 1e-9 if len(null) else np.ones(npar, bool)
    parameter_mean, parameter_error = identified_statistics(parameters, identifiable, utility)
    parameter_covariance = np.full((npar, npar), np.nan)
    if np.any(identifiable):
        _, _, covariance = utility.jackmec(parameters[:, identifiable])
        parameter_covariance[np.ix_(identifiable, identifiable)] = covariance
    curves = np.zeros((3, len(grid), npar))
    for ff in range(3):
        curves[ff, :, ff*(order+1):(ff+1)*(order+1)] = z_basis(grid, order, tcut)
    functionals = curves.reshape(-1, npar)
    curve_identified = (np.linalg.norm(functionals@null.T, axis=1) < 1e-9
                        if len(null) else np.ones(len(functionals), bool))
    curve_jk = beta_jk@(functionals@basis).T
    curve_mean, curve_error = identified_statistics(curve_jk, curve_identified, utility)
    parameters[:, ~identifiable] = np.nan
    curve_jk[:, ~curve_identified] = np.nan
    cindex = 2*(order+1)
    cmean, cerror = parameter_mean[cindex], parameter_error[cindex]
    chi2 = float(np.sum(((prediction-mean)/error)**2))
    return dict(
        parameter_names=[f"{ff}{i}" for ff in ("A", "B", "C") for i in range(order+1)],
        parameter_jk=parameters, parameter_mean=parameter_mean, parameter_error=parameter_error,
        parameter_covariance=parameter_covariance, parameter_identified=identifiable,
        beta_jk=beta_jk, parameter_basis=basis,
        singular_values=singular, rank=rank, rank_tolerance=tolerance,
        condition=float(singular[0]/singular[-1]) if rank == npar else np.inf,
        ndata=len(mean), nparameters=npar, fitted_parameters=rank, dof=len(mean)-rank,
        chi2=chi2, chi2_jk=chi2_jk, prediction=prediction, residual=mean-prediction,
        pull=(mean-prediction)/error, matrix_element_mean=mean, matrix_element_error=error,
        curve_jk=curve_jk.reshape(len(y_jk), 3, len(grid)),
        curve_mean=curve_mean.reshape(3, len(grid)), curve_error=curve_error.reshape(3, len(grid)),
        curve_identified=curve_identified.reshape(3, len(grid)),
        C20_0=dict(status="estimated" if identifiable[cindex] else "not_identifiable",
                   mean=cmean, error=cerror,
                   two_error_interval_contains_zero=bool(abs(cmean) <= 2*cerror)
                   if identifiable[cindex] else None),
        D_0=dict(convention="bare D=4*C20; no renormalization/mixing applied",
                 mean=4*cmean, error=4*cerror), fit_seconds=time.monotonic()-started,
    )


def write_fit(group, result, indices, q2, grid):
    for name in ("parameter_jk", "parameter_mean", "parameter_error", "parameter_covariance", "parameter_identified",
                 "beta_jk", "parameter_basis", "singular_values", "chi2_jk", "prediction",
                 "residual", "pull", "matrix_element_mean", "matrix_element_error",
                 "curve_jk", "curve_mean", "curve_error", "curve_identified"):
        group.create_dataset(name, data=result[name], compression="gzip" if np.size(result[name]) else None)
    group.create_dataset("parameter_names", data=np.asarray(result["parameter_names"], dtype="S"))
    group.create_dataset("row_indices", data=indices)
    group.create_dataset("Q2", data=q2)
    group.create_dataset("grid_Q2", data=grid)
    for key in ("rank", "rank_tolerance", "condition", "ndata", "nparameters",
                "fitted_parameters", "dof", "chi2", "fit_seconds"):
        group.attrs[key] = result[key]


def plot_results(results, selection, outdir, dpi):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 12, "axes.linewidth": 1.5, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for model, sectors in results.items():
        for sector, result in sectors.items():
            style = "-" if model == "zlinear" else "--"
            label = SECTOR_LABELS[sector]+(" linear $z$" if model == "zlinear" else " quadratic $z$")
            for ff, ax in enumerate(axes):
                x, mean, error = result["grid"], result["curve_mean"][ff], result["curve_error"][ff]
                if not np.any(np.isfinite(mean)):
                    continue
                ax.plot(x, mean, color=COLORS[sector], linestyle=style, marker="", label=label)
                ax.fill_between(x, mean-error, mean+error, color=COLORS[sector],
                                alpha=.16 if model == "zlinear" else .075, linewidth=0)
    for ff, ax in enumerate(axes):
        ax.set_xlabel(r"$Q^2\;[\mathrm{GeV}^2]$")
        ax.set_ylabel(r"$"+("A", "B", "C")[ff]+r"_{20}^{g,\mathrm{bare}}(Q^2)$")
        if ff and ax.get_ylim()[0] <= 0 <= ax.get_ylim()[1]:
            ax.axhline(0, color="0.5", linewidth=.8, linestyle=":", marker="")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False)
    fig.tight_layout()
    filename = outdir/f"bare_gffs_{selection}.png"
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    primary = results.get("zlinear", next(iter(results.values()), {}))
    for ax, sector in zip(axes, SECTORS):
        if sector in primary:
            result = primary[sector]
            ax.scatter(result["q2"], result["pull"], s=10, color=COLORS[sector],
                       alpha=.4, label=SECTOR_LABELS[sector], rasterized=True)
            ax.legend(frameon=False)
        ax.axhline(0, color="0.5", linewidth=.8, linestyle=":", marker="")
        ax.set_xlabel(r"$Q^2\;[\mathrm{GeV}^2]$")
        ax.set_ylabel(r"$(M_{\mathrm{plateau}}-M_{\mathrm{fit}})/\sigma_M$")
    fig.tight_layout()
    residual_name = outdir/f"me_residuals_{selection}.png"
    fig.savefig(residual_name, dpi=dpi)
    plt.close(fig)
    return [str(filename), str(residual_name)]


def run(args):
    started = time.monotonic()
    utility, signature = load_utility(args.util)
    models, selections = args.models.split(","), args.selections.split(",")
    if not models or any(m not in MODELS for m in models) or len(set(models)) != len(models):
        raise ValueError("Select distinct models from zlinear,zquadratic")
    if (not selections or len(set(selections)) != len(selections)
            or any(not s or "/" in s or "\\" in s or s in (".", "..") for s in selections)):
        raise ValueError("Selections must be plain HDF5 group names")
    if (not np.isfinite(args.q2_max) or not np.isfinite(args.mpi_gev)
            or args.q2_max <= 0 or args.mpi_gev <= 0 or args.grid_points < 2 or args.dpi < 1):
        raise ValueError("Positive Q2 cutoff/pion mass and at least two grid points are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tcut = 4*args.mpi_gev**2
    report = dict(input=str(args.input.resolve()), input_sha256=digest(args.input),
        producer_sha256=digest(__file__), utility=str(args.util.resolve()),
        utility_sha256=digest(args.util), loaded_utility_ast_sha256=signature,
        settings=dict(models=models, selections=selections, q2_max=args.q2_max,
                      tcut=tcut, t0=0, mpi_gev=args.mpi_gev, priors=[],
                      weighting="fixed diagonal errors of plateau means; paired JK propagation",
                      kinematics="central K and Q2; plateau JK includes C2/energy errors",
                      convention="bare A20,B20,C20; D=4*C20; no renormalization/mixing",
                      covariance_note="chi2/dof is descriptive for diagonal weighting of correlated MEs"),
        fits=[], C20_0_table=[], plots=[])
    path = args.output_dir/"stage2_bare_gffs.h5"
    if args.input.resolve() == path.resolve():
        raise ValueError("The input must not be the stage2 output file")
    with h5py.File(args.input, "r") as source, h5py.File(path, "w") as output:
        schema = source.attrs.get("schema", "")
        schema = schema.decode() if isinstance(schema, bytes) else schema
        if schema != "gff-plateau-v1":
            raise ValueError(f"Unexpected input schema: {schema}")
        if "complete" in source.attrs and not source.attrs["complete"]:
            raise ValueError("The input plateau file is not marked complete")
        report["input_attributes"] = json_ready(dict(source.attrs))
        output.attrs.update(schema="gff-bare-global-fit-v1", complete=False)
        for key in ("moms", "proj", "insert", "sector", "phase", "kinematic_null", "K", "Q2", "cfgs"):
            if key in source:
                source.copy(key, output)
        sectors = text_array(source["sector"])
        k, q2 = source["K"][:], source["Q2"][:]
        null = source["kinematic_null"][:].astype(bool)
        if k.shape != (len(q2), 3) or sectors.shape != q2.shape or null.shape != q2.shape:
            raise ValueError("Inconsistent row metadata shapes")
        if np.any(~np.isfinite(k)) or np.any(~np.isfinite(q2)) or np.any(q2 < -1e-12):
            raise ValueError("K/Q2 must be finite, with nonnegative Q2")
        if not set(sectors).issubset(SECTORS):
            raise ValueError("Sector labels must be equal or unequal")
        q2 = np.maximum(q2, 0)
        njk = None
        for selection in selections:
            group = source[f"selections/{selection}"]
            y_jk, ok = group["plateau_jk"][:], group["fit_ok"][:].astype(bool)
            if y_jk.ndim != 2 or y_jk.shape[1] != len(q2) or ok.shape != q2.shape:
                raise ValueError("Inconsistent plateau sample/fit-status shapes")
            if np.iscomplexobj(y_jk):
                raise ValueError("plateau_jk must contain the real signed scalar amplitude matching K")
            if len(y_jk) < 3:
                raise ValueError("At least three paired jackknife samples are required")
            if njk is not None and len(y_jk) != njk:
                raise ValueError("Selections must retain the same paired jackknife sample count")
            njk = len(y_jk)
            if "cfgs" in source and len(source["cfgs"]) != njk:
                raise ValueError("Configuration metadata and jackknife sample counts differ")
            report["jackknife_samples"] = njk
            results = {model: {} for model in models}
            for sector in SECTORS:
                indices = np.flatnonzero(ok & ~null & (sectors == sector) & (q2 <= args.q2_max))
                if not len(indices):
                    report["fits"].append(dict(selection=selection, sector=sector, status="no_rows"))
                    continue
                grid = np.linspace(0, float(q2[indices].max()), args.grid_points)
                for model in models:
                    print(f"selection={selection} sector={sector} model={model} rows={len(indices)}", flush=True)
                    result = fit_sector(y_jk[:, indices], k[indices], q2[indices],
                                        MODELS[model], tcut, utility, grid)
                    result.update(grid=grid, q2=q2[indices])
                    results[model][sector] = result
                    write_fit(output.create_group(f"selections/{selection}/{sector}/{model}"),
                              result, indices, q2[indices], grid)
                    summary = {key: result[key] for key in ("rank", "condition", "ndata", "nparameters",
                        "fitted_parameters", "dof", "chi2", "fit_seconds", "C20_0", "D_0")}
                    report["fits"].append(dict(selection=selection, sector=sector, model=model,
                        status="full_rank" if result["rank"] == result["nparameters"] else "rank_deficient",
                        selection_attributes=json_ready(dict(group.attrs)), **summary))
                    report["C20_0_table"].append(dict(selection=selection, sector=sector, model=model,
                                                    **result["C20_0"], D_0=result["D_0"]))
            report["plots"] += plot_results(results, selection, args.output_dir, args.dpi)
        report["elapsed_seconds"] = time.monotonic()-started
        output.create_dataset("metadata/report", data=json.dumps(json_ready(report), allow_nan=False))
        output.create_dataset("metadata/producer_source", data=Path(__file__).read_text())
        output.attrs["complete"] = True
    report_path = args.output_dir/"stage2_bare_gffs.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(json_ready(dict(output=str(path), report=str(report_path),
                                    C20_0=report["C20_0_table"])), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--selections", default="baseline")
    parser.add_argument("--models", default="zlinear,zquadratic")
    parser.add_argument("--q2-max", type=float, default=1.0)
    parser.add_argument("--mpi-gev", type=float, default=.1402)
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--util", type=Path, default=UTILITY)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
