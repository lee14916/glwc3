"""Reproduce Appendix A's ChPT pion-mass corrections with shared uncertainties.

Run with the glwc3 virtual environment. All masses and decay constants below
are in GeV. Generated reports go under __codex_ignore, not into the manuscript.
The formulas follow Appendix A and NST_ChPT.nb. The chiral-limit nucleon mass
is fixed to the paper's 0.8696 GeV, while e1 is derived from the same uncertain
inputs used in every subsequent evaluation.
"""

import argparse
import json
from pathlib import Path

import gvar as gv
import numpy as np
from scipy.optimize import brentq


PROJECT = Path(__file__).resolve().parent
TARGET_MASS = 0.135
CHARGED_PION_MASS = 0.13957
CHIRAL_NUCLEON_MASS = 0.8696
PROTON_MASS = 0.93827

# These are the current manuscript entries, retained only for comparison.
PAPER_TABLE = {
    "A48": {2: ("+3.5(3)", "1.069(7)"),
            3: ("+3.0(3)", "1.056(5)"),
            4: ("+2.9(5)", "1.055(7)")},
    "B64": {2: ("-4.2(2)", "0.927(3)"),
            3: ("-3.6(2)", "0.941(2)"),
            4: ("-3.5(5)", "0.942(6)")},
    "Phen.": {2: ("-3.7(1)", "0.9356"),
              3: ("-3.2(1)", "0.9475(3)"),
              4: ("-3.0(5)", "0.948(5)")},
}


def make_inputs():
    names = ["c1_p2", "c1_p3", "c1_p4", "c2", "c3", "gA", "Fpi",
             "l3bar", "sigma_charged", "mass_A48", "mass_B64"]
    means = np.array([-.74, -1.07, -1.11, 3.13, -5.61, 1.27641,
                      .092215, 3.41, .0590, .1306, .1402])
    errors = np.array([.02, .02, .03, .03, .06, .00056,
                       .000143, .41, .0035, .0004, .0002])
    covariance = np.diag(errors**2)
    correlation = np.array([[1, .18, .58], [.18, 1, -.64], [.58, -.64, 1]])
    covariance[2:5, 2:5] = correlation * np.outer(errors[2:5], errors[2:5])
    return names, means, covariance, dict(zip(names, gv.gvar(means, covariance)))


def sigma_p3(mass, c1, inputs):
    return -4 * c1 * mass**2 - 9 * inputs["gA"]**2 * mass**3 / (
        64 * np.pi * inputs["Fpi"]**2)


def p4_coefficients(mass, inputs, nucleon_mass):
    c1, c2, c3 = (inputs[name] for name in ("c1_p4", "c2", "c3"))
    axial_squared = inputs["gA"]**2
    decay_squared = inputs["Fpi"]**2
    log_coefficient = -3 * (axial_squared + nucleon_mass * (-8*c1 + c2 + 4*c3)) / (
        64 * np.pi**2 * decay_squared * nucleon_mass)
    coefficient_a = log_coefficient * (4 * gv.log(mass / nucleon_mass) + 1)
    coefficient_b_without_e1 = -3 * (2*axial_squared - c2*nucleon_mass) / (
        128 * np.pi**2 * decay_squared * nucleon_mass)
    coefficient_b_without_e1 += c1 * (inputs["l3bar"] - 1) / (
        16 * np.pi**2 * decay_squared)
    return coefficient_a, coefficient_b_without_e1


def matched_e1(inputs, nucleon_mass=CHIRAL_NUCLEON_MASS):
    # Keep e1 correlated with its matching inputs rather than re-entering its error.
    coefficient_a, coefficient_b = p4_coefficients(
        CHARGED_PION_MASS, inputs, nucleon_mass)
    lower_order = sigma_p3(CHARGED_PION_MASS, inputs["c1_p4"], inputs)
    return ((inputs["sigma_charged"] - lower_order) / CHARGED_PION_MASS**4
            - coefficient_a - 2 * coefficient_b) / 2


def sigma(mass, order, inputs):
    if order == 2:
        return -4 * inputs["c1_p2"] * mass**2
    if order == 3:
        return sigma_p3(mass, inputs["c1_p3"], inputs)
    if order != 4:
        raise ValueError("Order must be 2, 3, or 4.")
    coefficient_a, coefficient_b = p4_coefficients(mass, inputs, CHIRAL_NUCLEON_MASS)
    return (sigma_p3(mass, inputs["c1_p4"], inputs)
            + mass**4 * (coefficient_a + 2 * (matched_e1(inputs) + coefficient_b)))


def matched_nucleon_mass(inputs):
    """Central-only mass matching used in the Mathematica notebook."""
    central = {name: float(gv.mean(value)) for name, value in inputs.items()}
    mass = CHARGED_PION_MASS

    def residual(nucleon_mass):
        c1, c2, c3 = (central[name] for name in ("c1_p4", "c2", "c3"))
        axial_squared = central["gA"]**2
        decay_squared = central["Fpi"]**2
        nucleon = nucleon_mass - 4*c1*mass**2 - 3*axial_squared*mass**3 / (
            32*np.pi*decay_squared)
        nucleon -= 3*(axial_squared + nucleon_mass*(-8*c1 + c2 + 4*c3)) * (
            mass**4 * np.log(mass/nucleon_mass)) / (
                32*np.pi**2*decay_squared*nucleon_mass)
        nucleon += (matched_e1(central, nucleon_mass)
                    - 3*(2*axial_squared - c2*nucleon_mass) / (
                        128*np.pi**2*decay_squared*nucleon_mass)) * mass**4
        return float(nucleon - PROTON_MASS)

    return brentq(residual, .7, 1.0)


def evaluate(inputs):
    values, labels = [], []
    masses = {"A48": inputs["mass_A48"], "B64": inputs["mass_B64"],
              "Phen.": CHARGED_PION_MASS}
    for initial, mass in masses.items():
        for order in (2, 3, 4):
            final_sigma = sigma(TARGET_MASS, order, inputs)
            initial_sigma = sigma(mass, order, inputs)
            labels.extend([f"{initial}_p{order}_shift_MeV", f"{initial}_p{order}_ratio"])
            ratio = (TARGET_MASS / mass)**2 if order == 2 else final_sigma / initial_sigma
            values.extend([1000 * (final_sigma - initial_sigma), ratio])
    return labels, values


def verify(names, means, covariance, inputs, values):
    # Check the complete propagated covariance, not only individual errors.
    jacobian = np.empty((len(values), len(names)))
    for index, mean in enumerate(means):
        step = max(abs(mean), .001) * 1e-5
        upper, lower = means.copy(), means.copy()
        upper[index] += step
        lower[index] -= step
        upper_values = evaluate(dict(zip(names, upper)))[1]
        lower_values = evaluate(dict(zip(names, lower)))[1]
        jacobian[:, index] = (np.asarray(upper_values, dtype=float)
                              - np.asarray(lower_values, dtype=float)) / (2 * step)
    np.testing.assert_allclose(output_covariance(values), jacobian @ covariance @ jacobian.T,
                               rtol=2e-6, atol=1e-12)
    matched_difference = sigma(CHARGED_PION_MASS, 4, inputs) - inputs["sigma_charged"]
    assert abs(gv.mean(matched_difference)) < 1e-14
    assert gv.sdev(matched_difference) < 1e-14
    for initial in ("A48", "B64"):
        mass = inputs[f"mass_{initial}"]
        difference = sigma(TARGET_MASS, 2, inputs) / sigma(mass, 2, inputs)
        difference -= (TARGET_MASS / mass)**2
        assert abs(gv.mean(difference)) < 1e-14 and gv.sdev(difference) < 1e-14


def summary(value):
    return {"mean": float(gv.mean(value)), "error": float(gv.sdev(value)),
            "formatted": str(value)}


def output_covariance(values):
    return gv.evalcov([value if isinstance(value, gv.GVar) else gv.gvar(value, 0)
                       for value in values])


def split_mass_p3_corrections(mass, inputs):
    """Diagnostic of mass correlations lost by early Around substitution."""
    mass_factors = [gv.gvar(gv.mean(mass), gv.sdev(mass)) for _ in range(4)]
    quadratic_mass, cubic_squared_mass, charged_mass, neutral_mass = mass_factors
    c1 = inputs["c1_p3"]
    initial_sigma = -4*c1*quadratic_mass**2 - 3*inputs["gA"]**2 * (
        cubic_squared_mass**2 * (2*charged_mass + neutral_mass)) / (
            64*np.pi*inputs["Fpi"]**2)
    final_sigma = sigma(TARGET_MASS, 3, inputs)
    return 1000*(final_sigma - initial_sigma), final_sigma/initial_sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "__codex_ignore" / "app_a_chpt_codex")
    args = parser.parse_args()
    names, means, covariance, inputs = make_inputs()
    labels, values = evaluate(inputs)
    verify(names, means, covariance, inputs, values)
    matched_mass = matched_nucleon_mass(inputs)
    lines = ["Appendix A: ChPT pion-mass corrections",
             "Shared inputs retain their correlations at every evaluation.",
             "Units: GeV internally; shifts below are in MeV.",
             f"Central mass matching: m_N^(0) = {1000*matched_mass:.6f} MeV",
             f"Fixed value used in Appendix A: {1000*CHIRAL_NUCLEON_MASS:.1f} MeV",
             f"Derived e1 = {matched_e1(inputs)} GeV^-3",
             "",
             "Initial Order   Shift [MeV]       Ratio          Paper shift / ratio"]
    rows = []
    for index in range(0, len(values), 2):
        label = labels[index]
        initial, order_label, _, _ = label.split("_")
        order = int(order_label[1:])
        shift, ratio = values[index:index+2]
        paper_shift, paper_ratio = PAPER_TABLE[initial][order]
        lines.append(f"{initial:7} p{order}     {str(shift):>12}   {str(ratio):>14}"
                     f"   {paper_shift:>8} / {paper_ratio}")
        rows.append({"initial": initial, "order": order,
                     "shift_MeV": summary(shift), "ratio": summary(ratio),
                     "paper_shift": paper_shift, "paper_ratio": paper_ratio})
    diagnostic = {}
    lines.extend(["", "p3 diagnostic: intentionally discard shared-mass correlations",
                  "This explains the old errors; it is NOT the adopted propagation."])
    for initial in ("A48", "B64"):
        shift, ratio = split_mass_p3_corrections(inputs[f"mass_{initial}"], inputs)
        lines.append(f"{initial}: shift {shift} MeV; ratio {ratio}")
        diagnostic[initial] = {"shift_MeV": summary(shift), "ratio": summary(ratio)}
    corrected_phenomenology = inputs["sigma_charged"] + values[-2] / 1000
    lines.extend(["", f"ChPT p4 corrected phenomenology: {1000*corrected_phenomenology} MeV",
                  "This shares the 59.0(3.5) input with the derived e1.",
                  "", "Checks passed: finite-difference covariance, p4 matching, p2 ratio.",
                  "Current-paper errors are comparison entries, not regression targets."])
    report = {"inputs": {name: summary(value) for name, value in inputs.items()},
              "input_names": names, "input_covariance": covariance.tolist(),
              "target_mass_GeV": TARGET_MASS,
              "charged_pion_mass_GeV": CHARGED_PION_MASS,
              "fixed_chiral_nucleon_mass_GeV": CHIRAL_NUCLEON_MASS,
              "central_matched_nucleon_mass_GeV": matched_mass,
              "e1_GeV_minus3": summary(matched_e1(inputs)), "rows": rows,
              "split_mass_p3_diagnostic": diagnostic,
              "output_labels": labels, "output_covariance": output_covariance(values).tolist(),
              "corrected_phenomenology_MeV": summary(1000*corrected_phenomenology),
              "checks_passed": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("results.json").write_text(json.dumps(report, indent=2) + "\n",
                                                        encoding="utf-8")
    args.output_dir.joinpath("results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
