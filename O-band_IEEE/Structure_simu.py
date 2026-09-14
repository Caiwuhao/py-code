import numpy as np

# ============================================================
# 1. Bulk-material refractive-index models
#    Wavelength lambda is in micrometers (um)
#    Based on Appendix A of Xie et al., JSTQE 2018
# ============================================================

def n_AlP(lam):
    """
    AlP
    Valid approximately for 0.444 um < lambda < 2.49 um
    """
    return (
        -0.005002 * lam**(-4)
        + 0.04304 * lam**(-3)
        + 0.02511 * lam**(-2)
        - 0.08136 * lam**(-1)
        + 2.782
    )


def n_InP(lam):
    """
    InP
    Piecewise formulas given in Appendix A
    """
    if 0.4275 < lam < 0.8266:
        return (
            0.1991 * lam**(-4)
            - 1.079 * lam**(-3)
            + 2.634 * lam**(-2)
            - 3.003 * lam**(-1)
            + 4.719
        )

    elif 0.95 < lam < 10:
        n2_minus_1 = (
            6.225
            + 2.316 * lam**2 / (lam**2 - 0.6232**2)
            + 2.765 * lam**2 / (lam**2 - 32.935**2)
        )
        return np.sqrt(1 + n2_minus_1)

    else:
        raise ValueError("lambda is outside the InP fitting range")


def n_GaP(lam):
    """
    GaP
    Valid approximately for 0.5 um < lambda < 4 um
    """
    return (
        -0.09504 * lam**(-4)
        + 0.3917 * lam**(-3)
        - 0.412 * lam**(-2)
        + 0.2646 * lam**(-1)
        + 2.966
    )


def n_AlAs(lam):
    """
    AlAs
    Valid approximately for 0.56 um < lambda < 2.2 um
    """
    n2_minus_1 = (
        1.0792
        + 6.0840 * lam**2 / (lam**2 - 0.2822**2)
        + 1.9 * lam**2 / (lam**2 - 27.62**2)
    )

    return np.sqrt(1 + n2_minus_1)


def n_GaAs(lam):
    """
    GaAs
    Piecewise formulas from Appendix A
    """
    if 0.44 < lam < 0.84:
        return (
            1.148 * lam**(-4)
            - 6.58 * lam**(-3)
            + 14.47 * lam**(-2)
            - 13.91 * lam**(-1)
            + 8.491
        )

    elif 0.97 < lam < 2.73:
        return (
            0.3359 * lam**(-4)
            - 0.6715 * lam**(-3)
            + 0.7089 * lam**(-2)
            - 0.1857 * lam**(-1)
            + 3.317
        )

    else:
        raise ValueError("lambda is outside the GaAs fitting range")


# ============================================================
# 2. Alloy interpolation
# ============================================================

def n_AlGaAs(lam, x_Al):
    """
    Al_x Ga_(1-x) As

    n = x*n_AlAs + (1-x)*n_GaAs
    """
    return (
        x_Al * n_AlAs(lam)
        + (1 - x_Al) * n_GaAs(lam)
    )


def n_GaInP(lam, x_Ga):
    """
    Ga_x In_(1-x) P

    n = x*n_GaP + (1-x)*n_InP
    """
    return (
        x_Ga * n_GaP(lam)
        + (1 - x_Ga) * n_InP(lam)
    )


def n_AlGaInP(lam, x_Ga, y_Al):
    """
    Al_y Ga_x In_(1-x-y) P

    n = x*n_GaP
        + y*n_AlP
        + (1-x-y)*n_InP
    """
    x_In = 1 - x_Ga - y_Al

    return (
        x_Ga * n_GaP(lam)
        + y_Al * n_AlP(lam)
        + x_In * n_InP(lam)
    )


# ============================================================
# 3. Device parameters
# ============================================================

# Pump wavelength
lambda_p = 0.6538       # um

# Approximate degenerate SPDC wavelength
lambda_s = 2 * lambda_p
lambda_i = lambda_s

# DBR materials
# Al0.95Ga0.05As
n_DBR1 = n_AlGaAs(lambda_p, x_Al=0.95)

# Al0.55Ga0.45As
n_DBR2 = n_AlGaAs(lambda_p, x_Al=0.55)

# Central Al0.14Ga0.36In0.50P core/barrier
n_core_p = n_AlGaInP(
    lambda_p,
    x_Ga=0.36,
    y_Al=0.14
)

n_core_s = n_AlGaInP(
    lambda_s,
    x_Ga=0.36,
    y_Al=0.14
)

# Ga0.41In0.59P quantum well
n_QW_p = n_GaInP(
    lambda_p,
    x_Ga=0.41
)


# ============================================================
# 4. Print material refractive indices
# ============================================================

print("===== Refractive indices =====")

print(f"lambda_p = {lambda_p*1000:.1f} nm")
print(f"lambda_s = lambda_i = {lambda_s*1000:.1f} nm")

print()

print(
    "n[Al0.95Ga0.05As, pump] = "
    f"{n_DBR1:.6f}"
)

print(
    "n[Al0.55Ga0.45As, pump] = "
    f"{n_DBR2:.6f}"
)

print(
    "n[Al0.14Ga0.36In0.50P, pump] = "
    f"{n_core_p:.6f}"
)

print(
    "n[Al0.14Ga0.36In0.50P, signal] = "
    f"{n_core_s:.6f}"
)

print(
    "n[Ga0.41In0.59P, pump] = "
    f"{n_QW_p:.6f}"
)


# ============================================================
# 5. Analytical BRW thickness design
# ============================================================

# The analytical BRW design requires a target propagation
# effective index n_p.
#
# Fig. 6 of the paper gives the Bragg-mode effective index
# around ~3.02-3.03 near 654 nm.
#
# Here we use 3.020 as an initial analytical design value.
#
# IMPORTANT:
# COMSOL later solves the actual eigenmode and gives the
# final n_eff(lambda).
#
n_p = 3.020


def dbr_thickness(lambda_um, n_layer, n_eff):
    """
    Quarter-wave Bragg-reflector thickness

              lambda_p
    t = -------------------------
        4 sqrt(n_layer^2-n_eff^2)

    Output in nm.
    """

    t_um = lambda_um / (
        4 * np.sqrt(n_layer**2 - n_eff**2)
    )

    return t_um * 1000


def core_thickness(lambda_um, n_core, n_eff):
    """
    Half-wave central BRW core thickness

              lambda_p
    tb = -------------------------
         2 sqrt(n_core^2-n_eff^2)

    Output in nm.
    """

    t_um = lambda_um / (
        2 * np.sqrt(n_core**2 - n_eff**2)
    )

    return t_um * 1000


t1 = dbr_thickness(
    lambda_p,
    n_DBR1,
    n_p
)

t2 = dbr_thickness(
    lambda_p,
    n_DBR2,
    n_p
)

tb = core_thickness(
    lambda_p,
    n_core_p,
    n_p
)


print()
print("===== Analytical BRW thickness =====")

print(f"Assumed n_p = {n_p:.6f}")
print(f"t1 = {t1:.2f} nm")
print(f"t2 = {t2:.2f} nm")
print(f"tb = {tb:.2f} nm")


# ============================================================
# 6. Reverse calculation from reported thicknesses
# ============================================================

# Paper-reported values
t1_paper = 199.5e-3      # um
t2_paper = 103.5e-3      # um
tb_paper = 231.0e-3      # um


def effective_index_from_dbr(
        lambda_um,
        n_layer,
        thickness_um
):
    """
    Reverse the quarter-wave BRW equation:

                    lambda
    n_eff = sqrt(n^2 - (--------------)^2)
                          4 t
    """

    return np.sqrt(
        n_layer**2
        - (
            lambda_um
            / (4 * thickness_um)
        )**2
    )


def effective_index_from_core(
        lambda_um,
        n_core,
        thickness_um
):
    """
    Reverse the half-wave core equation:

                    lambda
    n_eff = sqrt(n^2 - (--------------)^2)
                          2 tb
    """

    return np.sqrt(
        n_core**2
        - (
            lambda_um
            / (2 * thickness_um)
        )**2
    )


np_from_t1 = effective_index_from_dbr(
    lambda_p,
    n_DBR1,
    t1_paper
)

np_from_t2 = effective_index_from_dbr(
    lambda_p,
    n_DBR2,
    t2_paper
)

np_from_tb = effective_index_from_core(
    lambda_p,
    n_core_p,
    tb_paper
)


print()
print("===== Reverse calculation =====")

print(
    "n_p inferred from t1 = "
    f"{np_from_t1:.6f}"
)

print(
    "n_p inferred from t2 = "
    f"{np_from_t2:.6f}"
)

print(
    "n_p inferred from tb = "
    f"{np_from_tb:.6f}"
)