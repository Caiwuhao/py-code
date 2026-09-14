"""PSO-assisted duty-cycle design for type-II periodically poled LiNbO3."""

from .model import (
    DesignGeometry,
    build_geometry,
    cumulative_erf_target,
    energy_conserving_idler,
    gaussian_pump_parameters,
    group_index,
    material_mismatch,
    pmf_from_duties,
    qpm_period,
    refractive_index,
    schmidt_purity,
)
from .optimizer import PSOConfig, PMFObjective, optimize_duty_cycles

__all__ = [
    "DesignGeometry",
    "PSOConfig",
    "PMFObjective",
    "build_geometry",
    "cumulative_erf_target",
    "energy_conserving_idler",
    "gaussian_pump_parameters",
    "group_index",
    "material_mismatch",
    "optimize_duty_cycles",
    "pmf_from_duties",
    "qpm_period",
    "refractive_index",
    "schmidt_purity",
]

