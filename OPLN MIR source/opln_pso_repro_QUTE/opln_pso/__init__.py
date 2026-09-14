"""PSO-assisted duty-cycle design for type-II periodically poled LiNbO3."""

from .model import (
    DesignGeometry,
    JSAConfig,
    build_geometry,
    cumulative_erf_target,
    energy_conserving_idler,
    gaussian_pump_parameters,
    group_index,
    material_mismatch,
    pump_amplitude_py_jsi,
    qute_3500_geometry,
    scan_optimal_pump_bandwidth,
    pmf_from_duties,
    qpm_period,
    refractive_index,
    schmidt_purity,
)
from .optimizer import (
    PSOConfig,
    PMFObjective,
    optimize_duty_cycles,
    polish_duty_cycles_lbfgsb,
)

__all__ = [
    "DesignGeometry",
    "JSAConfig",
    "PSOConfig",
    "PMFObjective",
    "build_geometry",
    "cumulative_erf_target",
    "energy_conserving_idler",
    "gaussian_pump_parameters",
    "group_index",
    "material_mismatch",
    "optimize_duty_cycles",
    "polish_duty_cycles_lbfgsb",
    "pmf_from_duties",
    "pump_amplitude_py_jsi",
    "qpm_period",
    "qute_3500_geometry",
    "refractive_index",
    "schmidt_purity",
    "scan_optimal_pump_bandwidth",
]
