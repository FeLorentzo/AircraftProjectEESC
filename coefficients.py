import numpy as np

from dataclasses import dataclass, field


@dataclass
class SurfaceCoefficients:
    """
    surface coefficients non-dimensionalized by local surface parameters,
    moments around root quarter chord
    """
    CL: float = np.nan
    CD: float = np.nan
    CM: float = np.nan
    CL_alpha: float = np.nan
    CD_alpha: float = np.nan
    CM_alpha: float = np.nan
    CL_delta: float = 0.0
    CD_delta: float = 0.0
    CM_delta: float = 0.0
    CL_dincid: float = 0.0
    CD_dincid: float = 0.0
    CM_dincid: float = 0.0
    

@dataclass
class AircraftCoefficients:
    """
    aircraft coefficients non-dimensionalized by wing parameters
    moments around cg
    """
    wing: SurfaceCoefficients = field(init=False)
    hstab: SurfaceCoefficients = field(init=False)
    vstab: SurfaceCoefficients = field(init=False)
    fuselage: SurfaceCoefficients = field(init=False)
    CM_motor: float = np.nan
    CD_fus: float = np.nan
    CM_fus: float = np.nan
    CM_alpha_fus: float = np.nan
    CL: float = np.nan
    CD: float = np.nan
    CM: float = np.nan
    CL_alpha: float = np.nan
    CD_alpha: float = np.nan
    CM_alpha: float = np.nan
    CL_deltah: float = np.nan
    CD_deltah: float = np.nan
    CM_deltah: float = np.nan
    CL_deltae: float = np.nan
    CD_deltae: float = np.nan
    CM_deltae: float = np.nan
    thrust: float = np.nan
    elev_utilization: float = np.nan
    hstab_utilization: float = np.nan
    static_margin: float = np.nan
    
    def __post_init__(self):
        wing = SurfaceCoefficients()
        hstab = SurfaceCoefficients()
        vstab = SurfaceCoefficients()
        fuselage = SurfaceCoefficients()