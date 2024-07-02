import numpy as np

from coefficients import SurfaceCoefficients, AircraftCoefficients
from dataclasses import dataclass


@dataclass
class AeroSurface:
    x: float
    z: float
    S: float
    b: float
    c: float
    c_root: float
    c_tip: float
    sweep: float
    CL0: float
    CD0: float
    CM0: float
    CL_max: float = np.inf
    CL_min: float = -np.inf
    CL_delta: float = 0.0
    CD_delta: float = 0.0
    CM_delta: float = 0.0
    oswald: float = 1.0
    base_incid: float = 0.0
    delta_max: float = np.inf
    delta_min: float = -np.inf
    delta_incid_max: float = np.inf
    delta_incid_min: float = -np.inf

    def __post_init__(self):
        self.AR = self.b**2/self.S
        self.taper = self.c_tip/self.c_root

        # Roskam, J. - "Methods for Estimating Stability Derivatives"
        self.CL_alpha = 2 * np.pi * self.AR / (2 + np.sqrt(self.AR**2 * (1 + (np.tan(np.radians(self.sweep)))**2) + 4))

        # Linear Lifting Line Theory
        self.k_CD = 1/(np.pi * self.AR * self.oswald)

        self.alpha0L = -np.degrees(self.CL0/self.CL_alpha)

    def coefficients(self, alpha: float, incid: float = 0.0, delta: float = 0.0) -> SurfaceCoefficients:
        alphar = np.radians(alpha + incid + self.base_incid)
        deltar = np.radians(delta)
        CL = self.CL0 + alphar*self.CL_alpha + self.CL_delta*deltar
        CD = self.CD0 + self.k_CD*CL**2 + self.CD_delta*deltar
        CM = self.CM0 + self.CM_delta*deltar
        CL_alpha = self.CL_alpha
        CD_alpha = 2*self.k_CD*CL*CL_alpha
        CM_alpha = 0.0

        if CL > self.CL_max:
            CL = self.CL_max
        elif CL < self.CL_min:
            CL = self.CL_min

        return SurfaceCoefficients(
            CL=CL,
            CD=CD,
            CM=CM,
            CL_alpha=CL_alpha,
            CD_alpha=CD_alpha,
            CM_alpha=CM_alpha,
            CL_delta=self.CL_delta,
            CD_delta=self.CD_delta,
            CM_delta=self.CM_delta,
            CL_dincid=CL_alpha,
            CD_dincid=CD_alpha,
            CM_dincid=CM_alpha
        )


@dataclass
class Powerplant:
    x: float
    z: float
    T0: float
    dTdV: float
    incid: float

    def calc_thrust(self, throttle: float, vel: float) -> float:
        return max(throttle*self.T0 + vel*self.dTdV, 0.0)


@dataclass
class MassProperties:
    x: float
    z: float
    mass: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float


@dataclass
class Fuselage:
    x: float
    z: float
    # NACA TR 711
    w_f: float
    l_f: float
    k_fus: float


@dataclass
class Airplane:
    cg: MassProperties
    wing: AeroSurface
    hstab: AeroSurface
    vstab: AeroSurface
    fuselage: Fuselage
    motor: Powerplant
    CD_extra: float

    def __post_init__(self):
        self.S = self.wing.S
        self.b = self.wing.b
        self.c = self.wing.c

        # USAF Stability and Control DATCOM
        k_AR = 1/self.wing.AR - 1/(1+self.wing.AR**1.7)
        k_lam = (10 - 3*self.wing.taper)/7
        k_h = (1 - abs(self.hstab.z/self.wing.b)) / (2*(self.hstab.x - self.wing.x)/self.wing.b)**(1/3)
        self.eps_alpha = 4.44*(k_AR*k_lam*k_h*np.sqrt(np.cos(np.radians(self.wing.sweep))))**1.19
        self.eps0 = -self.eps_alpha * self.wing.alpha0L
        pass

    def coefficients(
        self, alpha: float, vel: float, throttle: float, delta_hstab: float, delta_elev: float,
        delta_rudd: float = 0.0, delta_ail: float = 0.0, *, rho: float = 1.0
    ) -> AircraftCoefficients:

        ETA_H = 0.9
        dyn_pressure = 0.5 * rho * vel**2

        alphar = np.radians(alpha)
        epsr = np.radians(self.eps0) + alphar*self.eps_alpha
        cs = np.cos(alphar)
        sn = np.sin(alphar)

        cs_eps = np.cos(epsr)
        sn_eps = np.sin(epsr)
        cs_dw = np.cos(alphar-epsr)
        sn_dw = np.sin(alphar-epsr)
        cs_imot = np.cos(np.radians(self.motor.incid))
        sn_imot = np.sin(np.radians(self.motor.incid))

        dx_w = (self.wing.x - self.cg.x)/self.c
        dz_w = (self.wing.z - self.cg.z)/self.c
        dx_h = (self.hstab.x - self.cg.x)/self.c
        dz_h = (self.hstab.z - self.cg.z)/self.c
        dx_v = (self.vstab.x - self.cg.x)/self.b
        dz_v = (self.vstab.z - self.cg.z)/self.b
        dx_mot = (self.motor.x - self.cg.x)/self.c
        dz_mot = (self.motor.z - self.cg.z)/self.c

        coeffs = AircraftCoefficients()
        coeffs.thrust = self.motor.calc_thrust(throttle, vel)

        coeffs_w = self.wing.coefficients(alpha)  # local to surface
        coeffs.wing = coeffs_w  # contribution to aircraft
        #
        coeffs.wing.CM = (
            -(coeffs_w.CL * cs + coeffs_w.CD * sn) * dx_w +
            (coeffs_w.CD * cs - coeffs_w.CL * sn) * dz_w +
            coeffs_w.CM
        )
        coeffs.wing.CM_alpha = (
            -(coeffs_w.CL_alpha * cs - coeffs_w.CL * sn + coeffs_w.CD_alpha * sn + coeffs_w.CD * cs) * dx_w +
            (-coeffs_w.CL_alpha * sn - coeffs_w.CL * cs + coeffs_w.CD_alpha * cs - coeffs_w.CD * sn) * dz_w +
            coeffs_w.CM_alpha
        )

        coeffs_h = self.hstab.coefficients(alpha, delta_hstab, delta_elev)
        coeffs.hstab = SurfaceCoefficients()
        #
        coeffs.hstab.CL = (coeffs_h.CL * cs_eps - coeffs_h.CD * sn_eps) * (self.hstab.S / self.S) * ETA_H
        coeffs.hstab.CL_alpha = (
            (coeffs_h.CL_alpha * cs_eps - coeffs_h.CD_alpha * sn_eps) * (1 - self.eps_alpha) -
            (coeffs_h.CL * sn_eps + coeffs_h.CD * cs_eps) * self.eps_alpha
        ) * ETA_H * (self.hstab.S / self.S)
        coeffs.hstab.CL_delta = (
            (coeffs_h.CL_delta * cs_eps - coeffs_h.CD_delta * sn_eps) *
            (self.hstab.S / self.S) * ETA_H)
        coeffs.hstab.CL_dincid = (
            (coeffs_h.CL_dincid * cs_eps - coeffs_h.CD_dincid * sn_eps) *
            (self.hstab.S / self.S) * ETA_H)
        #
        coeffs.hstab.CD = (coeffs_h.CD * cs_eps + coeffs_h.CL * sn_eps) * (self.hstab.S / self.S) * ETA_H
        coeffs.hstab.CD_alpha = (
            (coeffs_h.CD_alpha * cs_eps + coeffs_h.CL_alpha * sn_eps) * (1 - self.eps_alpha) -
            (coeffs_h.CD * sn_eps - coeffs_h.CL * cs_eps) * self.eps_alpha
        ) * ETA_H * (self.hstab.S / self.S)
        coeffs.hstab.CD_delta = (
            (coeffs_h.CD_delta * cs_eps + coeffs_h.CL_delta * sn_eps) *
            (self.hstab.S / self.S) * ETA_H)
        coeffs.hstab.CD_dincid = (
            (coeffs_h.CD_dincid * cs_eps + coeffs_h.CL_dincid * sn_eps) *
            (self.hstab.S / self.S) * ETA_H)
        #
        coeffs.hstab.CM = (
            (-coeffs_h.CL * cs_dw - coeffs_h.CD * sn_dw) * dx_h +
            (-coeffs_h.CL * sn_dw + coeffs_h.CD * cs_dw) * dz_h +
            coeffs_h.CM * self.hstab.c / self.c
        ) * (self.hstab.S / self.S) * ETA_H
        coeffs.hstab.CM_alpha = (
            ((-coeffs_h.CL_alpha - coeffs_h.CD) * cs_dw + (coeffs_h.CL - coeffs_h.CD_alpha) * sn_dw) * dx_h +
            ((-coeffs_h.CL_alpha - coeffs_h.CD) * sn_dw + (-coeffs_h.CL + coeffs_h.CD_alpha) * cs_dw) * dz_h +
            coeffs_h.CD_alpha * self.hstab.c / self.c
        ) * (self.hstab.S / self.S) * ETA_H * (1 - self.eps_alpha)
        coeffs.hstab.CM_delta = (
            (-coeffs_h.CL_delta * cs_dw - coeffs_h.CD_delta * sn_dw) * dx_h +
            (-coeffs_h.CL_delta * sn_dw + coeffs_h.CD_delta * cs_dw) * dz_h +
            coeffs_h.CM_delta * self.hstab.c / self.c
        ) * (self.hstab.S / self.S) * ETA_H
        coeffs.hstab.CM_dincid = (
            (-coeffs_h.CL_dincid * cs_dw - coeffs_h.CD_dincid * sn_dw) * dx_h +
            (-coeffs_h.CL_dincid * sn_dw + coeffs_h.CD_dincid * cs_dw) * dz_h +
            coeffs_h.CM_dincid * self.hstab.c / self.c
        ) * (self.hstab.S / self.S) * ETA_H

        coeffs.CM_motor = (
            dx_mot * (-coeffs.thrust * sn_imot) +
            dz_mot * (-coeffs.thrust * cs_imot)
        ) / (self.S * dyn_pressure)

        coeffs.CD_fus = 0.0
        coeffs.CM_alpha_fus = (
            (180/np.pi) * self.fuselage.k_fus * self.fuselage.w_f**2 * self.fuselage.l_f
        ) / (self.c * self.S)
        coeffs.CM_fus = alphar * coeffs.CM_alpha_fus

        coeffs.CL = coeffs.wing.CL + coeffs.hstab.CL
        coeffs.CD = coeffs.wing.CD + coeffs.hstab.CD + self.CD_extra
        coeffs.CM = coeffs.wing.CM + coeffs.hstab.CM + coeffs.CM_motor + coeffs.CM_fus
        coeffs.CL_alpha = coeffs.wing.CL_alpha + coeffs.hstab.CL_alpha
        coeffs.CD_alpha = coeffs.wing.CD_alpha + coeffs.hstab.CD_alpha
        coeffs.CM_alpha = coeffs.wing.CM_alpha + coeffs.hstab.CM_alpha + coeffs.CM_alpha_fus
        coeffs.CL_deltae = coeffs.hstab.CL_delta
        coeffs.CD_deltae = coeffs.hstab.CD_delta
        coeffs.CM_deltae = coeffs.hstab.CM_delta
        coeffs.CL_deltah = coeffs.hstab.CL_dincid
        coeffs.CD_deltah = coeffs.hstab.CD_dincid
        coeffs.CM_deltah = coeffs.hstab.CM_dincid

        coeffs.elev_utilization = delta_elev/self.hstab.delta_max \
            if delta_elev >= 0.0 else delta_elev/self.hstab.delta_min
        coeffs.hstab_utilization = delta_hstab/self.hstab.delta_incid_max \
            if delta_hstab >= 0.0 else delta_hstab/self.hstab.delta_incid_min

        coeffs.static_margin = -coeffs.CM_alpha/coeffs.CL_alpha

        return coeffs

    def trim_coefficients(
        self, alpha: float, vel: float, throttle: float, *, delta_hstab: float | None = None, 
        rho: float = 1.0, num_iter: int = 3) -> AircraftCoefficients:        
        if delta_hstab is None:
            delta_hstab = 0.0
            trim_with_elev = False
        else:
            trim_with_elev = True
            
        delta_elev = 0.0
        coeffs = self.coefficients(alpha, vel, throttle, delta_hstab, delta_elev, rho=rho)
        
        for _ in range(num_iter):
            if trim_with_elev:
                delta_elev -= float(np.degrees(coeffs.CM/coeffs.CM_deltae))
            else:
                delta_hstab -= float(np.degrees(coeffs.CM/coeffs.CM_deltah))
            coeffs = self.coefficients(alpha, vel, throttle, delta_hstab, delta_elev, rho)
            
        return coeffs

if __name__ == "__main__":
    # Output from 'polar_process.py'
    #
    # Wing Data:
    # CL0_w = 0.4516,  CD0_w = 0.0113,  CM0_w =-0.1241,  CL_min_w   =-0.9806,  CL_max_w   = 2.0300

    # HStab Data:
    # CL0_h = 0.0000,  CD0_h = 0.0050,  CM0_h = 0.0000,  coeffs_h.CL_delta = 2.4731,  coeffs_h.CD_delta = 0.0854,  coeffs_h.CM_delta = -1.1932

    # VStab Data:
    # CL0_v = 0.0000,  CD0_v = 0.0051,  CM0_v = 0.0000,  CL_delta_v = 1.5204,  CD_delta_v = 0.0864,  CM_delta_v = -0.8348
    plane = Airplane(
        cg=MassProperties(
            x=-0.138,
            z=0.0,
            mass=45e+3,
            Ixx=0.0,
            Iyy=0.0,
            Izz=0.0,
            Ixz=0.0,
        ),
        wing=AeroSurface(
            x=0.0,
            z=0.0,
            S=42.22,
            b=21.0,
            c=2.07,
            c_root=2.600,
            c_tip=1.421,
            sweep=0.0,
            CL0=0.4516,
            CD0=0.0113,
            CM0=-0.1241,
            oswald=0.97
        ),
        hstab=AeroSurface(
            x=7.0,
            z=2.735,
            S=9.72,
            b=6.5,
            c=1.52,
            c_root=1.832,
            c_tip=1.160,
            sweep=15.5,
            CL0=0.0,
            CD0=0.0050,
            CM0=0.0,
            CL_delta=2.437,
            CD_delta=0.0854,
            CM_delta=-1.1932,
            oswald=0.95,
            base_incid=0.0,
            delta_min=-30,
            delta_max=30,
            delta_incid_min=-12,
            delta_incid_max=8,
        ),
        vstab=AeroSurface(
            x=6.0,
            z=0.95,
            S=6.05,
            b=5.5,
            c=2.2,
            c_root=2.749,
            c_tip=1.649,
            sweep=20.0,
            CL0=0.0,
            CD0=0.0051,
            CM0=0.0,
            CL_delta=1.5204,
            CD_delta=0.0864,
            CM_delta=-0.8348,
            oswald=0.95,
            delta_max=30,
            delta_min=-30,
        ),
        fuselage=Fuselage(
            x=0.0,
            z=0.0,
            w_f=2.3,
            l_f=18.0,
            k_fus=0.03,
        ),
        motor=Powerplant(
            x=-1.0,
            z=0.0,
            T0=20000,
            dTdV=-50,
            incid=0.0
        ),
        CD_extra=0.005
    )

    coeffs_trim = plane.trim_coefficients(5, 100, 1.0, delta_hstab=0.0)
    print(f"elev_use: {100*coeffs_trim.elev_utilization:.2f}%")
    print(f"static_margin: {100*coeffs_trim.static_margin:.2f}%")
    # print(coeffs_trim.CM)
    # for attr_name, attr_value in coeffs_trim.__dict__.items():
    #     if isinstance(attr_value, (SurfaceCoefficients, AircraftCoefficients)):
    #         for key, value in attr_value.__dict__.items():
    #             print(f"{attr_name}.{key} =  {value:.4f}")
    #     else:
    #         print(f"{attr_name} = {attr_value:.4f}")



# derivadas de estab
# dinamica por matriz A
# latero-direcional
# helice

# === === === plots === === ===

# escolha de incid_h == == == ==
# ábaco de trimagem com Hstab
# ábaco de trimagem com Elev

# passeio do cg == == == ==
# UH por xcg
# ME por xcg

