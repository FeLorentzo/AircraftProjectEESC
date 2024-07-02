import numpy as np


def polar_from_txt(file_path: str) -> np.ndarray:
    polar_data = []
    with open(file_path, 'r') as file:
        for (i, line) in enumerate(file, 1):
            if i < 9:
                continue
            values = line.split()
            if len(values) < 12:
                continue
            polar_data.append([float(values[0]), float(values[2]), float(values[5]), float(values[8])])
    polar_data = np.array(polar_data)
    return polar_data


def CL_from_alpha(polar: np.ndarray, alpha: float):
    return np.interp(alpha, polar[:, 0], polar[:, 1])


def CD_from_alpha(polar: np.ndarray, alpha: float):
    return np.interp(alpha, polar[:, 0], polar[:, 2])


def CM_from_alpha(polar: np.ndarray, alpha: float):
    return np.interp(alpha, polar[:, 0], polar[:, 3])


if __name__ == "__main__":
    # inputs
    polar_wing = polar_from_txt('arquivos/wing_llt.txt')
    polars_hstab = [
        polar_from_txt('arquivos/elevator-10.txt'),
        polar_from_txt('arquivos/elevator_0.txt'),
        polar_from_txt('arquivos/elevator+10.txt'),
    ]
    polars_vstab = [
        polar_from_txt('arquivos/fin-10.txt'),
        polar_from_txt('arquivos/fin_0.txt'),
        polar_from_txt('arquivos/fin+10.txt'),
    ]

    # wing
    CL0_w = CL_from_alpha(polar_wing, 0.0)
    CD0_w = CD_from_alpha(polar_wing, 0.0)
    CM0_w = CM_from_alpha(polar_wing, 0.0)
    CL_min_w = np.min(polar_wing[:, 1])
    CL_max_w = np.max(polar_wing[:, 1])

    # h_stab
    CL0_h = CL_from_alpha(polars_hstab[1], 0.0)
    CD0_h = CD_from_alpha(polars_hstab[1], 0.0)
    CM0_h = CM_from_alpha(polars_hstab[1], 0.0)
    CL_delta_h = (CL_from_alpha(polars_hstab[2], 0.0) - CL_from_alpha(polars_hstab[0], 0.0))/np.radians(20)
    CD_delta_h = (CD_from_alpha(polars_hstab[2], 0.0) - CD_from_alpha(polars_hstab[1], 0.0))/np.radians(10)
    CM_delta_h = (CM_from_alpha(polars_hstab[2], 0.0) - CM_from_alpha(polars_hstab[0], 0.0))/np.radians(20)

    # v_stab
    CL0_v = CL_from_alpha(polars_vstab[1], 0.0)
    CD0_v = CD_from_alpha(polars_vstab[1], 0.0)
    CM0_v = CM_from_alpha(polars_vstab[1], 0.0)
    CL_delta_v = (CL_from_alpha(polars_vstab[2], 0.0) - CL_from_alpha(polars_vstab[0], 0.0))/np.radians(20)
    CD_delta_v = (CD_from_alpha(polars_vstab[2], 0.0) - CD_from_alpha(polars_vstab[1], 0.0))/np.radians(10)
    CM_delta_v = (CM_from_alpha(polars_vstab[2], 0.0) - CM_from_alpha(polars_vstab[0], 0.0))/np.radians(20)

    print(
        f'\n'
        f'Wing Data:\n'
        f'{CL0_w = :.4f},  {CD0_w = :.4f},  {CM0_w =:.4f},  {CL_min_w   =:.4f},  {CL_max_w   = :.4f}\n'
        f'\n'
        f'HStab Data:\n'
        f'{CL0_h = :.4f},  {CD0_h = :.4f},  {CM0_h = :.4f},  {CL_delta_h = :.4f},  {CD_delta_h = :.4f},  {CM_delta_h = :.4f}\n'
        f'\n'
        f'VStab Data:\n'
        f'{CL0_v = :.4f},  {CD0_v = :.4f},  {CM0_v = :.4f},  {CL_delta_v = :.4f},  {CD_delta_v = :.4f},  {CM_delta_v = :.4f}\n'
    )
