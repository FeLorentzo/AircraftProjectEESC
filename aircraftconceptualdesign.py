import numpy as np

def first_weight_estimate(aircraft: dict) -> float :
    frac_weight = 1
    fracs = {
        'Take off':0.97,
        'Ascending':0.987,
        'Cruise_1':np.exp(-aircraft['missao']['Range_1'] * aircraft["engine"]["SFC_cruise"]/(aircraft["definitions"]['V_cruise'] * (aircraft["definitions"]["Cl cruise"]/aircraft["definitions"]["Cd cruise"]))),
        'Cruise_2':np.exp(-aircraft['missao']['Range_2'] * aircraft["engine"]["SFC_cruise"]/(aircraft["definitions"]['V_cruise'] * (aircraft["definitions"]["Cl cruise"]/aircraft["definitions"]["Cd cruise"]))),
        'Descending': 1,
        'Loitter': np.exp(-aircraft['missao']['Loitter_time'] * aircraft['engine']['SFC_loiter']/(aircraft["definitions"]["Cl cruise"]/aircraft["definitions"]["Cd cruise"])),
        'Landing': 0.995
    }

    print(fracs)

    for leg in aircraft['missao']['Perfil']:
        frac_weight = frac_weight * fracs[leg]

    return frac_weight

def estimate_aerodynamic_areas(aircraft):
    '''
    Estimate areas of aerodynamic surfaces based on the correlation between MTOW and areas of the cargo database
    '''
    # Parameter = MTOW

    x = aircraft['definitions']['MTOW']

    # horizontal tail

    aircraft['ht']['S'] = 4.3E-04*x + 6.03
    aircraft['ht']['S_elevator'] = 3.41E-04*x + 1.16

    aircraft['vt']['S'] = 4.77E-04*x + 2.57
    aircraft['vt']['S_rudder'] = 1.19E-04*x + 1.65

    aircraft['wing']['S_flap'] = 4.43E-04*x + 0.822
    aircraft['wing']['S_aileron'] = 1.8E-04*x + 1.63

    return aircraft


def Re_cutoff(l):

    # Para pintura em alumínio sem tratamento superficial

    k = 1.015e-5

    return 38.21 * (l/k)**(1.053)


def Cf_laminar(Re):
    return 1.328/np.sqrt(Re)


def Cf_turbulent(Re, M):
    return 0.455 / ( (np.log10(Re)**(2.58))*(1+0.144*M**2)**(0.65) )


def compute_FF(aircraft, M):
    '''
    Compute Form Factor drag for aerodynamic surfaces (wing, horizontal stabilizer and vertical stabilier) and fuselage
    ''' 

    for component, properties in aircraft.items():
        if component in ['wing', 'ht', 'vt']:
            x_c_m = aircraft[component]['CMA']
            t_c   = aircraft[component]['thickness']
            Lambda = aircraft[component]['sweep']

            term1 = 1 + ((0.6 / x_c_m) * t_c) + (100 * (t_c ** 4))
            term2 = 1.34 * (M ** 0.18) * (np.cos(np.radians(Lambda)) ** 0.28)
            if properties['hinge']:
                aircraft[component]['FF'] = 1.1 * term1 * term2
            else:
                aircraft[component]['FF'] = term1 * term2

        if component == 'fus':
            f = aircraft['fus']['length'] / aircraft['fus']['diameter']
            aircraft['fus']['FF'] = 0.9 + 5 / (f ** 1.5) + f / 400
    
    return aircraft


def compute_oswald(AR, Lambda = 0):
    if Lambda >= np.pi/6:
        return
    elif Lambda == 0:
        return 1.78 * (1 - 0.045*AR**(0.68)) - 0.64
    else:
        val_max = 4.61 * (1 - 0.045*AR**(0.68))*(np.cos(Lambda))**0.15 - 3.1
        val_min = 1.78 * (1 - 0.045*AR**(0.68)) - 0.64
        Lambda_min = 0
        Lambda_max = np.pi/6
        return np.interp(Lambda, [Lambda_min, Lambda_max], [val_min, val_max])
    
    
def T_W_takeoff(takeoff_distance, Clmax, Wing_load_vector):
    '''
    beta:   take off weight / MTOW
    K:      V_takeoff / V_stall
    alpha:  T_i / T_nominal
    S_g:    Takeoff distance
    CL_max
    W_S:    Wing load  
    '''
    
    beta = 0.97
    K = 1.2
    alpha = 1
    rho_SL = 1.225
    S_g = takeoff_distance 
    W_S = Wing_load_vector
    
    return  ((beta**2 * K**2)/(alpha*9.81*S_g*rho_SL*Clmax))*(W_S)   
    
    
def T_W_climb(V_climb, K, Cd0, RS, wing_load_factor):
    '''
    beta:    climb weight / MTOW
    alpha:   T_i / T_nominal
    RS:   raise of climb
    q:       dynamic pressure
    K:       1/(pi * e * AR) 
    V_climb
    Cd0
    W_S:    Wing load
    '''
    beta = 0.955
    alpha = 0.9
    rho_SL = 1.225
    q = 0.5 * rho_SL * V_climb ** 2
    W_S = wing_load_factor
    
    return (beta/alpha)*((Cd0*q/beta)*(1/W_S) + (beta*K/q)*W_S + ((RS)/(V_climb)))


def T_W_cruise(V_cruise, Cd0, K, wing_load_vector):
    '''
    beta:    climb weight / MTOW
    alpha:   T_i / T_nominal
    q:       dynamic pressure
    K:       1/(pi * e * AR) 
    Cd0
    W_S:    Wing load
    '''
    
    beta = 0.87
    alpha = 0.5
    rho_SL = 1.225
    q = 0.5 * rho_SL * V_cruise ** 2
    W_S = wing_load_vector
    
    return (beta/alpha)*((Cd0*q/beta)*(1/W_S) + (beta*K/q)*W_S)


def T_W_pouso(landing_distance, Clmax):
    '''
    beta:    climb weight / MTOW
    K:       V_landing / V_stall
    mu:      friction Coulomb coefficient
    S_l:     Landing distance
    Clmax
    '''
    beta = 0.853
    K = 1.3
    mu = 0.3
    S_l = landing_distance
    rho_SL = 1.225
    
    
    return (S_l*9.81*mu*rho_SL*Clmax)/(beta*K**2)
