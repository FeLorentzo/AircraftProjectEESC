#%%
import numpy as np
from ambiance import Atmosphere
import json
import matplotlib.pyplot as plt
import aircraftconceptualdesign as acd
import units
import desempenho
#%%
# Initial requirements based on cargo mission stablished. The project velocity is based on cruise mission
aircraft = {
   'definitions': { 
        'MTOW': units.lbm2kg(19000), # Kg, assuming MTOW given by FAR23
        'V_cruise': units.knot2ms(230), # m/s, assuming CAS
        'W_empty/W': 0.6, # from database
        'W_crew': 200, # 2 pilots
        'operational ceiling': units.ft2m(25000),
        'g' : 9.81,
        'rho_SL' : Atmosphere(0).density[0], # Air density sea level
   },
  'engine': {
      'power': units.hp2Watt(1600)
  },
  'wing': {},
  'ht': {},
  'vt': {},
  'fus': {},
  'landing gear': {},
  'coeficients':{"Cl_max" : 2},
  'speeds':{},
  'missao':{},
  'weights':{}
}

aircraft['speeds']['V_cruise'] = aircraft['definitions']['V_cruise']
aircraft['weights']['W_empty/W'] = aircraft['definitions']['W_empty/W']
aircraft['weights']['W_crew'] = aircraft['definitions']['W_crew']
aircraft['weights']['MTOW'] = aircraft['definitions']['MTOW']

# Atmosphere objects

isa_ceiling = Atmosphere(aircraft['definitions']['operational ceiling'])
isa_sealevel = Atmosphere(0)

# Compute range based on MTOW from database
range = 0.0854*aircraft['weights']['MTOW'] + 461
print(f'Range previewed by database: {units.mi2m(range)/1000} km') # ~ 1100 nm


# Mission profile
missao = {
    "Perfil" : ['takeoff','climb','cruise_1','decend','loitter','cruise_2','landing'],
    "Range_1" : units.mi2m(range),
    "Range_2" : units.mi2m(100),
    "Loitter_time" : 20 * 60
}

aircraft['missao'] = missao

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

########################################################################################################################################
#%%
## Calculations on initial performance ##

# Empty weight
aircraft['weights']["W_empty"] = aircraft['weights']["W_empty/W"] * aircraft['weights']["MTOW"]
print(f'Empty weight: {aircraft["weights"]["W_empty"]}')

# wing load estimation, through data linearization of database. is given by Kg of MTOW/ wing area; must be converted in N
aircraft['weights']["wing_load"] = (0.00846 * aircraft['weights']["MTOW"] + 121)*aircraft['definitions']['g']
print(f'Wing load: {aircraft["weights"]["wing_load"]:.2f}')

# Designing wing area for cruise condition, assuming 5% of fuel consumption
aircraft['wing']["S"] = 0.95*aircraft["weights"]["MTOW"] * aircraft['definitions']['g'] / aircraft['weights']["wing_load"]
print(f'Wing area: {aircraft["wing"]["S"]:.1f}')

# Defining Cl for cruise condition
aircraft['coeficients']["Cl cruise"] = aircraft["weights"]["wing_load"] / ( 0.5 * aircraft['definitions']['rho_SL'] * (aircraft['speeds']["V_cruise"]**2 ) )
print(f'cruise sea level Cl: {aircraft["coeficients"]["Cl cruise"]:.2f}')
print(f'Cl Max.: {aircraft["coeficients"]["Cl_max"]}')

# Estimate required thrust. Assuming, in cruise, 65% of maximum power;
# proppeler efficiency of 80%; sea level
power_cruise = 0.65 * aircraft["engine"]["power"]
aircraft["engine"]['eta'] = 0.65

aircraft['engine']["thrust cruise"] =  aircraft["engine"]['eta'] * power_cruise/ aircraft['speeds']["V_cruise"]
aircraft['engine']["Thrust_to_weight"] =  aircraft["engine"]["thrust cruise"]/(aircraft['weights']["MTOW"]*aircraft['definitions']['g'])
print(f'Empuxo do motor: {aircraft["engine"]["thrust cruise"]:.2f}')

aircraft['coeficients']["Cd cruise"] =  aircraft["engine"]["thrust cruise"]/(0.5 * aircraft['definitions']['rho_SL'] * (aircraft['speeds']["V_cruise"]**2) * aircraft["wing"]["S"])
print(f'Cd de cruzeiro a nível do mar: {aircraft["coeficients"]["Cd cruise"]:.6f}')
print(f'L/D de cruzeiro a nível do mar: {aircraft["coeficients"]["Cl cruise"]/aircraft["coeficients"]["Cd cruise"]:.2f}')

# Compute SFC cruise and loiter through Raymer method pg. 23
aircraft["engine"]['SFC_loiter'] = 0.6 * units.ms2knot(aircraft['speeds']['V_cruise']) / (units.hp2Watt(550) * 0.7)
aircraft["engine"]["SFC_cruise"] = 0.5 * units.ms2knot(aircraft['speeds']['V_cruise']) / (units.hp2Watt(550) * 0.8)
print(f'Consumo cruise : {aircraft["engine"]["SFC_cruise"]:e}')
print(f'Consumo loiter : {aircraft["engine"]["SFC_loiter"]:e}')

# Compute stall
aircraft['speeds']['V_stall'] = np.sqrt(2*aircraft['weights']["MTOW"]*aircraft['definitions']['g']/(aircraft['definitions']['rho_SL'] * aircraft['wing']['S'] * 2))
print(f"Vel. Stall : {aircraft['speeds']['V_stall']}")

########################################################################################################################################
#%%
## First weight estimative ##
def first_weight_estimate(aircraft: dict) -> float :
    frac_weight = 1
    fracs = {
        'takeoff':0.97,
        'climb':0.987,
        'cruise_1':np.exp(-aircraft['missao']['Range_1'] * aircraft["engine"]["SFC_cruise"]/(aircraft["speeds"]['V_cruise'] * (aircraft["coeficients"]["Cl cruise"]/aircraft["coeficients"]["Cd cruise"]))),
        'cruise_2':np.exp(-aircraft['missao']['Range_2'] * aircraft["engine"]["SFC_cruise"]/(aircraft["speeds"]['V_cruise'] * (aircraft["coeficients"]["Cl cruise"]/aircraft["coeficients"]["Cd cruise"]))),
        'decend': 1,
        'loitter': np.exp(-aircraft['missao']['Loitter_time'] * aircraft['engine']['SFC_loiter']/(aircraft["coeficients"]["Cl cruise"]/aircraft["coeficients"]["Cd cruise"])),
        'landing': 0.995
    }

    print(fracs)

    for leg in aircraft['missao']['Perfil']:
        frac_weight = frac_weight * fracs[leg]

    return frac_weight

wf_frac = 1 - first_weight_estimate(aircraft)

print(wf_frac)

print(f'wf/w0= {wf_frac}')
print(f'wf = {wf_frac*aircraft["weights"]["MTOW"]}')
print(f'vol = {wf_frac*aircraft["weights"]["MTOW"]/0.72}')

w_pay = aircraft["weights"]['MTOW'] * (1 - wf_frac - aircraft["weights"]['W_empty/W']) - aircraft["weights"]['W_crew']
aircraft['weights']['payload'] = w_pay
aircraft['weights']['fuel'] = aircraft["weights"]['MTOW'] * wf_frac # Em kg
aircraft['weights']['BOW'] = aircraft["weights"]['MTOW'] - aircraft['weights']['fuel'] # Em kg
print(f'Payload: {w_pay}')

########################################################################################################################################
#%%
## Estimating areas of tail, flap and aileron ##

print(f"S_ref: {aircraft['wing']['S']:.3f}")

aircraft = acd.estimate_aerodynamic_areas(aircraft)

print(
f"""
S_ref: {aircraft['wing']['S']:.3f}
S_Ht: {aircraft['ht']['S']:.3f}
S_profundor: {aircraft['ht']['S_elevator']:.3f}
S_Vt: {aircraft['vt']['S']:.3f}
S_leme: {aircraft['vt']['S_rudder']:.3f}
S_flap: {aircraft['wing']['S_flap']:.3f}
S_aileron: {aircraft['wing']['S_aileron']:.3f}
""")

## Including wet areas estimatives from CAD of the aircraft ##

aircraft['wing']['S_wet'] = 84.4
aircraft['ht']['S_wet'] = 18.4
aircraft['vt']['S_wet'] = 12.3
aircraft['fus']['S_wet'] = 95.4

## Updating parameters from CAD of the aircraft ##

wing_param = {
    'b': 21,
    'CMA': 1.965,
    'S_wet': 84.4,
    'Percent_laminar': 0.1,
    'sweep' : 0,
    'hinge': False,
    'thickness': 0.1, #provisory
    'material': "Aerospace aluminum", # must be compatible with weight_estimation_data.json file
}

wing_param['AR'] = wing_param['b']**2 / aircraft['wing']['S']
wing_param['e'] = compute_oswald(wing_param['AR'])

ht_param = {
    'b': 6.5,
    'CMA': 1.491,
    'S_wet': 18.4,
    'Percent_laminar': 0.1,
    'sweep' : 0,
    'hinge': True,
    'thickness': 0.1, #provisório
    'material': "Aerospace aluminum", # must be compatible with weight_estimation_data.json file
    'e': 0.8,
}

vt_param = {
    'b': 2.75,
    'CMA': 2.156,
    'S_wet': 12.3,
    'Percent_laminar': 0.1,
    'sweep' : 0,
    'hinge': True,
    'thickness': 0.1, #provisório
    'material': "Aerospace aluminum", # must be compatible with weight_estimation_data.json file
    'e': 0.8
}

fus_param = {
    'length': 18,
    'S_wet': 95.9,
    'diameter': 1.6,
    'Percent_laminar': 0,
    'material': "Aerospace aluminum", # must be compatible with weight_estimation_data.json file
}

aircraft['wing'].update(wing_param)
aircraft['ht'].update(ht_param)
aircraft['vt'].update(vt_param)
aircraft['fus'].update(fus_param)

########################################################################################################################################
#%%
## Drag estimative ##

# Friction drag

Cf = 0
isa = Atmosphere(aircraft['definitions']['operational ceiling'])
rho = isa.density[0]
T = isa.temperature[0]
mu = isa.dynamic_viscosity[0]
vel_som = isa.speed_of_sound[0]
Mach = aircraft['speeds']['V_cruise']/vel_som
Re_operational = rho * aircraft['speeds']['V_cruise'] * aircraft['wing']['CMA'] / mu

print(f'Operational Reynolds = {Re_operational:.3e}')
print(f'Operational Mach = {Mach:.3f}')

print(20*'-')

for key, component in aircraft.items():
    if key == 'fus':
        cut = acd.Re_cutoff(component['length'])
        Re = rho * aircraft['definitions']['V_cruise'] * component['length'] / mu
    elif 'CMA' in component:
        cut = acd.Re_cutoff(component['CMA'])
        Re = rho * aircraft['definitions']['V_cruise'] * component['CMA'] / mu
    else:
        continue

    print(f'Reynolds cutoff: {cut:.2e}')
    print(f'Reynolds: {Re:.2e}')

    if Re > cut:
        Re = cut

    component['Cf'] = component['Percent_laminar'] * acd.Cf_laminar(cut) + (1-component['Percent_laminar']) * acd.Cf_turbulent(cut,Mach)
    print(f'Cf do {key}: {component["Cf"]:.4f}\n')

# Form Factor drag

M = aircraft['definitions']['V_cruise']/vel_som

aircraft = acd.compute_FF(aircraft, M)

print(f"""
Wing form factor: {aircraft["wing"]["FF"]:.4f}
Horizontal stabilizer form factor:  {aircraft["ht"]["FF"]:.4f}
Vertical stabilizer form factor: {aircraft["vt"]["FF"]:.4f} 
Fuselage form factor: {aircraft["fus"]["FF"]:.4f}
""")

# Interference drag

aircraft['wing']['Q_int'] = 1
aircraft['ht']['Q_int'] = 1.05
aircraft['vt']['Q_int'] = 1.05
aircraft['fus']['Q_int'] = 1

# Miscelaneos drag due fuselage upsweep

mu = np.arctan(631/8656)
Amax = np.pi*0.8**2

Cd_misc = 3.83*mu**2.5*Amax / aircraft['wing']['S']

print(f'Cd de miscelânia = {Cd_misc:.4f}')

# Compute total parasite drag

Cd_p = 0

for component, values in aircraft.items():
    if 'Cf' in values:
        Cd_c = values["Cf"]*values['FF']*values['Q_int']*values['S_wet'] / aircraft["wing"]["S"]
        print(f'Componente de arrasto parasita em {component} vale = {Cd_c:.4f}\n')
        aircraft[component]['CD0'] = Cd_c
        Cd_p += Cd_c

print(f'Arrasto parasita total sem componente de miscelânia = {Cd_p:.4f}\n')

Cd_p += Cd_misc

print(f'Arrasto parasita total com componente de miscelânia = {Cd_p:.4f}\n')

aircraft['coeficients']['CD0'] = Cd_p

#####################################################################################################################################
#%%
## Drag x velocity curves ##

# Parasite drag

vector_v = np.linspace(15,1.4 * aircraft['speeds']['V_cruise'], 80)

rho_SL = isa_sealevel.density[0]

D_p = 0.5 * rho_SL * vector_v ** 2 * aircraft['wing']['S'] * Cd_p

# Induced drag

AR = aircraft['wing']['b']**2 / aircraft['wing']['S']
e = compute_oswald(AR)
W = aircraft['definitions']['MTOW']
S = aircraft['wing']['S']

CL_for_Cdi_induced = 2 * W * aircraft['definitions']['g'] / (rho_SL * vector_v ** 2 * S)

K = 1 / (np.pi * AR * e)

Cd_i = K * CL_for_Cdi_induced ** 2

D_i = 0.5 * rho_SL * vector_v ** 2 * aircraft['wing']['S'] * Cd_i

print(f'Alongamento = {AR:.2f}')
print(f'Coeficiente de Oswald = {e:.2f}')
print(f'Fator k = {K:.4f}')
# print(f'CDi = {Cd_i}')

# Plot

plt.figure()
plt.plot(vector_v, D_p, 'b', label='Parasite drag')
plt.plot(vector_v, D_i, 'r', label='Induced drag')
plt.plot(vector_v, D_p + D_i, 'g', label='Total drag')
plt.xlabel('Velocity CAS [m/s]')
plt.ylabel('Drag [N]')
plt.grid()
plt.legend()

#####################################################################################################################################
#%%
# Constraint analysis

#####################################################################################################################################
#%%
# Performance
print("\n\n# Performance")

## Decolagem:
takeoff_data = {
    'gamma_climb': 1,
    'Cl_run': 0.1, 
    'V_f': 100,
    'mi': 0.01, 
    'h_obstacle': 50 # FAR 23.53(b)
}


dist_ground, dist_transistion, dist_climb = desempenho.takeoff(dados_aeronave = aircraft,
                                                                takeoff_data = takeoff_data)
print('dist_ground, dist_transistion, dist_climb' )
print(dist_ground, dist_transistion, dist_climb )
consumo_takeoff =  aircraft['weights']['MTOW']*(1-0.97)
print('consumo_takeoff',consumo_takeoff)

## Subida
climb_data = {
    'h0':takeoff_data['h_obstacle'],
    'hf': 40_000,
    'W_fuel_init': aircraft['weights']['fuel'] - consumo_takeoff,
    'vmin': aircraft['speeds']['V_stall'],
    'vmax': aircraft['speeds']['V_cruise'] * 1.4,
    'dT': 0, # Variação da temperatura ISA
    'dh': 100, #ft
}
print(climb_data)
tempo_climb, consumo_climb, dist_climb, FL_final = desempenho.climb(dados_aeronave = aircraft,
                                                                    climb_dict = climb_data)
print('tempo_climb, consumo_climb, dist_climb, FL_final')
print(f"{tempo_climb/60:.2f}min", f"{consumo_climb:.2f}kg", f"{dist_climb/1000:.2f}km", FL_final)
print(f"Consumo de combustivel: {(100*consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

## Cruise
cruise_data = {
    'h0':FL_final, 
    'hf':40_000,
    'W_fuel_init': climb_data['W_fuel_init']- consumo_climb, 
    'mach': aircraft['speeds']['V_cruise']/Atmosphere(units.ft2m(25000)).speed_of_sound[0], 
    'W_reserve_fuel': aircraft['weights']['fuel'] * 0.30, 
    'step_climb':False,
    'dt': 10, #s?
} 
print(cruise_data)
tempo_cruise, consumo_cruise, dist_cruise, FL_final = desempenho.cruise(dados_aeronave = aircraft,
                                                                        cruise_dict = cruise_data)
print('tempo_cruise, consumo_cruise, dist_cruise, FL_final')
print(f"{tempo_cruise/3600:.2f}h", f"{consumo_cruise:.2f}kg", f"{dist_cruise/1000:.2f}km", FL_final)
print(f"Consumo de combustivel: {100*(consumo_cruise+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

## Descend
descend_data = {
    'h0':FL_final, 
    'hf':10_000,
    'W_fuel_init': cruise_data['W_fuel_init'] - consumo_cruise, 
    'vmin': aircraft['speeds']['V_stall'],
    'vmax': aircraft['speeds']['V_cruise'] * 1.4,
    'dt': 1, #s
    'dT': 0,
    'Thrust_percent': 10
}
print(descend_data)
tempo_descida, consumo_descida, distancia_descida, FL_final = desempenho.decend(dados_aeronave = aircraft,
                                                                               descend_dict=descend_data)
print('tempo_descida, consumo_descida, distancia_descida, FL_final')
print(f"{tempo_descida/60:.2f}min", f"{consumo_descida:.2f}kg", f"{distancia_descida/1000:.2f}km", FL_final)
print(f"Consumo de combustivel: {100*(consumo_cruise+consumo_descida+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

## Loitter

loitter_data = {
    'h_loitter': FL_final,
    'loitter_time': 20*60, # s
    'W_fuel_init':  descend_data['W_fuel_init'] - consumo_descida,
    'dt': 5, #s
    'dT': 0, #K
}
consumo_loitter = desempenho.loitter(dados_aeronave = aircraft,
                                     loitter_dict = loitter_data)[0]
print('consumo_loitter')
print(f"{consumo_loitter:.2f}kg")
print(f"Consumo de combustivel: {100*(consumo_cruise+consumo_loitter+consumo_descida+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")


## Cruise
cruise2_data = {
    'h0':FL_final, 
    'hf':10_000,
    'W_fuel_init': loitter_data['W_fuel_init']- consumo_loitter, 
    'mach': aircraft['speeds']['V_cruise']/Atmosphere(0).speed_of_sound[0], 
    'W_reserve_fuel': aircraft['weights']['fuel'] * 0.125, 
    'step_climb':False,
    'dt': 5, #s?
}
print(cruise2_data)
tempo_cruise2, consumo_cruise2, dist_cruise2, FL_final = desempenho.cruise(dados_aeronave = aircraft,
                                                                        cruise_dict = cruise2_data)
print('tempo_cruise2, consumo_cruise2, dist_cruise2, FL_final')
print(f"{tempo_cruise2/3600:.2f}h", f"{consumo_cruise2:.2f}kg", f"{dist_cruise2/1000:.2f}km", FL_final)
print(f"Consumo de combustivel: {100*(consumo_loitter+consumo_cruise+consumo_cruise2+consumo_descida+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

loitter_data2 = {
    'h_loitter': FL_final,
    'loitter_time': 20*60, # s
    'W_fuel_init':  descend_data['W_fuel_init'] - consumo_descida,
    'dt': 5, #s
    'dT': 0, #K
}
consumo_loitter2 = desempenho.loitter(dados_aeronave = aircraft,
                                     loitter_dict = loitter_data2)[0]
print('consumo_loitter')
print(f"{consumo_loitter2:.2f}kg")
print(f"Consumo de combustivel: {100*(consumo_cruise+consumo_cruise2+consumo_loitter+consumo_loitter2+consumo_descida+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

## Descend
descend_data = {
    'h0':FL_final, 
    'hf':100,
    'W_fuel_init': cruise2_data['W_fuel_init'] - consumo_cruise2 - consumo_loitter2, 
    'vmin': aircraft['speeds']['V_stall'],
    'vmax': aircraft['speeds']['V_cruise'] * 1.4,
    'dt': 1, #s
    'dT': 0,
    'Thrust_percent': 10
}
print(descend_data)
tempo_descida2, consumo_descida2, distancia_descida2, FL_final = desempenho.decend(dados_aeronave = aircraft,
                                                                               descend_dict=descend_data)
print('tempo_descida, consumo_descida, distancia_descida, FL_final')
print(f"{tempo_descida2/60:.2f}min", f"{consumo_descida2:.2f}kg", f"{distancia_descida2/1000:.2f}km", FL_final)
print(f"Consumo de combustivel: {100*(consumo_cruise+consumo_cruise2+consumo_loitter+consumo_loitter2+consumo_descida+consumo_descida2+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")

land_data={
    'gamma_descent': 3, # deg
    'Cl_run': -0.2,
    'mi': 0.01,
    'mi_break': 0.1,
    'rho': 1,
    'h_obstacle': 50,
    'reversores': 0.5,
}
## Land 
SA, SF, SFR, SB = desempenho.land(dados_aeronave=aircraft,
                                  land_dict=land_data)
print('SA, SF, SFR, SB')
print(SA, SF, SFR, SB)
# %%

## Endurance flight

Endurance_data = {
    'h_loitter': FL_final,
    'loitter_time': 5*3600 - 10*60, # s
    'W_fuel_init':  climb_data['W_fuel_init'] - consumo_takeoff,
    'dt': 5, #s
    'dT': 0, #K
}
consumo_endurance = desempenho.loitter(dados_aeronave = aircraft,
                                     loitter_dict = Endurance_data)[0]
print('consumo_endurance')
print(f"{consumo_endurance:.2f}kg")
print(f"Consumo de combustivel: {100*(consumo_endurance+consumo_climb+consumo_takeoff)/aircraft['weights']['fuel']:.2f}%\n")
# %%
