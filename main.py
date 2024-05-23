import numpy as np
from ambiance import Atmosphere
import json
import matplotlib.pyplot as plt
import aircraftconceptualdesign as acd
import units

# Initial requirements based on cargo mission stablished. The project velocity is based on cruise mission

aircraft = {
  'MTOW': units.lbm2kg(19000), # Kg, assuming MTOW given by FAR23
  'V_cruise': units.knot2ms(230), # m/s, assuming CAS
  'W_empty/W': 0.6, # from database
  'W_crew': 200, # 2 pilots
  'operational ceiling': units.ft2m(25000),
  'engine': {
  'power': units.hp2Watt(1600)
            },
  'wing': {},
  'elev': {},
  'rudd': {},
  'fuse': {},
  'landing gear': {},
}

# Compute range based on MTOW from database
range = 0.0854*aircraft["MTOW"] + 461
print(f'Range previewed by database: {range} nm') # ~ 1100 nm


g = 9.81
rho_SL = 1.225 # Air density sea level

# Mission profile
missao = {
    "Perfil" : ['Take off','Ascending','Cruise_1','Descending','Loitter','Cruise_2','Landing'],
    "Range_1" : units.mi2m(range),
    "Range_2" : units.mi2m(100),
    "Loitter_time" : 20 * 60
}

aircraft['missao'] = missao
