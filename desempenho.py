import numpy as np
from ambiance import Atmosphere

def takeoff(dados_aeronave:dict, takeoff_data:dict):
    '''
    Implementação baseada na teoria apresentada por Raymer.
    Feita de modo a ser compatível com a FAR 23.

    Inputs:
    dados_aeronave (dict): Dicionário contendo os dados da aeronave, incluindo peso, área da asa, etc.
    gamma_climb (float): Ângulo de subida em radianos.
    Cl_run (float): Coeficiente de sustentação durante a corrida de decolagem.
    Cl_max (float): Coeficiente de sustentação máximo.
    mi (float): Coeficiente de atrito entre o trem de pouso e a superfície.
    rho (float): Densidade do ar em kg/m^3.
    g (float): Aceleração devido à gravidade em m/s^2.
    V_i (float): Velocidade inicial de decolagem em m/s.
    V_f (float): Velocidade final de decolagem em m/s.

    Returns:
    dists (list): Distancias.
    '''

    # Decomposição dos dados
    ## Dados da Aeronave
    S = dados_aeronave['wing']['S']
    W = dados_aeronave['definitions']['MTOW']
    CD0 = dados_aeronave['coeficients']['CD0']
    rho = dados_aeronave['definitions']['rho_SL']
    g = dados_aeronave['definitions']['g']
    V_stall = dados_aeronave['speeds']['V_stall']
    K = 1/(np.pi * dados_aeronave['wing']['AR'] * dados_aeronave['wing']['e'])
    eta = dados_aeronave["engine"]['eta'] 

    # Dados da operação
    gamma = takeoff_data['gamma_climb']
    CL = takeoff_data['Cl_run']
    V_f = takeoff_data['V_f']
    mi = takeoff_data['mi']
    h_obstacle = takeoff_data['h_obstacle']

    # Calculo do tThrust
    T = eta * dados_aeronave["engine"]["power"]/ dados_aeronave['speeds']["V_cruise"]
    ### Aceleração (V: 0 - 1.1*V_stall)
    # Modelo: Aceleração constante 
    #   M.a(t) = T - D - mi.(W-L)
    #   a(t) = g.[(T/W - mi) - (rho/(W/S)).(CD0 + K.CL**2  - mi.CL).V**2]
    #   a(t) = g.[KT- KA.V**2]; KT=(T/W - mi); KA = (rho/(W/S)).(CD0 + K.CL**2  - mi.CL)

    KT = T/W - mi
    KA = (rho/(2*W/S)) * (CD0 + K*CL**2  - mi*CL)

    #   s(t) = \int_{v_i}^{v_f}(V/a)dV = 0.5 * \int_{v_i}^{v_f}(1/a)d(V**2)
    #   s(t) = (1/2gKA).ln((KT + KA.(V_f)**2)/(KT + KA.(V_i)**2))

    SG = -(1/(2*g*KA)) * np.log((KT - KA*(V_f)**2)/(KT))

    ### Transição (V: 1.1*V_stall - 1.2*V_stall)
    # Def: n = 1.2
    n = 1.2
    # n = 1 + V_tr**2/R.g
    R = (1.2*V_stall**2)/(g*(n-1))
    h_R = R * (1 - np.cos(np.deg2rad(gamma)))
    ST = np.sqrt(R**2 - (R - h_R)**2)

    # Climb
    if h_R > h_obstacle:
        SC = 0
    else:
        SC = (h_obstacle - h_R)/np.tan(np.rad2deg(gamma))

    return [SG, ST, SC]


def climb(dados_aeronave:dict, climb_dict:dict) -> tuple:

    ## Dados da aeronave
    # Dados Asa
    AR = dados_aeronave['wing']['AR']
    e=dados_aeronave['wing']["e"]
    S = dados_aeronave['wing']["S"]
    # Dados aerodinamicos
    Cd0 = dados_aeronave['coeficients']['CD0']
    K = 1/(np.pi*AR*e)
    # Dados de peso
    BOW = dados_aeronave['weights']["BOW"] * 9.81 # N
    W_fuel = dados_aeronave['weights']['fuel'] * 9.81 # N
    # Dados de motor
    eta = dados_aeronave["engine"]['eta'] 
    TSFC = dados_aeronave["engine"]["SFC_cruise"] # N

    ## Dados o climb
    h0 = climb_dict['h0']
    hf = climb_dict['hf']
    W_fuel_init = climb_dict['W_fuel_init'] * 9.81 # N
    vmin = climb_dict['vmin']
    vmax = climb_dict['vmax']
    dT = climb_dict['dT']
    dh = climb_dict['dh'] # intervalos de altitudes
    
    # Condições atmosféricas
    T0 = Atmosphere(0).temperature[0]
    P0 =  Atmosphere(0).pressure[0]
    rho0 =  Atmosphere(0).density[0]

    ## Iniciação das variaveis
    t = 0
    tempo_total = 0
    consumo_total = 0
    dist_percorrida = 0
    if W_fuel_init is not None:
      W = BOW + W_fuel_init
    else:
      W = BOW + W_fuel

    # Cria um range de velocidades a serem avaliadas
    V = np.arange(vmin, vmax, 10)
    Thrust0 = eta * dados_aeronave["engine"]["power"]/V

    # variáveis auxiliares
    consumos = []
    distancias = []
    tempos = []
    RCs_max = []

    alturas = np.arange(h0, hf+dh, dh)

    #print('alturas = ', alturas)
    for h_ind in alturas:
        # Corrige a variação de temperatura ISA
        T      = Atmosphere(h_ind * 0.3048).temperature[0]                      # altitude em metros
        h_real = h_ind * (T + dT)/T

        # Calculo
        rho = Atmosphere(h_real * 0.3048).density[0]
        sigma = rho/rho0
        M     = (V)/Atmosphere(h_real * 0.3048).speed_of_sound[0]               # V (true aero speed)  (V*sigma**0.5)= Ve (Velocidade equivalente)
        phi   = 1/(0.7 * M**2)  *  (((1 + 0.2 * M**2)**3.5 - 1)/((1 + 0.2 * M**2)**2.5))
        f_a   =  0.7 * (M**2) * (phi-(0.190263 * (T+dT)/T))

        # Variação do empuxo
        Thrust = Thrust0 * sigma**0.7                                           # empuxo para menor q troposfera
        if h_real > 36089:
            Thrust = 1.439 * Thrust0 * sigma                                    # empuxo para altitudes maiores q troposfera
            f_a    = phi * 0.7 * M**2                                           # fator de aceleração para altitudes maiores q troposfera

        Cl = (2 * (W/S))/(rho * V**2)
        Cd = Cd0 + K * (Cl**2)
        E = Cl/Cd

        # Calculo da razão de subida máxima
        # gamma = np.arcsin(Thrust/W - 1/E) # Aprox de pequenos angulos 
        A = (1 - (Thrust/W)**2)
        B = 2/E
        C = ((1/E**2) - (Thrust/W)**2)
        delta = B**2 - 4*A*C
        gamma = np.arctan((-B + np.sqrt(delta))/(2*A))
        RC_1 = V * np.sin(gamma) 
        RC = RC_1/(1 + f_a)
        RCs_max.append(np.max(RC))
        index = np.where(RC == np.max(RC))                                       # pega o index do vetor RC em que RC é máximo
        v_rcmax = V[index][0]                                                   # pega a velocidade em que RC é máximo
        gamma_rcmax = gamma[index][0]                                           # pega o gama em que RC é máximo
        v_horizontal = v_rcmax * np.cos(gamma_rcmax)                            # calcula a velocidade horizontal em que RC é máximo para determinada faixa de altitude


        ## Calcula Consumo
        t = dh * ((T + dT)/T) * 0.3048 / max(RC)                     # Tempo para o maximo RC de cada altitude = faixa de altitude/RC
        consumo = TSFC * Thrust[index][0] * t
        consumos.append(consumo)
        W = W - consumo
        tempos.append(t)
        d = v_horizontal * t                                                    # Distância percorrida
        distancias.append(d)

    # Logica para determinar se a aeronave deve fazer step-climb
    # Quando eu escrevi só eu e deus entendemos como isso funciona
    # Agora esse código está apenas nas mão do Senhor
        
    if int(hf/1000)%2 == 1:
        flight_levels = np.arange(1000, hf + 1000, 2000)                     # Array com todos os FL até o max
    else:
        flight_levels = np.arange(0, hf + 2000, 2000)
    
    mask = np.array(RCs_max)<2.54                                               # Mask p/ alturas que precisam de stepclimb
    
    if flight_levels[flight_levels > h0].size == 0:
      # Caso não haja mais nenhum flight level valido acima do atual permanece
      FL = h0
    else:
      # Caso hajam flight leves validos acima
      flight_levels = flight_levels[flight_levels>=h0]
      if alturas[mask].size == 0:
          # Caso satisfaça o critério de razão de subida
          FL = max(flight_levels[flight_levels>h0])
      else:
          min_alt_step_climb = min(alturas[mask])
          if flight_levels[flight_levels < min_alt_step_climb].size == 0:                                            # Min h p/ step climb
            FL = h0
          else:
            FL = max(flight_levels[flight_levels < min_alt_step_climb])
    
    mask2 = alturas < FL                                                        # Itens que estão a alturas menores que o flight level

    tempo_total     = sum(np.array(tempos)[mask2])
    dist_percorrida = sum(np.array(distancias)[mask2])
    consumo_total   = sum(np.array(consumos)[mask2])

    return tempo_total, consumo_total/9.81, dist_percorrida, FL


def cruise(dados_aeronave: dict,  cruise_dict:dict):
    # Deve retornar o tempo decorrido, combustivel consumido, distância percorida e variação da altura
    
    ## Dados da aeronave
    # Dados Asa
    AR = dados_aeronave['wing']['AR']
    e=dados_aeronave['wing']["e"]
    S = dados_aeronave['wing']["S"]
    # Dados aerodinamicos
    CD0 = dados_aeronave['coeficients']['CD0']
    K = 1/(np.pi*AR*e)
    # Dados de peso
    BOW = dados_aeronave['weights']["BOW"] * 9.81 # N
    W_fuel = dados_aeronave['weights']['fuel'] * 9.81 # N
    # Dados de motor
    eta = dados_aeronave["engine"]['eta']  # N
    TSFC = dados_aeronave["engine"]["SFC_cruise"]

    ## Dados Cruise
    h0 = cruise_dict['h0']
    hf = cruise_dict['hf']
    W_fuel_init = cruise_dict['W_fuel_init'] * 9.81
    mach = cruise_dict['mach']
    W_reserve_fuel = cruise_dict['W_reserve_fuel'] * 9.81
    step_climb = cruise_dict['step_climb']
    dt = cruise_dict['dt']

    ## Pesos
    W = BOW + W_fuel_init        # Peso total no inicio do cruzeiro
    W_min_cruseiro = BOW + W_reserve_fuel 
    
    # Acumuladores
    distancia_total=0
    t = 0
    h = h0 * 0.3048

    while True:
        t+=dt

        ## Condições de voo
        a = Atmosphere(h).speed_of_sound[0]  
        rho = Atmosphere(h).density[0] 
        V = mach*a

        ## Coeficientes 
        Cl = (2 * (W/S))/( rho * V**2) #calculo do cl
        Cd = CD0 + K * (Cl**2)
        E = Cl/Cd
        T = W/E

        ## Atualiza peso da aeronave
        W -= TSFC*dt*T

        ## Atualiza distância percorrida
        distancia_total+= V*dt

        ## Avalia condição de fim de cruseiro
        if  W < W_min_cruseiro:
            consumo_cruseiro = (BOW + W_fuel_init - W)/9.81
            return  t, consumo_cruseiro, distancia_total, h/0.3048
        
        if step_climb:
            ## Avalia possibilidade de um step-climb
            climb_data = {
                    'h0':h/0.3048,
                    'hf': hf,
                    'W_fuel_init': (W - BOW)/9.81,
                    'vmin': V * 0.9,
                    'vmax': V * 1.1,
                    'dT': 0, # Variação da temperatura ISA
                    'dh': 100, #ft
                }
            
            tempo_subida, consumo_subida, dist_percorrida_subida, FL_atual = climb(dados_aeronave, climb_data)
            
            h = FL_atual * 0.3048
            W -= consumo_subida * 9.81
            t += tempo_subida
            distancia_total+= dist_percorrida_subida


def turn(dados_aeronave: dict, turn_dict: dict):

    # Decomposição dos dados
    S = dados_aeronave['wing']['S']
    T = dados_aeronave['engine']["thrust cruise"]
    W = dados_aeronave['MTOW']
    CD0 = dados_aeronave['coeficients']['CD0']
    K = 1/(np.pi * dados_aeronave['AR'] * dados_aeronave['e'])
    CL_max = dados_aeronave['Cl_max']
    n_max = dados_aeronave['n_max']

    rho = turn_dict['rho']
    g = turn_dict['g']
    V_max = turn_dict['V_max']

    V = np.linspace(0, V_max*1.5,200)
    q = 0.5 * rho * (V**2)
   
    ### Stall limit
    # psi = g.sqrt(n**2 - 1)/V; n = L/W -> n = 0.5*rho*S*CL_max*V**2/W
    # psi = g.sqrt((n/V)**2 - 1/V**2)  -> psi = g.sqrt((0.5*rho*S*CL_max*V/W)**2 - 1/V**2) 
    L = q * S * CL_max
    n_stall = L/W
    psi_stall = g * np.sqrt(n_stall**2 - 1)/V

    # Filtra os valores de psi maiores que zero e menores que o limite estrutural
    msk = (psi_stall>0)*(n_stall<n_max)
    psi_stall = psi_stall[msk]
    V_stall   = V[msk]

    corner_speed = V_stall.max()

    ### Struvtural limit
    psi_lim = g * np.sqrt(n_max**2 - 1)/V

    msk = (V > corner_speed)
    psi_lim = psi_lim[msk]
    V_lim = V[msk]

    ### Sustained turn envelope
    # Raymer: Supondo alinhamento entre tração e arrasto, CL = nW/qS
    wing_load = W/S
    n_sust = np.sqrt((q/(K*wing_load))*(T/W - q*CD0/wing_load))
    psi_sust = g * np.sqrt(n_sust**2 - 1)/V

    msk = n_sust>0
    n_sust = n_sust[msk]
    psi_sust = psi_sust[msk]

    return NotImplementedError


def decend(dados_aeronave: dict, descend_dict: dict):
    ## Dados da aeronave
    # Dados Asa
    AR = dados_aeronave['wing']['AR']
    e=dados_aeronave['wing']["e"]
    S = dados_aeronave['wing']["S"]
    # Dados aerodinamicos
    Cd0 = dados_aeronave['coeficients']['CD0']
    K = 1/(np.pi*AR*e)
    # Dados de peso
    BOW = dados_aeronave['weights']["BOW"] * 9.81 # N
    W_fuel = dados_aeronave['weights']['fuel'] *9.81 # N
    # Dados de motor
    eta = dados_aeronave["engine"]['eta'] 
    TSFC = dados_aeronave["engine"]["SFC_cruise"] # N

    ## Dados o climb
    h0 = descend_dict['h0']
    hf = descend_dict['hf']
    W_fuel_init = descend_dict['W_fuel_init'] * 9.81 # N
    vmin = descend_dict['vmin']
    vmax = descend_dict['vmax']
    dt = descend_dict['dt']
    dT = descend_dict['dT']
    Thrust_percent = descend_dict['Thrust_percent']

    W0 =  BOW + W_fuel_init # N
    W  =  BOW + W_fuel_init # N

    T0   =  Atmosphere(0).temperature
    P0   =  Atmosphere(0).pressure
    rho0 =  Atmosphere(0).density

    # Cria um range de velocidades a serem avaliadas
    V = np.linspace(vmin, vmax+1, 10)
    Thrust0 = eta * dados_aeronave["engine"]["power"]/V
    Thrust_descent = Thrust0*Thrust_percent/100 # N thrust chute meu

    tempo_total=0
    distancia_descida = 0
    h = h0 * 0.3048

    while h > hf*0.3048:
        # Corrige a variação de temperatura ISA
        T = Atmosphere(h).temperature  # altitude em metros
        h_ind = h * T/(T + dT)

        # Calculo da densidade relativa
        rho = Atmosphere(h_ind).density
        sigma = rho/rho0

        # Calculo do Fator de Aceleração
        M     = (V)/Atmosphere(h).speed_of_sound[0]               # V (true aero speed)  (V*sigma**0.5)= Ve (Velocidade equivalente)
        phi   = 1/(0.7 * M**2)  *  (((1 + 0.2 * M**2)**3.5 - 1)/((1 + 0.2 * M**2)**2.5))
        f_a   =  0.7 * (M**2) * (phi-(0.190263 * (T+dT)/T))

        # Variação do empuxo
        Thrust = Thrust_descent*sigma**0.7 # empuxo para menor q troposfera
        if h > 36089:
            Thrust = 1.439*Thrust_descent*sigma # empuxo para altitudes maiores q troposfera
        
        
        # Calculo dos coeficientes
        Cl = (2 * (W/S))/(rho * V**2)
        Cd = Cd0 + K * (Cl**2)
        E = Cl/Cd

        ## Calculo da razão de descidaw
        A = (1 - (Thrust/W)**2)
        B = 2/E
        C = ((1/E**2) - (Thrust/W)**2)
        delta = B**2 - 4*A*C
        gamma = np.arctan((-B - np.sqrt(delta))/(2*A))
        RD = V*np.sin(gamma)
        RD = RD/(1+f_a)
        index = np.where(RD == max(RD)) # pega o index do vetor RC em que RC é máximo
        v_rdmin = V[index][0]           # pega a velocidade em que RD é máximo
        gamma_rdmin = gamma[index][0]   # pega o gama em que RC é máximo
        v_horizontal = v_rdmin*np.cos(gamma_rdmin) #calcula a velocidade horizontal em que RC é máximo para determinada faixa de altitude
        v_vertical   = v_rdmin*np.sin(gamma_rdmin)

        # Calculo do consumo
        consumo=TSFC*Thrust*dt #  com fator de aceleração
        W -= consumo[index][0]  # com fator de aceleração
        
        tempo_total+= dt
        distancia_descida += v_horizontal*dt
        h += v_vertical*dt

    consumo_descida = W0 - W

    return  tempo_total, consumo_descida/9.81, distancia_descida, h/0.3048


def loitter(dados_aeronave: dict, loitter_dict: dict):

    ## Dados da aeronave
    # Dados Asa
    AR = dados_aeronave['wing']['AR']
    e=dados_aeronave['wing']["e"]
    S = dados_aeronave['wing']["S"]
    # Dados aerodinamicos
    Cd0 = dados_aeronave['coeficients']['CD0']
    K = 1/(np.pi*AR*e)
    # Dados de peso
    BOW = dados_aeronave['weights']["BOW"] * 9.81 # N
    W_fuel = dados_aeronave['weights']['fuel'] *9.81 # N
    # Dados de motor
    TSFC = dados_aeronave["engine"]["SFC_cruise"] # N

    h_loitter = loitter_dict['h_loitter']
    loitter_time = loitter_dict['loitter_time']
    dT = loitter_dict['dT']
    dt = loitter_dict['dt']

    # Condições atmosféricas
    T = Atmosphere(h_loitter * 0.3048).temperature[0]   
    altitude = h_loitter * 0.3058 * T/(T + dT)
    P = Atmosphere(altitude).pressure #Dados atmosféricos da biblioteca Ambiance
    rho =  Atmosphere(altitude).density

    # Peso total e de combustível ao se iniciar cruzeiro
    W =  BOW + W_fuel
    W_init = W

    tempo = list(range(0,loitter_time,dt))

    for t in tempo:
        V_emax = np.sqrt(2*W/rho/S)*np.sqrt(np.sqrt(K/Cd0))
        Cl = (2 * (W/S))/(rho*V_emax**2)
        Cd = Cd0 + K * (Cl**2)
        E = Cl/Cd # Eficiência aerodinâmica
        Thrust0 = W/E # corrigindo o empuxo
        W = W - TSFC*dt*Thrust0 # peso da aeronave a cada faixa de tempo dt percorrida

    consumo_loitter = (W_init-W)/9.81

    return consumo_loitter


def land(dados_aeronave: dict, land_dict: dict):
    '''
    Implementação baseada na teoria apresentada por Raymer.
    Feita de modo a ser compatível com a FAR 23.

    Inputs:
    dados_aeronave (dict): Dicionário contendo os dados da aeronave, incluindo peso, área da asa, etc.
    gamma_descent (float): Ângulo de subida em radianos.
    Cl_run (float): Coeficiente de sustentação durante a corrida de decolagem.
    mi (float): Coeficiente de atrito entre o trem de pouso livre e a superfície.
    mi_break (float): Coeficiente de atrito entre o trem de pouso freado e a superfície.
    rho (float): Densidade do ar em kg/m^3.
    g (float): Aceleração devido à gravidade em m/s^2.
    h_obstacle (float): Altura do obstaculo em m.
    rev (float): Porcentagem da tração maxima para reversor.

    Returns:
    dists (list): Distancias.
    '''

    # Decomposição dos dados
    S = dados_aeronave['wing']['S']
    W = dados_aeronave['weights']['MTOW'] * 9.81
    CD0 = dados_aeronave['coeficients']['CD0']
    V_stall = dados_aeronave['speeds']['V_stall']
    # Dados de motor
    eta = dados_aeronave["engine"]['eta'] 

    K = 1/(np.pi * dados_aeronave['wing']['AR'] * dados_aeronave['wing']['e'])

    gamma = land_dict['gamma_descent']
    CL = land_dict['Cl_run']
    mi = land_dict['mi']
    mi_break = land_dict['mi_break']
    rho = land_dict['rho']
    h_obstacle = land_dict['h_obstacle']
    rev = land_dict['reversores']

    V = 1.15 * V_stall
    T = eta * dados_aeronave["engine"]["power"]/V

    ### Flare
    # Def: n = 1.2
    n = 1.2
    # n = 1 + V_tr**2/R.g
    R = (1.2*V**2)/(9.81*(n-1))
    h_f = R * (1 - np.cos(np.deg2rad(gamma)))
    SF = np.sqrt(R**2 - (R - h_f)**2)

    ### Aproach
    SA = -(h_obstacle - h_f)/np.tan(np.rad2deg(gamma))

    ### Ground roll
    # Break free for 3s
    SFR =  0
    KT = (0/W - mi)
    KA = (rho/(2*W/S))*(CD0 + K*CL**2  - mi*CL)
    for dt in np.diff(np.linspace(0,3,100)):
        a = 9.81 * (KT - KA*V**2)
        SFR = SFR + V * dt + 0.5 * a * dt**2
        V = V + a * dt

    KT = (-rev * T/W - mi_break)
    KA = (rho/(2*W/S))*(CD0 + K*CL**2  - mi_break*CL)
    # Break roll
    SB = -(1/(2*9.81*KA)) * np.log((KT)/(KT - KA*(V)**2))

    return [SA, SF, SFR, SB]

