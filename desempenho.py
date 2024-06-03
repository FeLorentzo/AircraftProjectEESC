import numpy as np
from ambiance import Atmosphere

# TODO: 
# - Implementar decolagem
# - Implementar pouso
# - Compatibilizar input com dados do dict Aircraft

# Inputs
dados_aeronave = {
    "AR": 8.90,
    "e": 0.85,
    "CD0": 0.025,
    "MTOW": 450_300, # N
    "W_fuel": 130_000, # N
    "thrust": 92300, # N
    "S": 92.5, # m**2
    "TSFC": 0.85/3600, # N/s/N
    "BOW": 320_300, # N peso da aeronave sem combustivel
    'Cl_max' : 2
}

# Condições de voo
condicoes = {
    "H_partida": 0,
    "ISA_destino": 15 , # ISA + 15
    "H_max": 41_000, #ft
    "H_final": 0,
    "d_alternativa": 200, #nm
    "H_cruise": 20_000, # ft
    "ISA_alternativa" : 20, # ISA+20
    "H_loiter": 10_000, #ft
    "dT_loiter": 30, # min
    "H_alternativa": 2000, # ft
}

decolagem = {
   'gamma_climb': 1,
    'Cl_run': 0.1, 
    'Cl_max': 2, 
    'mi': 0.01, 
    'rho': 1.225, 
    'g': 9.81, 
    'V_i': 0, 
    'V_f': float, 
    'V_stall': float, 
    'h_obstacle': float
}

# Outros
regime_long_range = {
    "M": 0.79,
    "RS_min": 500, # ft/min
    "dH_step_climb": 2000
}

def decolagem(dados_aeronave: dict, decolagem: dict):
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
    S = dados_aeronave['S']
    T = dados_aeronave['thrust']
    W = dados_aeronave['MTOW']
    CD0 = dados_aeronave['CD0']
    K = 1/(np.pi() * dados_aeronave['AR'] * dados_aeronave['e'])
    gamma = decolagem['gamma_climb']
    CL = decolagem['Cl_run']
    mi = decolagem['mi']
    rho = decolagem['rho']
    g = decolagem['g']
    V_i = decolagem['V_i']
    V_f = decolagem['V_f']
    V_stall = decolagem['V_stall']
    h_obstacle = decolagem['h_obstacle']

    ### Aceleração (V: 0 - 1.1*V_stall)
    # Modelo: Aceleração constante 
    #   M.a(t) = T - D - mi.(W-L)
    #   a(t) = g.[(T/W - mi) - (rho/(W/S)).(CD0 + K.CL**2  - mi.CL).V**2]
    #   a(t) = g.[KT- KA.V**2]; KT=(T/W - mi); KA = (rho/(W/S)).(CD0 + K.CL**2  - mi.CL)

    KT = T/W - mi
    KA = (rho/(W/S)) * (CD0 + K.CL**2  - mi.CL)

    #   s(t) = \int_{v_i}^{v_f}(V/a)dV = 0.5 * \int_{v_i}^{v_f}(1/a)d(V**2)
    #   s(t) = (1/2gKA).ln((KT + KA.(V_f)**2)/(KT + KA.(V_i)**2))

    SG = (1/(2*g*KA)) * np.ln((KT + KA*(V_f)**2)/(KT + KA*(V_i)**2))

    ### Transição (V: 1.1*V_stall - 1.2*V_stall)
    # Def: n = 1.2
    n = 1.2
    # n = 1 + V_tr**2/R.g
    R = (1.2*V_stall**2)/(g*(n-1))
    h_R = R * (1 - np.cos(np.deg2rad(gamma)))
    ST = np.sqrt(R**2 - (R - h_R)**2)

    # Climb

    SC = (h_obstacle - h_R)/np.tan(np.rad2deg(gamma))

    return [SG, ST, SC]

def subida(dados_aeronave: dict, h0:float, hf:float, W_fuel_init:float|None = None, vmin:float=100.0, vmax:float= 300.0, dT:float = 0) -> tuple:

    ## Dados da aeronave
    AR=dados_aeronave["AR"]
    e=dados_aeronave["e"]
    Cd0 = dados_aeronave["CD0"]
    BOW = dados_aeronave["BOW"] # N
    W_fuel = dados_aeronave['W_fuel'] # N
    Thrust0 = dados_aeronave["thrust"] # N
    S = dados_aeronave["S"]
    TSFC = dados_aeronave["TSFC"]
    K = 1/(np.pi*AR*e)

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


    # variáveis auxiliares
    consumos = []
    distancias = []
    tempos = []
    RCs_max = []

    discretização = 100  # intervalos de altitudes
    alturas = np.arange(h0, hf+discretização, discretização)

    #print('alturas = ', alturas)
    for h_ind in alturas:
        # Corrige a variação de temperatura ISA
        T      = Atmosphere(h_ind * 0.3048).temperature[0]                      # altitude em metros
        h_real = h_ind * (T + dT)/T

        # Calculo
        sigma = (Atmosphere(h_real * 0.3048).density)[0]/rho0
        M     = (V)/Atmosphere(h_real * 0.3048).speed_of_sound[0]               # V (true aero speed)  (V*sigma**0.5)= Ve (Velocidade equivalente)
        phi   = 1/(0.7 * M**2)  *  (((1 + 0.2 * M**2)**3.5 - 1)/((1 + 0.2 * M**2)**2.5))
        f_a   =  0.7 * (M**2) * (phi-(0.190263 * (T+dT)/T))

        # Variação do empuxo
        Thrust = Thrust0 * sigma**0.7                                           # empuxo para menor q troposfera
        if h_real > 36089:
            Thrust = 1.439 * Thrust0 * sigma                                    # empuxo para altitudes maiores q troposfera
            f_a    = phi * 0.7 * M**2                                           # fator de aceleração para altitudes maiores q troposfera

        ## Calcula Consumo
        consumo = TSFC * Thrust * t
        consumos.append(consumo)
        W = W - consumo

        Cl = (2 * (W/S))/( Atmosphere(h_real*0.3048).density * V**2)
        Cd = Cd0 + K * (Cl**2)
        E = Cl/Cd

        # Calculo da razão de subida máxima?
        gamma = np.arcsin(Thrust/W - 1/E)
        RC_1 = V * np.sin(gamma)
        RC = RC_1/(1 + f_a)
        RCs_max.append(max(RC))
        index = np.where(RC == max(RC))                                         # pega o index do vetor RC em que RC é máximo
        v_rcmax = V[index][0]                                                   # pega a velocidade em que RC é máximo
        gamma_rcmax = gamma[index][0]                                           # pega o gama em que RC é máximo
        v_horizontal = v_rcmax * np.cos(gamma_rcmax)                            # calcula a velocidade horizontal em que RC é máximo para determinada faixa de altitude


        t = discretização * ((T + dT)/T) * 0.3048 / max(RC)                     # Tempo para o maximo RC de cada altitude = faixa de altitude/RC
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

    return tempo_total, consumo_total, dist_percorrida, FL


def cruseiro(dados_aeronave: dict,  h0:float, hf:float, W_fuel_init:float, mach:float = 0.79, W_reserve_fuel:float=10_000, step_climb:bool=True):
    # Deve retornar o tempo decorrido, combustivel consumido, distância percorida e variação da altura
    
    ## Dados da aeronave
    AR= dados_aeronave["AR"]
    BOW= dados_aeronave["BOW"]
    e = dados_aeronave["e"]
    CD0 = dados_aeronave["CD0"]
    MTOW = dados_aeronave["MTOW"] # N
    S = dados_aeronave["S"]
    TSFC = dados_aeronave["TSFC"]              
    K = 1/(np.pi*AR*e)

    ## Discretização 
    dt=100    

    ## Pesos
    W = BOW + W_fuel_init        # Peso total no inicio do cruzeiro
    W_min_cruseiro = BOW + W_reserve_fuel 
    
    # Acumuladores
    distancia_total=0
    dist=0
    t = 0
    h = h0 * 0.3048

    while True:
      t+=dt

      ## Condições de voo
      a = Atmosphere(h).speed_of_sound  
      rho = Atmosphere(h).density    
      V = mach*a

      ## Coeficientes 
      Cl = (2 * (W/S))/( rho * V**2) #calculo do cl
      Cd = CD0 + K * (Cl**2)
      E = Cl/Cd
      T = W/E

      ## Atualiza peso da aeronave
      W -= TSFC*dt*T

      ## Avalia condição de fim de cruseiro
      if  W < W_min_cruseiro:
        consumo_cruseiro = BOW + W_fuel_init - W
        return  t, consumo_cruseiro, distancia_total, h/0.3048
      
      if step_climb:
        ## Avalia possibilidade de um step-climb
        tempo_subida, consumo_subida, dist_percorrida_subida, FL_atual = subida(dados_aeronave, h/0.3048, hf, W_fuel_init= (W - BOW))
        
      h = FL_atual * 0.3048
      W -= consumo_subida
      t += tempo_subida

    # Calcula a distância percorrida
      dist = V * dt # distancia percorrida para cada faixa de tempo
      distancia_total = distancia_total + dist + dist_percorrida_subida # soma de distancias percorridas para cada velocidade ideal


def descida(dados_aeronave: dict, h0:float, hf:float, W_fuel_init:float, vmin:float=100.0, vmax:float= 300.0, dT:float = 15, dt:float = 1, Thrust_percent:float = 50):
    ## Dados da aeronave
    AR=dados_aeronave["AR"]
    e=dados_aeronave["e"]
    Cd0 = dados_aeronave["CD0"]
    BOW= dados_aeronave["BOW"]
    Thrust0 = dados_aeronave["thrust"] # N
    S = dados_aeronave["S"]
    TSFC = dados_aeronave["TSFC"]
    K = 1/(np.pi*AR*e)

    W0 =  BOW + W_fuel_init # N
    W  =  BOW + W_fuel_init # N
    Thrust_descent = Thrust0*Thrust_percent/100 # N thrust chute meu

    T0   =  Atmosphere(0).temperature
    P0   =  Atmosphere(0).pressure
    rho0 =  Atmosphere(0).density

    # Cria um range de velocidades a serem avaliadas
    V = np.linspace(vmin, vmax+1, 10)

    tempo_total=0
    distancia_descida = 0
    h = h0 * 0.3048

    while h > hf*0.3048:
        # Corrige a variação de temperatura ISA
        T = Atmosphere(h).temperature  # altitude em metros
        h_ind = h / (T + dT) *T

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
        
        # Calculo do consumo
        consumo=TSFC*Thrust*dt #  com fator de aceleração
        W -= consumo  # com fator de aceleração
        
        # Calculo dos coeficientes
        Cl = (2 * (W/S))/( rho * V**2)
        Cd = Cd0 + K * (Cl**2)
        D = (0.5 * rho * S * V**2) * Cd

        ## Calculo da razão de descidaw
        gamma = np.arcsin((Thrust-D)/W)
        RD = V*np.sin(gamma)
        RD = RD/(1+f_a)
        index = np.where(RD == min(RD)) # pega o index do vetor RC em que RC é máximo
        v_rdmin = V[index][0]           # pega a velocidade em que RD é máximo
        gamma_rdmin = gamma[index][0]   # pega o gama em que RC é máximo
        v_horizontal = v_rdmin*np.cos(gamma_rdmin) #calcula a velocidade horizontal em que RC é máximo para determinada faixa de altitude
        v_vertical   = v_rdmin*np.sin(gamma_rdmin)

        tempo_total+= dt
        distancia_descida += v_horizontal*dt
        h += v_vertical*dt

    print(np.rad2deg(gamma))
    consumo_descida = W0 - W

    return  tempo_total, consumo_descida, distancia_descida, h/0.3048
    
def loitter():

    ## Dados da aeronave
    AR= dados_aeronave["AR"]
    e = dados_aeronave["e"]
    Cd0 = dados_aeronave["CD0"]

    MTOW = dados_aeronave["MTOW"] # N
    Thrust0 = dados_aeronave["thrust"] # N
    S = dados_aeronave["A"]
    TSFC = dados_aeronave["TSFC"]
    BOW = dados_aeronave["BOW"]
    K = 1/(np.pi*AR*e)



    # Condições atmosféricas
    isa = +15
    altitude = 10000*0.3058 # 10000 pés
    temp = Atmosphere(altitude).temperature - isa #Dados atmosféricos da biblioteca Ambiance
    a = Atmosphere(altitude).speed_of_sound #Dados atmosféricos da biblioteca Ambiance
    P = Atmosphere(altitude).pressure #Dados atmosféricos da biblioteca Ambiance
    rho =  Atmosphere(altitude).density


    # Peso total e de combustível ao se iniciar cruzeiro


    W =  BOW + 14100 + 9350 # N# Peso total no inicio da espera precisa mudar aqui
    Win = W


    # Cd e eficiência no início do cruzeiro


    dt = 10 # faixa de tempo de 100s para processo iterativo
    tempo = list(range(0,30*60,dt))

    for t in tempo:
      V_emax = np.sqrt(2*W/rho/S)*np.sqrt(np.sqrt(K/Cd0))
      Cl = (2 * (W/S))/( rho*V_emax**2)
      Cd = Cd0 + K * (Cl**2)
      E = Cl/Cd # Eficiência aerodinâmica
      T = W/E # corrigindo o empuxo
      W = W - TSFC*dt*T # peso da aeronave a cada faixa de tempo dt percorrida

    print('gasto total de combustível [N] =', Win-W)

def pouso():
   raise NotImplementedError


def main()->None:
   '''Main code'''



if __name__ == '__main__':
   main()
