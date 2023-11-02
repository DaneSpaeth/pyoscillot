from cfg import parse_global_ini, parse_ticket
from pathlib import Path
import numpy as np
from datetime import timedelta, datetime
import configparser

global_dict = parse_global_ini()
TICKETROOT = global_dict["ticketpath"]

#### INSERT YOUR NAMES ETC HERE ####
gridname = "NGC4349_improved_fine_grid_match_RV"
baseticket = TICKETROOT / gridname / "base.ini"

#### DEFINE THE RANGES THAT YOU WANT TO SIMULATE ####
# dt = np.arange(21, 24, 1, dtype=int)
# v_rot = np.arange(2000, 3500, 500, dtype=int)
# v_macro = np.arange(4000, 6000, 1000, dtype=int)
# v_p = np.arange(0.20727599796-0.03, 0.20727599796+0.04, 0.01)
# K = np.arange(2528-100, 2528+150, 50)
# dT = np.arange(1.0, 5.0, 1.0)
# T_phase = np.arange(0, 75+15, 15)
# inclination = np.arange(0.0, 105, 15)
step = 0.01
v_p = np.arange(0.28247640045, 0.28247640045+11*step, 0.01)
v_rot = np.arange(2000, 3200, 200)

vps, v_rots = np.meshgrid(v_p, v_rot)
# print(T_phases)
# v_rots, vps, Ks = np.meshgrid(v_rot, v_p, K)
# CB_models = ['alpha_boo', 'alpha_ari', 'alpha_sct', 'alpha_ser', 'alpha_uma', 'beta_boo', 'beta_cet', 'beta_gem', 'beta_oph', 'delta_dra', 'epsilon_cyg', 'epsilon_hya', 'epsilon_vir', 'eta_cyg', 'eta_dra', 'eta_her', 'eta_ser', 'gamma_psc', 'gamma_tau', 'iota_cep', 'kappa_cyg', 'kappa_per', 'mu_peg', 'nu_oph', 'nu_uma', 'rho_boo', 'xi_her', 'zeta_cyg']


for idx, (v_p, v_rot) in enumerate(zip(vps.flatten(), v_rots.flatten())):
    idx_plus = idx + 1
    print(idx)
    
    config = configparser.ConfigParser()
    config.read(baseticket)
    
    simname = f"{gridname}_{idx_plus:03d}"
    config["GLOBAL"]["name"] = simname
    config["GLOBAL"]["date"] = datetime.today().strftime("%d.%m.%Y")
    config["GLOBAL"]["v_rot"] = str(v_rot)
    config["pulsation"]["v_p"] = str(v_p)
    # config["GLOBAL"]["inclination"] = str(inclination)
    # config["pulsation"]["dt"] = str(round(dT, 1))
    # config["pulsation"]["t_phase"] = str(T_phase)
    
    # config["pulsation"]["v_p"] = str(v_p)
    
    
    outfile = TICKETROOT / gridname / f"{simname}.ini"
    with open(outfile, "w") as f:
        config.write(f)

# Save the settings
est_time = idx * timedelta(minutes=15)
print(est_time)

