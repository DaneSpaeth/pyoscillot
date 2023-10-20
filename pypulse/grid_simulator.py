from cfg import parse_global_ini, parse_ticket
from pathlib import Path
import numpy as np
from datetime import timedelta, datetime
import configparser

global_dict = parse_global_ini()
TICKETROOT = global_dict["ticketpath"]

#### INSERT YOUR NAMES ETC HERE ####
gridname = "NGC4349_lm_grid"
baseticket = TICKETROOT / gridname / "base.ini"

#### DEFINE THE RANGES THAT YOU WANT TO SIMULATE ####
# dt = np.arange(21, 24, 1, dtype=int)
# v_rot = np.arange(2000, 3500, 500, dtype=int)
# v_macro = np.arange(4000, 6000, 1000, dtype=int)
# v_p = np.arange(0.5, 0.7, 0.05)
# K = np.arange(2528-600, 2528+500, 200)

# v_rots, vps, Ks = np.meshgrid(v_rot, v_p, K)
# CB_models = ['alpha_boo', 'alpha_ari', 'alpha_sct', 'alpha_ser', 'alpha_uma', 'beta_boo', 'beta_cet', 'beta_gem', 'beta_oph', 'delta_dra', 'epsilon_cyg', 'epsilon_hya', 'epsilon_vir', 'eta_cyg', 'eta_dra', 'eta_her', 'eta_ser', 'gamma_psc', 'gamma_tau', 'iota_cep', 'kappa_cyg', 'kappa_per', 'mu_peg', 'nu_oph', 'nu_uma', 'rho_boo', 'xi_her', 'zeta_cyg']

ls = [1, 1, 1, 
      2, 2, 2, 2, 2, 
      3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 4]
ms = [1, 0, -1, 
      2, 1, 0, -1, -2, 
      3, 2, 1, 0, -1, -2, -3,
      4, 3, 2, 1, 0, -1, -2, -3, -4]
for idx, (l, m) in enumerate(zip(ls, ms)):
    idx_plus = idx + 1
    
    config = configparser.ConfigParser()
    config.read(baseticket)
    
    simname = f"{gridname}_{idx_plus:02d}"
    config["GLOBAL"]["name"] = simname
    config["GLOBAL"]["date"] = datetime.today().strftime("%d.%m.%Y")
    # config["GLOBAL"]["convective_blueshift_model"] = cb_model
    config["pulsation"]["l"] = str(l)
    config["pulsation"]["m"] = str(m)
    
    
    outfile = TICKETROOT / gridname / f"{simname}.ini"
    with open(outfile, "w") as f:
        config.write(f)
    # with open(overviewfile, "a") as f:
    #     f.write(f"{idx_plus}    {v_rot}    {v_macro}\n")

# Save the settings
# est_time = v_rots.size * timedelta(minutes=30)
# print(est_time)

