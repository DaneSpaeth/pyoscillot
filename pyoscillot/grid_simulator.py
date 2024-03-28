from cfg import parse_global_ini, parse_ticket
from pathlib import Path
import numpy as np
from datetime import timedelta, datetime
import configparser

global_dict = parse_global_ini()
TICKETROOT = global_dict["ticketpath"]



#### INSERT YOUR NAMES ETC HERE ####
# gridname = "NGC4349_l2_m-2_20K_grid"
gridname = "NGC4349_l2_m-2_vp_grid"
    # gridname = "NGC4349_l2_m-2_vp_grid"
    
PhD_root = TICKETROOT / "PhD_parameter_grids"
gridname = "PhD_param_grid_k"
grid_folder = PhD_root / gridname
if not grid_folder.is_dir():
    grid_folder.mkdir()
    
# baseticket = TICKETROOT / gridname / "base.ini"
# pyoscillot/tickets/TICKETS_USED_FOR_PAPER/PAPER_NGC4349-127_K-1_phase_full_dT2p5_vp03.ini was copied
baseticket = PhD_root / "base_ngc4349_127.ini"

ROOT = PhD_root

# First round
# Now based on nr 40 before
v_ps = [0.1 + i*0.1 for i in range(10)]

# Second round, based of 01 but still slightly smaller
# v_p = 0.18
v_rots = [500 + i*500 for i in range(12)]
v_macros = [500 + i*500 for i in range(16)]

ks = [0 + i*500 for i in range(10)]

# # New test grid, based on 15 -> smaller dTs and smaller v_rot

# # Only continue with dT = 5
# v_rots = [500+i*500 for i in range(2, 12)]
dTs = np.array([0.0 + i*1.0 for i in range(16)])

T_phases = [0. + i*30. for i in range(12)]
inclinations = [0 + i*15 for i in range(0, 7)]

# dTs, v_rots = np.meshgrid(dTs, v_rots)

# print(v_ps)
# exit()

#### DEFINE THE RANGES THAT YOU WANT TO SIMULATE ####
# dt = np.arange(21, 24, 1, dtype=int)
# v_rot = np.arange(2000, 3500, 500, dtype=int)
# v_macro = np.arange(4000, 6000, 1000, dtype=int)
# v_p = np.arange(0.20727599796-0.03, 0.20727599796+0.04, 0.01)
# K = np.arange(2528-100, 2528+150, 50)
# dT = np.arange(1.0, 5.0, 1.0)
# T_phase = np.arange(0, 75+15, 15)
# inclination = np.arange(0.0, 105, 15)
# step = 0.001
# v_p = np.arange(0.28247640045, 0.28247640045+30*step, step)
# v_rot = np.arange(1700, 2050, 50)

# ms = []
# ls = []
# for l in range(1, 5):
#     for m in range(-l, l+1):
#         ms.append(m)
#         ls.append(l)

# inclinations = [0.0, 23.9, 31.1, 40.9, 45.0, 54.7, 60., 90.]

# ms, v_rots = np.meshgrid(v_p, v_rot)
# print(T_phases)
# v_rots, vps, Ks = np.meshgrid(v_rot, v_p, K)
# CB_models = ['alpha_boo', 'alpha_ari', 'alpha_sct', 'alpha_ser', 'alpha_uma', 'beta_boo', 'beta_cet', 'beta_gem', 'beta_oph', 'delta_dra', 'epsilon_cyg', 'epsilon_hya', 'epsilon_vir', 'eta_cyg', 'eta_dra', 'eta_her', 'eta_ser', 'gamma_psc', 'gamma_tau', 'iota_cep', 'kappa_cyg', 'kappa_per', 'mu_peg', 'nu_oph', 'nu_uma', 'rho_boo', 'xi_her', 'zeta_cyg']



idx_plus = 0

for idx, (k) in enumerate(ks):
    idx_plus +=1
    # if dT == 10:
    #     # No need to run these models 
    #     continue
    
    config = configparser.ConfigParser()
    config.read(baseticket)
    
    simname = f"{gridname}_{idx_plus:02d}"
    config["GLOBAL"]["name"] = simname
    config["GLOBAL"]["date"] = datetime.today().strftime("%d.%m.%Y")
    # config["GLOBAL"]["v_rot"] = str(v_rot)
    # config["GLOBAL"]["v_macro"] = str(v_macro)
    # config["pulsation"]["v_p"] = str(round(v_p,3))
    # config["GLOBAL"]["inclination"] = str(i)
    # config["pulsation"]["dt"] = str(round(dT, 1))
    # config["pulsation"]["t_phase"] = str(T_phase)
    # config["pulsation"]["l"] = str(l)
    # config["pulsation"]["m"] = str(m)
    config["pulsation"]["k"] = str(k)
    
    # config["pulsation"]["v_p"] = str(v_p)
    
    
    outfile = ROOT / gridname / f"{simname}.ini"
    with open(outfile, "w") as f:
        config.write(f)
    # print(simname, l, m, inclination)
# 
# Save the settings
# est_time = idx_plus * timedelta(minutes=25)
# print(est_time)


