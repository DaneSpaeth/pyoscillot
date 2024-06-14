from pathlib import Path
from cfg import parse_global_ini, parse_ticket
import configparser


all_grid_inis = sorted(list((parse_global_ini()["ticketpath"] / "PhD_parameter_grids").rglob("PhD_param_grid_*.ini")))

for ini in all_grid_inis:
    config = configparser.ConfigParser()
    config.read(ini)
    
    simname = config["GLOBAL"]["name"]
    config["GLOBAL"]["name"] = simname+"_NEW"
    
    config["pulsation"]["T_phase"] = str(float(config["pulsation"]["T_phase"]))
    config["GLOBAL"]["date"] = "14.06.2024"
    
    outfile = ini.parent / f"{ini.stem}_NEW.ini"
    with open(outfile, "w") as f:
        config.write(f)
    # print(outfile)
    
    # exit()
    