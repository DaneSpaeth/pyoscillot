from pathlib import Path
from animation_pulsation_rvo import animate_rv_lambda
from animation_pulsation import animate_pulse
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

if __name__ == "__main__":
    root = Path("/home/dane/mounted_srv/simulations/fake_spectra")
    names = ["pulsation_dT50", "pulsation_dT100", "pulsation_phase180"]
    names = ["pulsation_phase180"]
    for name in names:
        ticket_name = name
        if name == "pulsation_phase180":
            ticket_name = "pulsation180"
        ticket = root / name / f"{ticket_name}.ini"
        animate_rv_lambda(ticket)
        animate_pulse(ticket)