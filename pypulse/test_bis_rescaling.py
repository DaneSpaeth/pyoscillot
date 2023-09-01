import numpy as np
import matplotlib.pyplot as plt
import cfg
from pathlib import Path
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from CB_models import simple_alpha_boo_CB_model

rootdir = cfg.parse_global_ini()["datapath"] / "BIS_rescale_tests"
files = sorted(list(rootdir.glob("R*.csv")))

colordict = {100000:"blue", 
             180000:"lime",
             250000: "tab:red",
             700000: "black"}
fig, ax = plt.subplots(1, 2, figsize=(6.35, 3.5), sharey=True)
Rs = []
factors = []
for i, file in enumerate(files):
    df = pd.read_csv(file)
    v = df["v"]
    bis = df["bis"]
    
    mask = bis < 0.9
    v = v[mask]
    bis = bis[mask]
    
    R = int(file.stem.split("R_")[-1])
    Rs.append(R)
    
    ax[0].plot(v, bis, color=colordict[R], marker=".", linestyle="None", label=f"R={R}")
    poly_fit = Polynomial.fit(bis,v, 5, window=(0, 1))
    lin_bis = np.linspace(0.27, 0.9)
    ax[0].plot(poly_fit(lin_bis), lin_bis, color=colordict[R])
    
    
    if R == 100000:
        v_scale = poly_fit(lin_bis)
        bis_scale = lin_bis
    factor = poly_fit(lin_bis) / v_scale
    if not R == 100000:    
        factors.append(factor)
    ax[1].plot(factor, lin_bis, color=colordict[R])

# integrated_params = np.zeros(6)
factors = np.array(factors)
R_test = 500000
factor_test = np.zeros(factors.shape[-1])
for i in range(factors.shape[-1]):

    factor_test[i]  = np.interp(R_test, Rs[1:], factors[:,i])

v_test = v_scale * factor_test
ax[0].plot(v_test, bis_scale, marker="s", markersize=3, linestyle="None", label=f"Interpolated to R={R_test}", color="tab:orange")
ax[1].plot(factor_test, lin_bis, color="tab:orange")
ax[0].set_xlabel("Doppler Velocity [m/s]")
ax[0].set_ylabel("Normalized Flux")
ax[1].set_xlabel("Scaling factor")


fig.subplots_adjust(wspace=0, left=0.1, right=.99, top=.99, bottom=0.1)
ax[0].legend(loc="upper right")
plt.savefig("bis_rescale.png", dpi=300)
plt.close()


BIS_model = simple_alpha_boo_CB_model()[1.0]
fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
lin_bis = np.linspace(0, 1, 100)
shift = np.mean(BIS_model(lin_bis))
ax.plot((BIS_model(lin_bis)) -shift, lin_bis, label="Alpha Boo BIS model")
ax.plot((BIS_model(lin_bis)-shift)*factor_test, lin_bis, label="Rescaled Alpha Boo BIS model")
fig.set_tight_layout(True)
plt.savefig("alpha_boo_rescale.png", dpi=300)
