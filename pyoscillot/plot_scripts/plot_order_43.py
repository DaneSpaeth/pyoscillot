import dataloader
from dataloader import carmenes_template, phoenix_spectrum
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from physics import planck

def plot_1():
    template_file = Path("/home/dane/Documents/PhD/plapy/data/RV_lib/serval/SIMULATION/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER/CARMENES_VIS/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER.fits")

    root = Path("/home/dane/mounted_srv/simulations/fake_spectra/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER/CARMENES_VIS")
    files = sorted(list(root.glob("*.fits")))

    dataloader.DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")
    (templ_spec, templ_cont, templ_sig, templ_wave) = carmenes_template(template_file, serval_output=True)
    file = files[1]
    (spec, cont, sig, wave) = carmenes_template(file)

    phoenix_wave, phoenix_spec, header = phoenix_spectrum(3300, 5.0, -0.5, wavelength_range=(6500, 6700))

    order = 22

    fig, ax = plt.subplots(1, figsize=(16, 9))

    templ_spec[order] = templ_spec[order]/np.nanmedian(templ_spec[order])*np.nanmedian(spec[order])
    phoenix_spec = phoenix_spec/np.nanmedian(phoenix_spec)*np.nanmedian(spec[order])
    # ax.plot(templ_wave[order], templ_spec[order], color="red", linestyle="--", label="SERVAL template")
    ax.plot(wave[order], spec[order], color="blue", linestyle="-", label=f"{file.name}")

    print(phoenix_wave)
    # ax.plot(phoenix_wave, phoenix_spec, color="green", linestyle="-", label=f"PHOENIX", alpha=0.5)
    ax.legend()
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("FLux")
    physical_order = 118 - order
    ax.set_title(f"Order:{order}/{physical_order}")
    plt.show()

def plot_planck(T_star=2900, T_spot=1900):
    lin_wave = np.linspace(520, 960, 1000)
    lin_wave = lin_wave*1e-9

    order_file = Path("/home/dane/Documents/PhD/pypulse/data/WaveRange_VIS.txt")
    data = np.loadtxt(order_file)
    order_idx = data[:,0].astype(int)
    order = data[:, 1].astype(int)
    wave_begin = data[:,2]
    wave_end = data[:,1]

    planck_star = planck(lin_wave, T_star)
    planck_spot = planck(lin_wave, T_spot)

    fig, ax = plt.subplots(2, figsize=(16,9), sharex=True)
    ax[0].plot(lin_wave*1e9, planck_star, color="orange", label="Star")
    ax[0].plot(lin_wave*1e9, planck_spot, color="black", label="Spot")

    ax[1].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Flux")
    ax[0].legend()
    ax[1].plot(lin_wave*1e9, planck_star - planck_spot)
    ax[1].set_ylabel("Flux Difference")

    for a in ax:
        for idx, o, beg, end in zip(order_idx, order, wave_begin, wave_end):
            beg /= 10
            end /= 10
            ylim = a.get_ylim()
            a.vlines(beg, 0, ylim[1], color="black", linestyles="--")
            a.text(beg, ylim[0]*1.1, f"{idx}")
            a.set_ylim(ylim)
    plt.show()





if __name__ == "__main__":
    plot_1()




