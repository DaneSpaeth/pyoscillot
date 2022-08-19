import numpy as np
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
from parse_ini import parse_global_ini
from plapy.rv.plotter import plot_rv, plot_activity, plot_activity_rv
from pathlib import Path


def check_time_series(name, instrument=None, reduction="serval", downscale_raccoon_errors=False):
    """ Plot a check plot.

            TODO: Change it to use the usual plot functions.
    """
    # Calculate V band photometry
    bjd, band_photometry = calc_photometry(name)

    # Read in RV, CRX and DLW
    fig, ax = plt.subplots(4, 2, figsize=(20, 10))
    rv_dict = load.rv(name)
    print(list(rv_dict.keys()))
    if reduction == "serval":
        # instrument = "CARMENES_VIS"
        activity1_dict = load.crx(name, NIR=True)
        activity2_dict = load.dlw(name, NIR=True)
        activity3_dict = load.halpha(name, NIR=True)
        try:
            rv_dict.pop("CARMENES_VIS_CCF")
            rv_dict.pop("CARMENES_NIR_CCF")
        except KeyError:
            pass
    else:
        # instrument = "CARMENES_VIS_CCF"
        activity1_dict = load.fwhm(name)
        activity2_dict = load.bis(name)
        activity3_dict = load.contrast(name)

        for inst in activity1_dict:
            offset = np.mean(activity1_dict[inst]["fwhm"])
            activity1_dict[inst]["fwhm"] = activity1_dict[inst]["fwhm"] - offset
        for inst in activity2_dict:
            offset = np.mean(activity2_dict[inst]["bis"])
            activity2_dict[inst]["bis"] = activity2_dict[inst]["bis"] - offset
        for inst in activity3_dict:
            offset = np.mean(activity3_dict[inst]["contrast"])
            activity3_dict[inst]["contrast"] = activity3_dict[inst]["contrast"] - offset

        # for act_dict in (activity1_dict, activity2_dict, activity3_dict):
        #     error_key = list(act_dict[instrument].keys())[-1]
        #     if downscale_raccoon_errors:
        #         act_dict[instrument][error_key] = act_dict[instrument][error_key] * 0.1

    # make all rvs around 0
    for inst in rv_dict:
        offset = np.mean(rv_dict[inst]["rv"])
        rv_dict[inst]["rv"] = rv_dict[inst]["rv"] - offset
    plot_rv(rv_dict, ax=ax[0, 0])
    plot_activity(activity1_dict, ax=ax[1, 0])
    plot_activity(activity2_dict, ax=ax[2, 0])
    plot_activity(activity3_dict, ax=ax[3, 0])
    plot_activity_rv(rv_dict, activity1_dict, ax=ax[1, 1], fit=None)
    plot_activity_rv(rv_dict, activity2_dict, ax=ax[2, 1], fit=None)
    plot_activity_rv(rv_dict, activity3_dict, ax=ax[3, 1], fit=None)
    fig.set_tight_layout(True)

    out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/26.07.2022")
    # plt.savefig(out_dir / "two_spots.png", dpi=300)
    plt.show()


def plot_temperature(name):
    """ Plot one array of the temperature maps.

        In the future plot it as a gif.
    """
    # Read in the flux
    global_dict = parse_global_ini()
    datapath = global_dict["datapath"]

    temperature_folder = datapath / "fake_spectra" / name / "arrays" / "temperature"
    array_paths = list(temperature_folder.glob("*.npy"))

    array = np.load(array_paths[0])
    plt.imshow(array, origin="lower", cmap="seismic")
    plt.show()

def plot_temperatures(name):
    """ Plot all temperature arrays."""
    global_dict = parse_global_ini()
    datapath = global_dict["datapath_laptop"]

    temperature_folder = datapath / "fake_spectra" / name / "arrays" / "temperature"
    array_paths = sorted(list(temperature_folder.glob("*.npy")))

    if len(array_paths) == 20:
        fig, ax = plt.subplots(4, 5, figsize=(16,9))
    elif len(array_paths) == 100:
        fig, ax = plt.subplots(5, 20, figsize=(16,9))

    for a, p in zip(ax.flatten(), array_paths):
        array = np.load(p)
        a.imshow(array, origin="lower", cmap="seismic")
    plt.show()


def calc_photometry(name, band="V"):
    """ Calculate the fluctuations in the V band."""
    global_dict = parse_global_ini()
    # Get raw spectra filesglobal_dict = parse_global_ini()
    rvlibpath = global_dict["rvlibpath"]
    RAW_folder = rvlibpath / name / "RAW"

    # print(RAW_folder)

    wave_files = sorted(list(RAW_folder.glob("wave_*.npy")))
    spec_files = sorted(list(RAW_folder.glob("spec_*.npy")))

    band_photometry = []
    bjd = []

    for wv_file, sp_file in zip(wave_files, spec_files):
        wave = np.load(wv_file)
        spec = np.load(sp_file)

        bjd.append(float(str(wv_file).split("wave_")[-1].split("_")[0]))

        if band == "V":
            band_center = 5510
            band_fwhm = 880
        wave_mask_low = wave > band_center - (band_fwhm / 2)
        wave_mask_high = wave < band_center + (band_fwhm / 2)
        wave_mask = np.logical_and(wave_mask_low, wave_mask_high)
        band_spec = spec[wave_mask]

        band_photometry.append(np.sum(band_spec))

    band_photometry = np.array(band_photometry)
    return bjd, band_photometry

def plot_vsini_series():
    """ Plot the vsini timeseries"""
    vsini = range(20, 70, 5)
    fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    colors = ["red", "blue", "green", "yellow", "pink", "purple", "black", "gray", "navy"]
    for v, c in zip(vsini, colors):
        name = f"YZ_CMi_TWO_SPOTS_vsini{v}"
        rv_dict = load.rv(name)
        crx_dict = load.crx(name, NIR=True)
        dlw_dict = load.dlw(name, NIR=True)

        label = f"vsin(i)={v/10}km/s"

        plot_rv(rv_dict, ax=ax[0, 0], instrument="CARMENES_VIS", label=label, color=c)
        plot_activity(crx_dict, ax=ax[1, 0], instrument="CARMENES_VIS", label=label, color=c)
        plot_activity(dlw_dict, ax=ax[2, 0], instrument="CARMENES_VIS", label=label, color=c)

        plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument="CARMENES_VIS", fit=None, label=label, color=c)
        plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1], instrument="CARMENES_VIS", fit=None, label=label, color=c)


    ax[1, 0].legend().remove()
    ax[1, 1].legend().remove()
    ax[2, 0].legend().remove()
    ax[2, 1].legend().remove()

    fig.set_tight_layout(True)

    out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/26.07.2022")
    plt.savefig(out_dir / "vsini_grid.png", dpi=300)
    # plt.show()

def plot_dT_series():
    """ Plot the dT timeseries"""
    dTs = range(100, 600, 100)
    fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    colors = ["red", "blue", "green", "yellow", "pink"]
    for dT, c in zip(dTs, colors):
        name = f"YZ_CMi_TWO_SPOTS_dT{dT}"
        rv_dict = load.rv(name)
        crx_dict = load.crx(name, NIR=True)
        dlw_dict = load.dlw(name, NIR=True)

        label = f"dT={dT}K"

        plot_rv(rv_dict, ax=ax[0, 0], instrument="CARMENES_VIS", label=label, color=c)
        plot_activity(crx_dict, ax=ax[1, 0], instrument="CARMENES_VIS", label=label, color=c)
        plot_activity(dlw_dict, ax=ax[2, 0], instrument="CARMENES_VIS", label=label, color=c)

        plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument="CARMENES_VIS", fit=None, label=label, color=c)
        plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1],instrument="CARMENES_VIS", fit=None, label=label, color=c)


    ax[1, 0].legend().remove()
    ax[1, 1].legend().remove()
    ax[2, 0].legend().remove()
    ax[2, 1].legend().remove()

    fig.set_tight_layout(True)

    out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/26.07.2022")
    plt.savefig(out_dir / "dT_grid.png", dpi=300)
    plt.show()

def plot_vsini_pulsation_series():
    """ Plot the vsini pulsation timeseries"""
    vsini = range(1, 6)
    colors = ["red", "blue", "green", "yellow", "black"]
    for channel in ["VIS", "NIR"]:
        fig, ax = plt.subplots(7, 2, figsize=(20, 10), sharex="col", sharey="row")
        for v, c in zip(vsini, colors):
            name = f"pulsation_vsini{v}"
            rv_dict = load.rv(name)
            crx_dict = load.crx(name, NIR=True)
            dlw_dict = load.dlw(name, NIR=True)
            halpha_dict = load.halpha(name)
            fwhm_dict = load.fwhm(name)
            bis_dict = load.bis(name)
            contrast_dict = load.contrast(name)

            label = f"vsini={v}km/s"

            for inst in rv_dict.keys():
                rv_dict[inst]["rv"] -= np.mean(rv_dict[inst]["rv"])

            # normalize fwhm, bis, contrast
            inst = f"CARMENES_{channel}_CCF"
            fwhm_dict[inst]["fwhm"] -= np.mean(fwhm_dict[inst]["fwhm"])
            contrast_dict[inst]["contrast"] -= np.mean(contrast_dict[inst]["contrast"])
            bis_dict[inst]["bis"] -= np.mean(bis_dict[inst]["bis"])

            inst = f"CARMENES_{channel}"
            halpha_dict[inst]["halpha"] -= np.mean(halpha_dict[inst]["halpha"])

            plot_rv(rv_dict, ax=ax[0, 0], instrument=f"CARMENES_{channel}", label=label, color=c)
            plot_activity(crx_dict, ax=ax[1, 0], instrument=f"CARMENES_{channel}", label=label, color=c,
                          short_label=True)
            plot_activity(dlw_dict, ax=ax[2, 0], instrument=f"CARMENES_{channel}", label=label, color=c,
                          short_label=True)
            plot_activity(halpha_dict, ax=ax[3, 0], instrument=f"CARMENES_{channel}", label=label, color=c,
                          short_label=True)
            plot_activity(fwhm_dict, ax=ax[4, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c,
                          short_label=True)
            plot_activity(bis_dict, ax=ax[5, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c,
                          short_label=True)
            plot_activity(contrast_dict, ax=ax[6, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c,
                          short_label=True)

            plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument=f"CARMENES_{channel}", fit=None, label=label,
                             color=c, short_label=True)
            plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1], instrument=f"CARMENES_{channel}", fit=None, label=label,
                             color=c, short_label=True)
            plot_activity_rv(rv_dict, halpha_dict, ax=ax[3, 1], instrument=f"CARMENES_{channel}", fit=None, label=label,
                             color=c, short_label=True)
            plot_activity_rv(rv_dict, fwhm_dict, ax=ax[4, 1], instrument=f"CARMENES_{channel}_CCF", fit=None,
                             label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, bis_dict, ax=ax[5, 1], instrument=f"CARMENES_{channel}_CCF", fit=None,
                             label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, contrast_dict, ax=ax[6, 1], instrument=f"CARMENES_{channel}_CCF", fit=None,
                             label=label, color=c, short_label=True)

        for i in range(1, ax.shape[0]):
            ax[i, 0].legend().remove()
            ax[i, 1].legend().remove()

        plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0, wspace=0.03)

        out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/04.09.2022")
        plt.savefig(out_dir / f"{channel}_vsini_pulsation_grid.png", dpi=300)
        plt.close()
    # plt.show()

def plot_phase_pulsation_series():
    """ Plot the phase pulsation timeseries"""
    phases = [0, 90, 180, 270]
    colors = ["red", "blue", "green", "black"]
    for channel in ["VIS", "NIR"]:
        fig, ax = plt.subplots(7, 2, figsize=(20, 10), sharex="col", sharey="row")
        for p, c in zip(phases, colors):
            name = f"pulsation_phase{p}"
            rv_dict = load.rv(name)
            crx_dict = load.crx(name, NIR=True)
            dlw_dict = load.dlw(name, NIR=True)
            halpha_dict = load.halpha(name)
            fwhm_dict = load.fwhm(name)
            bis_dict = load.bis(name)
            contrast_dict = load.contrast(name)

            label=f"T_phase={p}°"

            for inst in rv_dict.keys():
                rv_dict[inst]["rv"] -= np.mean(rv_dict[inst]["rv"])

            # normalize fwhm, bis, contrast
            inst = f"CARMENES_{channel}_CCF"
            fwhm_dict[inst]["fwhm"] -= np.mean(fwhm_dict[inst]["fwhm"])
            contrast_dict[inst]["contrast"] -= np.mean(contrast_dict[inst]["contrast"])
            bis_dict[inst]["bis"] -= np.mean(bis_dict[inst]["bis"])

            inst = f"CARMENES_{channel}"
            halpha_dict[inst]["halpha"] -= np.mean(halpha_dict[inst]["halpha"])

            plot_rv(rv_dict, ax=ax[0, 0], instrument=f"CARMENES_{channel}", label=label, color=c)
            plot_activity(crx_dict, ax=ax[1, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(dlw_dict, ax=ax[2, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(halpha_dict, ax=ax[3, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(fwhm_dict, ax=ax[4, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)
            plot_activity(bis_dict, ax=ax[5, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)
            plot_activity(contrast_dict, ax=ax[6, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)

            plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, halpha_dict, ax=ax[3, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, fwhm_dict, ax=ax[4, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, bis_dict, ax=ax[5, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, contrast_dict, ax=ax[6, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)


        for i in range(1, ax.shape[0]):
            ax[i, 0].legend().remove()
            ax[i, 1].legend().remove()

        plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0, wspace=0.03)

        out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/04.09.2022")
        plt.savefig(out_dir / f"{channel}_phase_pulsation_grid.png", dpi=300)
        plt.close()

def plot_dT_pulsation_series():
    """ Plot the dT pulsation timeseries"""
    temps = [25, 50, 75, 100, 125, 150, 175, 200]
    colors = ["red", "blue", "green", "black", "purple", "yellow", "cyan", "orange"]
    for channel in ["VIS", "NIR"]:
        fig, ax = plt.subplots(7, 2, figsize=(20, 10), sharex="col", sharey="row")
        for t, c in zip(temps, colors):
            name = f"pulsation_dT{t}"
            rv_dict = load.rv(name)
            crx_dict = load.crx(name, NIR=True)
            dlw_dict = load.dlw(name, NIR=True)
            halpha_dict = load.halpha(name)
            fwhm_dict = load.fwhm(name)
            bis_dict = load.bis(name)
            contrast_dict = load.contrast(name)

            label=f"dT={t}K"

            for inst in rv_dict.keys():
                rv_dict[inst]["rv"] -= np.mean(rv_dict[inst]["rv"])

            # normalize fwhm, bis, contrast
            inst = f"CARMENES_{channel}_CCF"
            fwhm_dict[inst]["fwhm"] -= np.mean(fwhm_dict[inst]["fwhm"])
            contrast_dict[inst]["contrast"] -= np.mean(contrast_dict[inst]["contrast"])
            bis_dict[inst]["bis"] -= np.mean(bis_dict[inst]["bis"])

            inst = f"CARMENES_{channel}"
            halpha_dict[inst]["halpha"] -= np.mean(halpha_dict[inst]["halpha"])

            plot_rv(rv_dict, ax=ax[0, 0], instrument=f"CARMENES_{channel}", label=label, color=c)
            plot_activity(crx_dict, ax=ax[1, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(dlw_dict, ax=ax[2, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(halpha_dict, ax=ax[3, 0], instrument=f"CARMENES_{channel}", label=label, color=c, short_label=True)
            plot_activity(fwhm_dict, ax=ax[4, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)
            plot_activity(bis_dict, ax=ax[5, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)
            plot_activity(contrast_dict, ax=ax[6, 0], instrument=f"CARMENES_{channel}_CCF", label=label, color=c, short_label=True)

            plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, halpha_dict, ax=ax[3, 1], instrument=f"CARMENES_{channel}", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, fwhm_dict, ax=ax[4, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, bis_dict, ax=ax[5, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)
            plot_activity_rv(rv_dict, contrast_dict, ax=ax[6, 1], instrument=f"CARMENES_{channel}_CCF", fit=None, label=label, color=c, short_label=True)


        for i in range(1, ax.shape[0]):
            ax[i, 0].legend().remove()
            ax[i, 1].legend().remove()

        ax[0,0].legend(ncol=2)

        plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0, wspace=0.03)

        out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/04.09.2022")
        plt.savefig(out_dir / f"{channel}_dT_pulsation_grid.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    # bjd, band_photometry = calc_photometry("TALK_0")
    # plt.plot(bjd, band_photometry / np.median(band_photometry))
    # plt.show()

    # name = "PULSATION_l1_m1_phase0"
    # check_time_series(name, reduction="raccoon")
    #plot_vsini_series()
    #plot_dT_series()
    #fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    # plot_temperatures(name)
    # plot_vsini_pulsation_series()
    #plot_phase_pulsation_series()
    # plot_dT_pulsation_series()
    name = "PHOTON_FLUX_REBINNED"
    check_time_series(name, reduction="serval")