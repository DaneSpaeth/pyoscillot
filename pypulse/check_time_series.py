import numpy as np
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
from parse_ini import parse_global_ini
from plapy.rv.plotter import plot_rv, plot_activity, plot_activity_rv


def check_time_series(name, instrument=None, reduction="serval", downscale_raccoon_errors=False):
    """ Plot a check plot.

            TODO: Change it to use the usual plot functions.
    """
    # Calculate V band photometry
    bjd, band_photometry = calc_photometry(name)

    # Read in RV, CRX and DLW
    fig, ax = plt.subplots(4, 2, figsize=(20, 10))
    rv_dict = load.rv(name)
    if reduction == "serval":
        instrument = "CARMENES_VIS"
        activity1_dict = load.crx(name)
        activity2_dict = load.dlw(name)
        activity3_dict = load.halpha(name)
    else:
        instrument = "CARMENES_VIS_CCF"
        activity1_dict = load.fwhm(name)
        activity2_dict = load.bis(name)
        activity3_dict = load.contrast(name)

        for act_dict in (activity1_dict, activity2_dict, activity3_dict):
            error_key = list(act_dict[instrument].keys())[-1]
            if downscale_raccoon_errors:
                act_dict[instrument][error_key] = act_dict[instrument][error_key] * 0.1

    plot_rv(rv_dict, ax=ax[0, 0], instrument=instrument)
    plot_activity(activity1_dict, ax=ax[1, 0], instrument=instrument)
    plot_activity(activity2_dict, ax=ax[2, 0], instrument=instrument)
    plot_activity(activity3_dict, ax=ax[3, 0], instrument=instrument)
    plot_activity_rv(rv_dict, activity1_dict, ax=ax[1, 1], instrument=instrument)
    plot_activity_rv(rv_dict, activity2_dict, ax=ax[2, 1], instrument=instrument)
    plot_activity_rv(rv_dict, activity3_dict, ax=ax[3, 1], instrument=instrument)
    fig.set_tight_layout(True)
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


if __name__ == "__main__":
    # bjd, band_photometry = calc_photometry("TALK_0")
    # plt.plot(bjd, band_photometry / np.median(band_photometry))
    # plt.show()

    name = "GRANULATION_HIGHRES"
    check_time_series(name)
    plt.savefig(f"/home/dane/Documents/PhD/Sabine_overviews/18.05.2022/{name}_serval.pdf")
    plt.close()
    check_time_series(name, reduction="raccoon", downscale_raccoon_errors=False)
    plt.savefig(f"/home/dane/Documents/PhD/Sabine_overviews/18.05.2022/{name}_raccoon.pdf")

    check_time_series(name, reduction="raccoon", downscale_raccoon_errors=True)
    plt.savefig(f"/home/dane/Documents/PhD/Sabine_overviews/18.05.2022/{name}_raccoon_errors_downscaled.pdf")
    plt.close()
    # check_time_series(
    #     "talk_ngc2423_0_dt50_k100_vrot3000_oldtemplate_snronlyheader", instrument="CARMENES_VIS")
