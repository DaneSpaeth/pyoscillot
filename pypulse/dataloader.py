from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wget
from parse_ini import parse_global_ini
import idlsave

global_dict = parse_global_ini()
DATAROOT = global_dict["datapath"]


def phoenix_spectrum(Teff=4800, logg=3.0, feh=-0.5, wavelength_range=(3000, 7000)):
    """Return phenix spectrum and header."""
    Teff, logg, feh = _check_and_prepare_for_phoenix(Teff, logg, feh)

    folder = DATAROOT / "phoenix_spectra"

    filename = f"lte{int(Teff):05d}-{logg:.2f}{feh:+.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    file = folder / filename

    if not file.is_file():
        print(f"{file} not found. Begin download!")
        download_phoenix(file.name, folder, feh)

    print(f"Load file {file}")
    with fits.open(file) as hdul:
        header = hdul[0].header
        spectrum = hdul[0].data

    wavelength_file = folder / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

    with fits.open(wavelength_file) as hdul:
        wavelength = hdul[0].data

    if wavelength_range:
        wavelength_mask = np.logical_and(wavelength >= wavelength_range[0],
                                         wavelength <= wavelength_range[1],)
        spectrum = spectrum[wavelength_mask]
        wavelength = wavelength[wavelength_mask]

    wavelength = wavelength.astype("float64")
    spectrum = spectrum.astype("float64")

    return wavelength, spectrum, header


def _check_and_prepare_for_phoenix(Teff, logg, feh):
    """ Check the values and prepare for filename determination.

        This is a refactored function to make the code DRY.
    """
    assert 2300 <= Teff <= 12000, f"Teff={Teff} out of range [2300, 12000]"
    assert 0.0 <= logg <= 6.0, f"logg={logg} out of range [0.0, 6.0]"
    assert -4.0 <= feh <= 1.0, f"[Fe/H]={feh} out of range [-4.0, 1.0]"

    # Round to full .5
    logg = round(logg / 5, 1) * 5
    feh = round(feh / 5, 1) * 5
    # Round to full 100
    Teff = round(Teff / 100, 0) * 100

    # Give feh=0 a small negative number such that the sign operation
    # in the string formatting gives a minus (this is how it works for PHOENIX)
    if feh == 0.0:
        feh = -0.0000001

    return Teff, logg, feh


def phoenix_spec_intensity(Teff=4800, logg=3.0, feh=-0.5, wavelength_range=(3000, 7000)):
    """Return phenix spectrum and header."""
    Teff, logg, feh = _check_and_prepare_for_phoenix(Teff, logg, feh)

    folder = DATAROOT / "phoenix_spec_intensities"

    filename = f"lte{int(Teff):05d}-{logg:.2f}{feh:+.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits"
    file = folder / filename

    if not file.is_file():
        print(f"File not found. Begin download!")
        download_phoenix(file.name, folder, feh, spec_intensity=True)

    print(f"Load file {file}")
    with fits.open(file) as hdul:
        header = hdul[0].header
        # The wave grid is essentially defined with a reference pixel
        # with a reference wavelength in Angstrom and a step size
        wave_step_size = header["CDELT1"]
        wave_reference = header["CRVAL1"]
        wave_ref_pix = header["CRPIX1"]

        msg = "The wavelength grid is not in Angstrom"
        assert header["CUNIT1"] == "Angstrom", msg
        # The spec_int will be an array of shape (25500, 78) with the first
        # axis being along the spectrum and the second column corresponding
        # to the values stored in the mu array
        spec_int = hdul[0].data
        # This will be a 1d array of length 78 containing all the values of mu
        mu = hdul[1].data
        # Not sure if I will need these
        abundances = hdul[2].data

    wave_idx = np.arange(0, spec_int.shape[1], 1)
    # We need the 1 since I believe it starts indxing with 1 in the FITS header
    wavelength = (wave_reference +
                  ((wave_idx + 1) - wave_ref_pix) * wave_step_size)

    if wavelength_range:
        wavelength_mask = np.logical_and(wavelength >= wavelength_range[0],
                                         wavelength <= wavelength_range[1],)
        spec_int = spec_int[:, wavelength_mask]
        wavelength = wavelength[wavelength_mask]

    print(spec_int.shape)

    return wavelength, spec_int, mu, header


def download_phoenix(filename, out_dir, feh, spec_intensity=False):
    """ Download the file from the PHOENIX ftp server.

        :param filename: Name of file to download.
        :param out_dir. pathlib.Path to output directory.
        :param float feh: Metallicity of the star (needed again for the correct
                          ftp folder)
    """
    if not spec_intensity:
        ftp_root = f"ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{feh:+.1f}/"
    else:
        ftp_root = f"ftp://phoenix.astro.physik.uni-goettingen.de/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/Z{feh:+.1f}/"
    ftp_link = ftp_root.strip() + filename.strip()

    print(f"Download from {ftp_link}")
    wget.download(ftp_link, out=str(out_dir / filename))
    print("Download complete!")
    print(f"Saved to {str(out_dir / filename)}")


def carmenes_template(filename="CARMENES_VIS_template.fits", serval_output=False):
    """ Return spec, sig, cont and wave of Carmenes template."""
    if Path(filename).is_absolute():
        template = filename
    else:
        template = DATAROOT / filename
    print(f"Load template from {template}")
    with fits.open(template) as hdul:
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data
        if serval_output:
            wave = np.exp(sig)
            sig = None

    return (spec, cont, sig, wave)


def harps_template(spec_filename="HARPS_template_e2ds_A.fits",
                   wave_filename="HARPS_template_wave_A.fits",
                   blaze_filename="HARPS_template_blaze_A.fits"):
    """ Return spec, sig, cont and wave of Carmenes template."""
    spec_template = DATAROOT / spec_filename
    with fits.open(spec_template) as hdul:
        hdu = hdul[0]
        header = hdul[0].header
        spec = np.array(hdu.data)

    wave_template = DATAROOT / wave_filename
    with fits.open(wave_template) as hdul:
        hdu = hdul[0]
        wave = np.array(hdu.data)

    blaze_template = DATAROOT / blaze_filename
    with fits.open(blaze_template) as hdul:
        hdu = hdul[0]
        blaze = np.array(hdu.data)

    return (spec, wave, blaze)


def granulation_map():
    """ Laod a granulation map from Hans. At the moment always the same"""
    filenames = ["d3t50g25mm00n01.3-c.idlsave",
                 "d3t50g25mm00n01.d-p.idlsave", "d3t50g25mm00n01.q-z.idlsave", ]
    intensity = None
    for filename in filenames:
        file = DATAROOT / "granulation_Hans" / filename
        s = idlsave.read(file)
        print(len(s.intens))
        if intensity is None:
            intensity = s.intens
        else:
            intensity = np.concatenate((intensity, s.intens))
        # break

    return intensity


def plot_central_order_intensitites():
    color_dict = {"HIP73620": "green",
                  "fake": "purple"}
    for star, color in color_dict.items():
        if star == "fake":
            directory = Path(
                f"/home/dane/Documents/PhD/pypulse/data/fake_spectra/hip16335_talk_refined_highres")
        else:
            directory = Path(
                f"/home/dane/Documents/PhD/pyCARM/data/by_hip/{star}")
        files = directory.glob("*vis_A.fits")
        overall_mean_spec = []
        for file in files:
            (spec, cont, sig, wave) = carmenes_template(file)
            mid_idx = int(spec.shape[1] / 2)
            # mean_spec = np.mean(spec[:, mid_idx - 10:mid_idx + 10], axis=1)
            mean_spec = np.nanmean(spec, axis=1)
            mean_wave = wave[:, mid_idx]
            mean_spec = mean_spec / np.max(mean_spec)
            overall_mean_spec.append(mean_spec)

        # overall_mean_spec = np.array(overall_mean_spec)
        # mean_spec = np.mean(overall_mean_spec, axis=0)

            plt.plot(mean_wave, mean_spec, label=star, color=color)

    (spec, cont, sig, wave) = carmenes_template(file)
    mid_idx = int(spec.shape[1] / 2)
    mean_spec = np.mean(spec[:, mid_idx - 10:mid_idx + 10], axis=1)
    mean_wave = wave[:, mid_idx]
    mean_spec = mean_spec / np.max(mean_spec)
    plt.plot(mean_wave, mean_spec, label="template", color="black")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    file = "/home/dane/mounted_srv/simulations/fake_spectra/YZ_CMi_TWO_SPOTS_dT200/CARMENES_VIS/car-20220313T21h00m33s-sci-fake-vis_A.fits"

    file = "/home/dane/mounted_srv/simulations/fake_spectra/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER/CARMENES_VIS/car-20220430T18h01m32s-sci-fake-vis_A.fits"
    # file = "/home/dane/Documents/PhD/plapy/data/RV_lib/serval/SIMULATION/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER/CARMENES_VIS/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER.fits"
    template = "/home/dane/Documents/PhD/pypulse/data/CARMENES_templates/CARMENES_template_YZ_CMi.fits"
    (spec, cont, sig, wave) = carmenes_template(
        file, serval_output=False)



    def _plot_order_range(start, filename=None):
        out_root = Path("/home/dane/Documents/PhD/Sabine_overviews/02.08.2022/")
        fig, ax = plt.subplots(3, 4,figsize=(16,9))
        for order, a in zip(range(start, start+12), ax.flatten()):

            a.plot(wave[order], spec[order], label="YZ_CMi template")
            a.set_xlabel("Wavelength [A]")
            physical_order = 118 - order
            a.set_title(f"Order:{order}/{physical_order}")
            a.set_ylabel("Flux")
        fig.set_tight_layout(True)
        if filename:
            fig.suptitle(filename)
        plt.savefig(out_root / f"Orders{start}-{start+12}.png", dpi=300)
        plt.show()

    # _plot_order_range(10, filename=Path(file).name)
    # _plot_order_range(20, filename=Path(file).name)
    # _plot_order_range(30, filename=Path(file).name)
    _plot_order_range(40, filename=Path(file).name)
