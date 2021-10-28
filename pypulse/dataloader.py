from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wget


# DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")
DATAROOT = Path(__file__).parent.parent / "data"


def phoenix_spectrum(Teff=4800, logg=3.0, feh=-0.5, wavelength_range=(3000, 7000)):
    """Return phenix spectrum and header."""
    assert 2300 <= Teff <= 12000, f"Teff={Teff} out of range [2300, 12000]"
    assert 0.0 <= logg <= 6.0, f"logg={logg} out of range [0.0, 6.0]"
    assert -4.0 <= feh <= 1.0, f"[Fe/H]={feh} out of range [-4.0, 1.0]"

    # Round to full .5
    logg = round(logg / 5, 1) * 5
    feh = round(feh / 5, 1) * 5
    # Round to full 100
    Teff = round(Teff / 100, 0) * 100

    folder = DATAROOT / "phoenix_spectra"

    # Give feh=0 a small negative number such that the sign operation
    # in the string formatting gives a minus (this is how it works for PHOENIX)
    if feh == 0.0:
        feh = -0.0000001

    filename = f"lte{int(Teff):05d}-{logg:.2f}{feh:+.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    file = folder / filename

    if not file.is_file():
        print(f"File not found. Begin download!")
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

    return wavelength, spectrum, header


def download_phoenix(filename, out_dir, feh):
    """ Download the file from the PHOENIX ftp server.

        :param filename: Name of file to download.
        :param out_dir. pathlib.Path to output directory.
        :param float feh: Metallicity of the star (needed again for the correct
                          ftp folder)
    """
    ftp_root = f"ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{feh:+.1f}/"
    ftp_link = ftp_root.strip() + filename.strip()

    print(f"Download from {ftp_link}")
    wget.download(ftp_link, out=str(out_dir / filename))
    print("Download complete!")
    print(f"Saved to {str(out_dir / filename)}")


def carmenes_template(filename="CARMENES_template.fits"):
    """ Return spec, sig, cont and wave of Carmenes template."""
    template = DATAROOT / filename
    with fits.open(template) as hdul:
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    return (spec, cont, sig, wave)


def harps_template(filename="HARPS_template.fits"):
    """ Return spec, sig, cont and wave of Carmenes template."""
    template = DATAROOT / filename
    with fits.open(template) as hdul:
        hdu = hdul[1]
        wave = np.array(hdu.data["WAVE"][0])
        spec = np.array(hdu.data["FLUX"][0])
        sig = np.array(hdu.data["ERR"][0])

    return (spec, sig, wave)


if __name__ == "__main__":
    (spec, sig, wave) = harps_template()

    plt.plot(wave, sig)
    plt.show()
