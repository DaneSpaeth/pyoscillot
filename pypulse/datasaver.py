from astropy.io import fits
from pathlib import Path
from shutil import copy2
import tarfile
import numpy as np
from parse_ini import parse_global_ini


class DataSaver():
    """ Handle all saving and restructring of data."""

    def __init__(self, simulation_name):
        """ Construct instance.

            :param str simulation_name: Name of simulation to use for folders
        """
        self.global_dict = parse_global_ini()
        self.dataroot = self.global_dict["datapath"]
        self.simulation_name = simulation_name

    def save_spectrum(self, spectrum, new_header, name,
                      instrument="CARMENES_VIS",
                      fits_comment_dict=None):
        """ Save a Carmenes spectrum from spectrum."""
        if instrument == "CARMENES_VIS":
            template = self.dataroot / "CARMENES_template.fits"
            if not name.endswith("fits"):
                name += ".fits"
        elif instrument == "HARPS":
            template = self.dataroot / "HARPS_template_e2ds_A.fits"
            if not name.endswith("_e2ds_A.fits"):
                if name.endswith(".fits"):
                    name = name.replace(".fits", "_e2ds_A.fits")
                else:
                    name += "_e2ds_A.fits"

        # Create the simulation folder
        folder = self._create_folder(instrument)

        outfile = folder / name

        print(f"Copy template to {outfile}")
        copy2(template, outfile)

        if instrument == "CARMENES_VIS":
            with fits.open(outfile, mode="update") as hdul:
                # First fix the index error
                for i in range(0, len(hdul)):
                    hdul[i].verify("fix")

                # Now update the primary header
                for key, value in new_header.items():
                    if key in hdul[0].header.keys():
                        hdul[0].header[key] = value
                # Needed that serval shows the correct filename
                # during the analysis
                hdul[0].header["FILENAME"] = outfile.name

                hdul[1].data = spectrum
                hdul.flush()
        elif instrument == "HARPS":
            with fits.open(outfile, mode="update") as hdul:
                # Now update the primary header
                for key, value in new_header.items():
                    hdul[0].header[key] = value
                if fits_comment_dict is not None:
                    for key, value in fits_comment_dict.items():
                        hdul[0].header.comments[key] = value
                hdul[0].data = spectrum
                hdul.flush()

            with tarfile.open(str(outfile).replace('_e2ds_A.fits', '.tar'), "w") as tar:
                tar.add(
                    outfile, arcname=f"{outfile.name}")
            outfile.unlink()

    def save_arrays(self, array_dict, bjd, instrument):
        """ Save the 2D arrays of the simulation along with the spectra.

            :param dict array_dict: Dictionary of savename:array
            :param dict bjd: BJD to save
        """
        # Make sure the folder is created, also creates the array folder
        folder = self._create_folder(instrument)
        array_folder = folder / "arrays"

        for key, array in array_dict.items():
            component_folder = array_folder / key
            if not component_folder.is_dir():
                component_folder.mkdir(parents=True)

            array_path = component_folder / f"{bjd}.npy"
            print(f"Save {key} array to {array_path}")
            np.save(array_path, array, fix_imports=False)

    def save_flux(self, bjd, flux, instrument):
        """ Save flux to file."""
        folder = self._create_folder(instrument)

        with open(folder / "flux.txt", "a") as f:
            f.write(f"{bjd}    {flux}\n")

    def copy_flux(self):
        """ Copy the flux file to the serval folder."""
        folder = self._create_folder()

        fluxfile = folder / "flux.txt"
        if fluxfile.is_file():
            new_fluxfile = self.global_dict["rvlibpath"] / \
                self.simulation_name / "flux.txt"
            copy2(fluxfile, new_fluxfile)

    def _create_folder(self, instrument):
        """ Create the folder to contain all data."""
        folder = self.dataroot / "fake_spectra" / \
            self.simulation_name / instrument

        if not folder.is_dir():
            folder.mkdir(parents=True)
            array_folder = folder / "arrays"
            array_folder.mkdir()
        return folder

    def copy_ticket(self, ticket):
        """ Copy the ticket both to the spectrum folder and the serval output
            folder.
        """
        ticket = Path(ticket)

        # First copy to the spectra folder
        folder = self._create_folder()
        new_ticket = folder / ticket.name

        copy2(ticket, new_ticket)

        # Now copy the ticket to the serval folder
        serval_ticket = self.global_dict["rvlibpath"] / \
            self.simulation_name / (str(self.simulation_name) + ".ini")

        if serval_ticket.parent.is_dir():
            copy2(ticket, serval_ticket)


def read_in(file):
    with fits.open(file) as hdul:
        header = hdul[0].header
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    return header, spec, cont, sig, wave
