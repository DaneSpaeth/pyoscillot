from astropy.io import fits
from pathlib import Path
from shutil import copy2
import tarfile
import numpy as np
import cfg
import socket
laptop = socket.gethostname() == "dane-ThinkPad-E460"


class DataSaver():
    """ Handle all saving and restructring of data."""

    def __init__(self, simulation_name):
        """ Construct instance.

            :param str simulation_name: Name of simulation to use for folders
        """
        self.global_dict = cfg.parse_global_ini()
        if not laptop:
            self.dataroot = self.global_dict["datapath"]
            self.outroot = self.global_dict["outpath"]
        else:
            self.dataroot = self.global_dict["datapath_laptop"]
        self.simulation_name = simulation_name
        self.debug_dir = self._create_folder("plots")
        
        # Change the singleton in the parse_ini module
        cfg.debug_dir = self.debug_dir

    def save_spectrum(self, spectrum, new_header, name,
                      CARMENES_template=None,
                      instrument="CARMENES_VIS",
                      fits_comment_dict=None):
        """ Save a Carmenes spectrum from spectrum."""
        if instrument == "CARMENES_VIS":
            if CARMENES_template is None:
                template = self.dataroot / "CARMENES_VIS_template.fits"
            else:
                template = CARMENES_template
            if not name.endswith("fits"):
                name += ".fits"
        elif instrument == "CARMENES_NIR":
            if CARMENES_template is None:
                template = self.dataroot / "CARMENES_NIR_template.fits"
            else:
                template = CARMENES_template
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

        print(f"Copy template {template} to {outfile}")
        copy2(template, outfile)

        if instrument == "CARMENES_VIS" or instrument == "CARMENES_NIR":
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

    def save_raw(self, wave, spec, bjd, v_theo):
        """ Save the Raw wavelength and spectrum as npy files."""
        folder = self._create_folder("RAW")
        np.save(folder / f"wave_{bjd}_{v_theo}.npy", wave)
        print(f"Save RAW wavelength to {folder / f'wave_{bjd}_{v_theo}.npy'}")
        np.save(folder / f"spec_{bjd}_{v_theo}.npy", spec)
        print(f"Save RAW spectrum to {folder / f'spec_{bjd}_{v_theo}.npy'}")

    def save_arrays(self, array_dict, bjd):
        """ Save the 2D arrays of the simulation along with the spectra.

            :param dict array_dict: Dictionary of savename:array
            :param dict bjd: BJD to save
        """
        # Make sure the folder is created, also creates the array folder
        folder = self._create_folder()
        array_folder = folder / "arrays"

        for key, array in array_dict.items():
            component_folder = array_folder / key
            try:
                if not component_folder.is_dir():
                    component_folder.mkdir(parents=True)
            except FileExistsError:
                pass

            array_path = component_folder / f"{bjd}.npy"
            print(f"Save {key} array to {array_path}")
            np.save(array_path, array, fix_imports=False)

    def save_flux(self, bjd, flux):
        """ Save flux to file."""
        folder = self._create_folder()

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

    def _create_folder(self, instrument=None):
        """ Create the folder to contain all data."""
        folder = self.outroot / self.simulation_name
        if instrument is not None:
            folder = folder / instrument
        try:
            if not folder.is_dir():
                folder.mkdir(parents=True)
            if instrument is None:
                if not (folder / "arrays").is_dir():
                    array_folder = folder / "arrays"
                    array_folder.mkdir()
        except FileExistsError:
            pass
        return folder

    def copy_ticket_spectrafolder(self, ticket):
        """ Copy the ticket to the spectrum folder.
        """
        ticket = Path(ticket)
        if not ticket.is_file():
            ticket = Path("tickets") / ticket
        if not ticket.is_file():
            raise FileNotFoundError(ticket)
            exit()

        # First copy to the spectra folder
        folder = self._create_folder()
        new_ticket = folder / ticket.name

        copy2(ticket, new_ticket)

        return new_ticket

    def copy_ticket_servalfolder(self, ticket):
        """ Copy the ticket to the spectrum folder."""

        ticket = Path(ticket)
        if not ticket.is_file():
            ticket = Path("tickets") / ticket
        if not ticket.is_file():
            raise FileNotFoundError(ticket)
            exit()

        # Now copy the ticket to the serval folder
        serval_ticket = self.global_dict["rvlibpath"] / "serval" / "SIMULATION" /\
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
