from astropy.io import fits
from pathlib import Path
from shutil import copy2
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

    def save_spectrum(self, spectrum, new_header, name):
        """ Save a Carmenes spectrum from spectrum."""
        template = self.dataroot / "template.fits"
        if not name.endswith("fits"):
            name += ".fits"

        # Create the simulation folder
        folder = self._create_folder()

        outfile = folder / name

        print(f"Copy template to {outfile}")
        copy2(template, outfile)

        with fits.open(outfile, mode="update") as hdul:
            # First fix the index error
            for i in range(0, len(hdul)):
                hdul[i].verify("fix")

            # Now update the primary header
            for key, value in new_header.items():
                if key in hdul[0].header.keys():
                    hdul[0].header[key] = value

            hdul[1].data = spectrum
            hdul.flush()

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

    def _create_folder(self):
        """ Create the folder to contain all data."""
        folder = self.dataroot / "fake_spectra" / self.simulation_name
        if not folder.is_dir():
            folder.mkdir()
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
