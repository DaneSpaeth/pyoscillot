from datetime import timedelta
import numpy as np
from barycorrpy import utc_tdb
from astropy.time import Time
from concurrent.futures import ProcessPoolExecutor
from plapy.obs import observatories
from plapy.constants import C
from dataloader import phoenix_spectrum, phoenix_spec_intensity
from datasaver import DataSaver
from star import GridSpectrumSimulator
from pathlib import Path
from parse_ini import parse_ticket, parse_global_ini
from time_sampling import sample_phase, load_presampled_phase
import carmenes_simulator as carmenes
import harps_simulator as harps
from theoretical_rvs import calc_theoretical_results


class SimulationController():
    """ Control the actual Simulation Handling, i.e. creation of
        GridSpectrumSimulators etc.
    """

    def __init__(self, ticketpath):
        """ Initialize the Controller."""
        self.determine_simulation_params(ticketpath)
        self.saver = DataSaver(self.conf["name"])

        self.create_rv_series()

    def determine_simulation_params(self, ticketpath):
        """ Read in the simulation params from the ini file."""
        config_dict = parse_ticket(ticketpath)
        simulation_keys = config_dict["simulations"]

        # if len(simulation_keys) != 1:
        #    raise NotImplementedError("Currently only one mode is implemented")

        self.conf = config_dict


        self.simulation_keys = simulation_keys
        self.instrument = config_dict["instrument"]

    def create_rv_series(self):
        """ Create a fake RV series."""
        # TODO later make that as a loop
        # TODO: decide where to coadd different effects and loop over different
        # mechanisms
        mode = self.conf[self.simulation_keys[0]]["mode"]
        if mode == "planet":
            self.simulate_planet()
        elif mode == "spot":
            self.simulate_spot()
        elif mode == "pulsation":
            self.simulate_pulsation()
        elif mode == "granulation":
            self.simulate_granulation()
        else:
            print("Select a Mode")
            exit()

    def _save_to_disk(self, shift_wavelength, spectrum, time, bc, bjd, v_theo):
        """ Helper function to save the spectrum to disk."""
        # Interpolate onto the CARMENES template
        if self.instrument in ["CARMENES_VIS", "CARMENES", "ALL"]:
            # Determine the template and SNR file from the star name
            # NOTE: AT THE MOMENT ONLY THE NAME IS CHECKED AND NOT THE
            # TEMPERATURE OR SO
            try:
                hip = int(self.conf["hip"])
                star = f"HIP{hip}"
            except ValueError:
                star = self.conf["hip"]

            global_dict = parse_global_ini()
            template_directory = Path(
                global_dict["datapath"]) / "CARMENES_VIS_templates"
            fits_template = template_directory / \
                f"CARMENES_template_{star}.fits"
            if not fits_template.is_file():
                fits_template = None



            # TODO REMOVE
            # fits_template = global_dict["datapath"] / "CARMENES_template.fits"

            global_dict = parse_global_ini()
            snr_directory = Path(
                global_dict["datapath"]) / "CARMENES_VIS_SNR_profiles"
            snr_file = snr_directory / f"{star}.npy"
            try:
                snr_profile = np.load(snr_file)
            except FileNotFoundError:
                snr_profile = None

            # TODO REMOVE
            snr_profile = None

            order_levels = self.conf.get("order_levels", "star")

            shifted_spec, wave = carmenes.interpolate(
                spectrum, shift_wavelength,
                template_file=fits_template,
                snr_profile=snr_profile,
                target_max_snr=float(self.conf["snr"]),
                adjust_snr=False,
                channel="VIS",
                order_levels=order_levels)

            new_header = carmenes.get_new_header(time, bc, bjd,
                                                 snr_profile=snr_profile,
                                                 target_max_snr=float(self.conf["snr"]))
            timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
            filename = f"car-{timestr}-sci-fake-vis_A.fits"

            self.saver.save_spectrum(shifted_spec,
                                     new_header,
                                     filename,
                                     CARMENES_template=fits_template,
                                     instrument="CARMENES_VIS")
        if self.instrument in ["CARMENES_NIR", "CARMENES", "ALL"]:
            # Determine the template and SNR file from the star name
            # NOTE: AT THE MOMENT ONLY THE NAME IS CHECKED AND NOT THE
            # TEMPERATURE OR SO
            try:
                hip = int(self.conf["hip"])
                star = f"HIP{hip}"
            except ValueError:
                star = self.conf["hip"]

            global_dict = parse_global_ini()
            template_directory = Path(
                global_dict["datapath"]) / "CARMENES_NIR_templates"
            fits_template = template_directory / \
                f"CARMENES_template_{star}.fits"
            if not fits_template.is_file():
                fits_template = Path(global_dict["datapath"] / "CARMENES_NIR_template.fits")



            # TODO REMOVE
            # fits_template = global_dict["datapath"] / "CARMENES_template.fits"

            # global_dict = parse_global_ini()
            # snr_directory = Path(
            #     global_dict["datapath"]) / "CARMENES_NIR_SNR_profiles"
            # snr_file = snr_directory / f"{star}.npy"
            # try:
            #     snr_profile = np.load(snr_file)
            # except FileNotFoundError:
            #     snr_profile = None

            # TODO REMOVE
            snr_profile = None

            order_levels = self.conf.get("order_levels", "star")

            shifted_spec, wave = carmenes.interpolate(
                spectrum, shift_wavelength,
                template_file=fits_template,
                snr_profile=snr_profile,
                target_max_snr=float(self.conf["snr"]),
                adjust_snr=False,
                channel="NIR",
                order_levels=order_levels)

            new_header = carmenes.get_new_header(time, bc, bjd,
                                                 snr_profile=snr_profile,
                                                 target_max_snr=float(self.conf["snr"]))
            timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
            filename = f"car-{timestr}-sci-fake-nir_A.fits"

            self.saver.save_spectrum(shifted_spec,
                                     new_header,
                                     filename,
                                     CARMENES_template=fits_template,
                                     instrument="CARMENES_NIR")
        if self.instrument in ["HARPS", "ALL"]:
            shifted_spec, wave = harps.interpolate(spectrum, shift_wavelength)
            new_header, new_comments = harps.get_new_header(time, bc, bjd)

            timestr = time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            filename = f"ADP.{timestr}.fits"
            self.saver.save_spectrum(shifted_spec,
                                     new_header,
                                     filename,
                                     instrument="HARPS",
                                     fits_comment_dict=new_comments)
        if self.instrument in ["RAW", "ALL"]:
            self.saver.save_raw(shift_wavelength, spectrum, bjd, v_theo)

    def simulate_planet(self):
        """ Return a list of wavelengths and fluxes for a planetary signal."""
        sim = self.simulation_keys[0]
        P = self.conf[sim]["period"]
        N = int(self.conf["n"])
        K = self.conf[sim]["k"]

        if not self.conf["timesampling"] == "auto":
            # TODO allow to sample also with presampling here
            raise NotImplementedError
        phase_sample, time_sample = sample_phase(P, N)

        K_sample = K * np.sin(2 * np.pi * phase_sample)

        bjds = self.get_bjd(time_sample, int(self.conf["hip"]))
        bcs = np.zeros(len(bjds))

        # Load one rest_spectrum, all units in Angstrom
        wavelength_range = (self.conf["min_wave"] - 10,
                            self.conf["max_wave"] + 10)
        if self.conf["mode"] == "spectrum":
            rest_wavelength, spectrum, _ = phoenix_spectrum(
                Teff=int(self.conf["teff"]), wavelength_range=wavelength_range)
        else:
            # Actually just for testing in low res mode
            rest_wavelength, spectra, mu, _, = phoenix_spec_intensity(
                Teff=int(self.conf["teff"]), wavelength_range=wavelength_range)
            spectrum = spectra[-1]

        # Add the Doppler shifts
        shift_wavelengths = []
        spectra = []
        for v, time, bc, bjd in zip(K_sample, time_sample, bcs, bjds):
            vo = v
            # ve = 0
            a = (1.0 + vo / C)  # / (1.0 + ve / C)
            # shift_wavelength = np.exp(np.log(rest_wavelength) + np.log(a))
            shift_wavelength = rest_wavelength * v / C + rest_wavelength

            shift_wavelengths.append(shift_wavelength)
            spectra.append(spectrum)

            self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd, v)

        # calc_theoretical_results(shift_wavelengths, spectra, bjds)

    def simulate_spot(self):
        """ Simulate the spot spectra."""
        # TODO allow multiple spots
        phase_samples = []
        time_samples = []
        # For the moment assume that you have only spots
        N_spots = len(self.simulation_keys)
        N_processes = int(self.conf["n_processes"])
        # start out with the first spot sim
        if self.conf["timesampling"] == "auto":
            N = int(self.conf["n"])
            sim = self.simulation_keys[0]
            P = self.conf[sim]["period"]
            N_periods = int(self.conf["n_periods"])
            phase_sample, time_sample = sample_phase(P, N, N_periods=N_periods)
        else:
            # In this mode the infos N, N_periods and P are not used
            filename = self.conf["timesampling"]
            phase_sample, time_sample = load_presampled_phase(filename)
            print(time_sample)
            # Overwrite N
            N = len(phase_sample)
            self.conf["n"] = N

        # Now you have a global and raw phase and time_sample
        # The time_sample is the same for all spots
        # The phase sample is adjusted wrt to the starting phi angle
        # This allows us later to adjust e.g. the rotation speed or so
        phase_samples = np.zeros((N, N_spots))
        for idx, sim in enumerate(self.simulation_keys):
            # Add the phi_angle but divided by the full circle to the phase and fold it back to [0,1]
            phase_samples[:, idx] = np.mod((phase_sample + self.conf[sim]["phi_start"]/360), 1)

            # Also save the theta angle
            # theta_samples[:, idx] = np.ones(len(time_sample)) * self.conf[sim]["theta"]

        # At the moment assume that there is no planetary signal present
        # But still create K_sample for barycentric correction
        K_sample = np.zeros(len(time_sample))
        bjds = self.get_bjd(time_sample, self.conf["hip"])
        bcs = np.zeros(len(bjds))


        idx_list = list(range(len(K_sample)))
        if N_processes > 1:
            with ProcessPoolExecutor(max_workers=N_processes) as executor:
                for r in executor.map(
                        self._run_spot_sim, idx_list, K_sample, phase_samples,
                        time_sample, bjds, bcs):
                    print(r)
        else:
            for r in map(self._run_spot_sim, idx_list, K_sample, phase_samples,
                        time_sample, bjds, bcs):
                print(r)

    def _run_spot_sim(self, idx, v, phases, time, bjd, bc):
        """ Isolated function to actually run the spot simulation.

            It is splitted from the simulate_pulsation function to allow
            multiprocessing which works via pickling.

            :param idx: Idx of current simulation (just for counting)
            :param v: Current velocity due to barycentric correction in m/s
                      (Could in principle be used to add another v shift)
            :param time: Time as datetime.datetime
            :param bjd: Barycentric Julian Date as float
            :param bc: Barycentric correction in m/s

            All parameters should be single values
        """
        N = int(self.conf["n"])
        N_star = int(self.conf["n_star"])
        limb_darkening = bool(int(self.conf["limb_darkening"]))
        inclination = self.conf["inclination"]
        v_rot = self.conf["v_rot"]

        print(f"Calculate star {idx}/{N-1} at bjd {bjd}")
        star = GridSpectrumSimulator(
            N_star=N_star,
            Teff=self.conf["teff"],
            logg=self.conf["logg"],
            feh=self.conf["feh"],
            v_rot=v_rot,
            inclination=inclination,
            limb_darkening=limb_darkening)

        # Assume that you only have spots
        for sim, phase in zip(self.simulation_keys, phases):
            star.add_spot(phase=phase,
                          radius=self.conf[sim]["radius"],
                          T_spot=self.conf[sim]["t_spot"],
                          theta_pos=self.conf[sim]["theta"])

        # Wavelength in restframe of phoenix spectra but already perturbed
        # by spot
        rest_wavelength, spectrum, v_theo = star.calc_spectrum(
            self.conf["min_wave"] - 10, self.conf["max_wave"] + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelength = rest_wavelength + v / C * rest_wavelength

        self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd, v_theo)
        # Save the arrays
        array_dict = star.get_arrays()
        self.saver.save_arrays(array_dict, bjd)
        # Save the flux
        self.saver.save_flux(bjd, star.flux)

        return(f"Star {idx+1}/{N} finished")

    def _get_pulsation_hyperparams(self):
        """ Convenience method to get the hyperparameters and make the code DRY"""
        N = int(self.conf["n"])
        N_local_min = int(self.conf["n_local_min"])
        N_local_max = int(self.conf["n_local_max"])
        rand_day_min = int(self.conf["random_day_local_range_min"])
        rand_day_max = int(self.conf["random_day_local_range_max"])

        N_periods = int(self.conf["n_periods"])
        N_processes = int(self.conf["n_processes"])

        # Determine the time sample
        # At the moment take the first mode as sampling period
        P = self.conf[self.simulation_keys[0]]["period"]
        if self.conf["timesampling"] == "auto":
            _, time_sample = sample_phase(
                P, N_periods=N_periods, N_global=N,
                N_local=(N_local_min, N_local_max),
                random_day_range=(rand_day_min, rand_day_max))
        else:
            filename = self.conf["timesampling"]
            phase_sample, time_sample = load_presampled_phase(filename)
            print(time_sample)
            # Overwrite N
            N = len(phase_sample)
            self.conf["n"] = N

        K_sample = np.zeros(len(time_sample))

        bjds = self.get_bjd(time_sample, int(self.conf["hip"]))
        bcs = np.zeros(len(bjds))

        return N, N_local_min, N_local_max, rand_day_min, rand_day_max, N_periods, N_processes, P, K_sample, time_sample, bjds, bcs

    def simulate_pulsation(self):
        """ Simulate the pulsation spectra."""
        (N, N_local_min, N_local_max, rand_day_min, rand_day_max,
         N_periods, N_processes, P, K_sample, time_sample, bjds, bcs) = self._get_pulsation_hyperparams()

        idx_list = list(range(len(K_sample)))
        if N_processes > 1:
            with ProcessPoolExecutor(max_workers=N_processes) as executor:
                for r in executor.map(
                        self._run_pulsation_sim, idx_list,
                        K_sample, time_sample, bjds, bcs):
                    print(r)
        else:
            for r in map(self._run_pulsation_sim, idx_list,
                         K_sample, time_sample, bjds, bcs):
                print(r)

    def _run_pulsation_sim(self, idx, v, time, bjd, bc):
        """ Isolated function to actually run the pulsation simulation.

            It is splitted from the simulate_pulsation function to allow
            mulitprocessing which works via pickling.

            :param idx: Idx of current simulation (just for counting)
            :param v: Current velocity due to barycentric correction in m/s
                      (Could in principle be used to add another v shift)
            :param time: Time as datetime.datetime
            :param bjd: Barycentric Julian Date as float
            :param bc: Barycentric correction in m/s

            All parameters should be single values
        """
        N = int(self.conf["n"])
        limb_darkening = bool(int(self.conf["limb_darkening"]))
        v_rot = self.conf["v_rot"]
        inclination = self.conf["inclination"]
        N_star = int(self.conf["n_star"])

        print(f"Calculate star {idx+1}/{N} at bjd {bjd}")
        star = GridSpectrumSimulator(
            N_star=N_star, N_border=3,
            Teff=int(self.conf["teff"]),
            logg=float(self.conf["logg"]),
            feh=float(self.conf["feh"]),
            v_rot=v_rot, inclination=inclination,
            limb_darkening=limb_darkening)

        # Add all specified pulsations
        for sim in self.simulation_keys:
            P = self.conf[sim]["period"]
            l = int(self.conf[sim]["l"])
            k = int(self.conf[sim]["k"])
            v_p = self.conf[sim]["v_p"]
            dT = self.conf[sim]["dt"]
            T_phase = self.conf[sim]["t_phase"]

            if "m" in list(self.conf[sim].keys()):
                ms = [int(self.conf[sim]["m"])]
            else:
                ms = range(-l, l + 1)
            for m in ms:
                print(
                    f"Add Pulsation {sim}, with P={P}, l={l}, m={m}, v_p={v_p}, k={k}, dT={dT}, T_phase={T_phase}")

                star.add_pulsation(t=bjd, l=l, m=m, nu=1 / P, v_p=v_p, k=k,
                                   T_var=dT, T_phase=T_phase)

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # pulsation
        if self.conf["mode"] == "spectrum":
            print(f"Run in FULL SPECTRUM MODE")
            mode = "phoenix"
        elif self.conf["mode"] == "spec_intensity":
            print(f"Run in SPECIFIC INTENSITY MODE")
            mode = "spec_intensity"
        rest_wavelength, spectrum, v_theo = star.calc_spectrum(
            self.conf["min_wave"] - 10,
            self.conf["max_wave"] + 10,
            mode=mode)

        # if not Path(f"{name}/spectrum_0.0_0.npy").is_file():
        #     # TODO REFACTOR
        #     ref_star = GridSpectrumSimulator(
        #         N_star=N_star, N_border=3,
        #         Teff=int(self.conf["teff"]),
        #         logg=float(self.conf["logg"]),
        #         feh=float(self.conf["feh"]),
        #         v_rot=v_rot, inclination=inclination,
        #         limb_darkening=limb_darkening)
        #     ref_wave, ref_spec, v = ref_star.calc_spectrum(self.conf["min_wave"] - 10,
        #                                                    self.conf["max_wave"] + 10)
        #     np.save(f"{name}/wave_{v}_{0}.npy", ref_wave)
        #     np.save(f"{name}/spectrum_{v}_{0}.npy", ref_spec)

        # Add doppler shift due to barycentric correction
        # shift_wavelength = rest_wavelength + v / C * rest_wavelength

        # Save the arrays
        array_dict = star.get_arrays()
        self.saver.save_arrays(array_dict, bjd)
        array_dict = None
        del array_dict
        # Save the flux
        self.saver.save_flux(bjd, star.flux)

        self._save_to_disk(rest_wavelength, spectrum, time, bc, bjd, v_theo)

        return(f"Star {idx+1}/{N} finished")

    def simulate_granulation(self):
        """ Simulate the granulation spectra."""
        # Get the global parameters
        (N, N_local_min, N_local_max, rand_day_min, rand_day_max,
         N_periods, N_processes, P, K_sample, time_sample, bjds, bcs) = self._get_pulsation_hyperparams()

        idx_list = list(range(len(K_sample)))
        if N_processes > 1:
            with ProcessPoolExecutor(max_workers=N_processes) as executor:
                for r in executor.map(
                        self._run_granulation_sim, idx_list,
                        K_sample, time_sample, bjds, bcs):
                    print(r)
        else:
            for r in map(self._run_pulsation_sim, idx_list,
                         K_sample, time_sample, bjds, bcs):
                print(r)

    def _run_granulation_sim(self, idx, v, time, bjd, bc):
        """ Isolated function to actually run the granulation simulation.

            It is splitted from the simulate_granulation function to allow
            mulitprocessing which works via pickling.

            :param idx: Idx of current simulation (just for counting)
            :param v: Current velocity due to barycentric correction in m/s
                      (Could in principle be used to add another v shift)
            :param time: Time as datetime.datetime
            :param bjd: Barycentric Julian Date as float
            :param bc: Barycentric correction in m/s

            All parameters should be single values
        """
        # TODO refactor the parameters of the funcion (I don't need any of these)
        N = int(self.conf["n"])
        limb_darkening = bool(int(self.conf["limb_darkening"]))
        v_rot = self.conf["v_rot"]
        inclination = self.conf["inclination"]
        N_star = int(self.conf["n_star"])

        sim = "granulation"
        dT = self.conf[sim]["dt"]
        dv = self.conf[sim]["dv"]
        granule_size = self.conf[sim]["granule_size"]

        print(f"Calculate star {idx+1}/{N} at bjd {bjd}")
        star = GridSpectrumSimulator(
            N_star=N_star, N_border=3,
            Teff=int(self.conf["teff"]),
            logg=float(self.conf["logg"]),
            feh=float(self.conf["feh"]),
            v_rot=v_rot, inclination=inclination,
            limb_darkening=limb_darkening)


        # Todo Implement the automatic number of granules
        star.add_granulation(dT=dT, dv=dv, granule_size=granule_size)

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # pulsation
        if self.conf["mode"] == "spectrum":
            print(f"Run in FULL SPECTRUM MODE")
            mode = "phoenix"
        elif self.conf["mode"] == "spec_intensity":
            print(f"Run in SPECIFIC INTENSITY MODE")
            mode = "spec_intensity"
        rest_wavelength, spectrum, v_theo = star.calc_spectrum(
            self.conf["min_wave"] - 10,
            self.conf["max_wave"] + 10,
            mode=mode)

        # Save the flux
        self.saver.save_flux(bjd, star.flux)

        # Save the arrays
        array_dict = star.get_arrays()
        self.saver.save_arrays(array_dict, bjd)
        array_dict = None
        del array_dict

        self._save_to_disk(rest_wavelength, spectrum, time, bc, bjd, v_theo)

        return(f"Star {idx+1}/{N} finished")

    def get_bjd(self, time_list, star, t_exp=106.091):
        """ Get the BJD for times in time_list as UTC.

            At the moment I decide to set the barycentric correction to 0.
            In serval we later use the BERV modes to not correct for
            the barycentric correction.

            :param list time_list: List of datetime.datetime as utc
            :param int star: Name of star to simulate
            :param float t_exp: Exposure time in seconds
        """
        time_list = [t + timedelta(seconds=t_exp / 2) for t in time_list]
        jdutc_times = [Time(t, scale="utc") for t in time_list]

        for jdutc in jdutc_times:
            jdutc.format = "jd"

        # Define the Calar Alto Observatory
        caha = observatories.calar_alto
        lat = float(caha["lat"].replace(" N", ""))
        lon = -float((caha["lon"].replace(" W", "")))
        alt = 2168.

        bjds = []
        for jdutc in jdutc_times:
            try:
                star = f"HIP{int(star)}"
            except ValueError:
                star = star
            bjd_result = utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdutc, starname=star,
                                                 lat=lat, longi=lon,
                                                 alt=alt, ephemeris='de430')
            bjds.append(float(bjd_result[0]))

        return bjds




if __name__ == "__main__":
    ticket = "tickets/uniform_test.ini"
    SimulationController(ticket)
