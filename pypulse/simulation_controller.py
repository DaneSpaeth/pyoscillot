from datetime import datetime, date, timedelta
import numpy as np
from barycorrpy import get_BC_vel, utc_tdb
import random
from astropy.time import Time
from concurrent.futures import ProcessPoolExecutor
from plapy.obs import observatories
from plapy.constants import C
from dataloader import phoenix_spectrum
from datasaver import DataSaver
from star import GridSpectrumSimulator
from parse_ini import parse_ticket
import carmenes_simulator as carmenes
import harps_simulator as harps


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
        else:
            print("Select a Mode")
            exit()

    def _save_to_disk(self, shift_wavelength, spectrum, time, bc, bjd):
        """ Helper function to save the spectrum to disk."""
        # Interpolate onto the CARMENES template
        if self.instrument in ["CARMENES_VIS", "ALL"]:
            shifted_spec, wave = carmenes.interpolate(
                spectrum, shift_wavelength)
            new_header = carmenes.get_new_header(time, bc, bjd)
            timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
            filename = f"car-{timestr}-sci-fake-vis_A.fits"

            self.saver.save_spectrum(shifted_spec,
                                     new_header,
                                     filename,
                                     instrument="CARMENES_VIS")
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

    def sample_phase(self, sample_P, N_global=30, N_periods=1,
                     N_local=(1, 1), random_day_range=(0, 1)):
        """ New a bit more random phase sampling. Really ugly at the moment

            :param float sample_P: Global Period to sample
            :param int N_phase: Number of phases to sample
            :param int N_global: Global Number of local datapoints to draw
                                 to sample
            :param tuple of ints N_local: Min and max number of datapoints to
                                          draw for one global datapoint
            :param tuple of ints random_day_range: Min and max deviation from
                                                   global day to allow for
                                                   local days

            The current default params allow a uniform sampling with 1
            local day per global day without deviation.

            It therefore replaces the previous function.
        """
        stop = datetime.combine(date.today(), datetime.min.time())

        start = stop - timedelta(days=int(sample_P * N_periods))

        time_sample = []

        global_days = np.linspace(0, int(sample_P * N_periods), N_global)
        local_days = []
        for gday in global_days:
            for i in range(random.randint(N_local[0], N_local[1])):
                # Simulate multiple observations shortly after each other
                day_random = random.randrange(random_day_range[0],
                                              random_day_range[1])
                local_day = start + \
                    timedelta(days=int(gday)) + timedelta(days=int(day_random))
                time_sample.append(local_day)

        time_sample = sorted(time_sample)
        time_sample = np.array(time_sample)
        phase_sample = (np.mod((time_sample - start) /
                               timedelta(days=1), sample_P)) / sample_P
        return phase_sample.astype(float), time_sample

    def simulate_planet(self):
        """ Return a list of wavelengths and fluxes for a planetary signal."""
        sim = self.simulation_keys[0]
        P = self.conf[sim]["period"]
        N = int(self.conf["n"])
        K = self.conf[sim]["k"]

        phase_sample, time_sample = self.sample_phase(P, N)

        K_sample = K * np.sin(2 * np.pi * phase_sample)

        K_sample, bcs, bjds = self.add_barycentric_correction(
            K_sample, time_sample, int(self.conf["hip"]))

        # Load one rest_spectrum, all units in Angstrom
        wavelength_range = (self.conf["min_wave"] - 10,
                            self.conf["max_wave"] + 10)
        rest_wavelength, spectrum, _ = phoenix_spectrum(
            Teff=int(self.conf["teff"]), wavelength_range=wavelength_range)

        # Add the Doppler shifts
        for v, time, bc, bjd in zip(K_sample, time_sample, bcs, bjds):
            print(v)
            vo = v
            ve = 0
            a = (1.0 + vo / C) / (1.0 + ve / C)
            shift_wavelength = np.exp(np.log(rest_wavelength) + np.log(a))

            self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd)

    def simulate_spot(self):
        """ Simulate the spot spectra."""
        # TODO allow multiple spots
        sim = self.simulation_keys[0]
        N = int(self.conf["n"])
        P = self.conf[sim]["period"]
        N_processes = int(self.conf["n_processes"])

        phase_sample, time_sample = self.sample_phase(P, N)

        # At the moment assume that there is no planetary signal present
        # But still create K_sample for barycentric correction
        K_sample = np.zeros(len(time_sample))
        K_sample, bcs, bjds = self.add_barycentric_correction(
            K_sample, time_sample, int(self.conf["hip"]))

        idx_list = list(range(len(K_sample)))
        if N_processes > 1:
            with ProcessPoolExecutor(max_workers=N_processes) as executor:
                for r in executor.map(
                        self._run_spot_sim, idx_list, K_sample, phase_sample,
                        time_sample, bjds, bcs):
                    print(r)
        else:
            for r in map(self._run_spot_sim, idx_list,
                         K_sample, time_sample, bjds, bcs):
                print(r)

    def _run_spot_sim(self, idx, v, phase, time, bjd, bc):
        """ Isolated function to actually run the spot simulation.

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
        sim = self.simulation_keys[0]
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
        star.add_spot(phase=phase,
                      radius=self.conf[sim]["radius"],
                      T_spot=self.conf[sim]["t_spot"])

        # Wavelength in restframe of phoenix spectra but already perturbed
        # by spot
        rest_wavelength, spectrum = star.calc_spectrum(
            self.conf["min_wave"] - 10, self.conf["max_wave"] + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelength = rest_wavelength + v / C * rest_wavelength

        self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd)
        if self.instrument in ["CARMENES_VIS", "ALL"]:
            self.saver.save_flux(bjd, star.flux, "CARMENES_VIS")
        if self.instrument in ["HARPS", "ALL"]:
            self.saver.save_flux(bjd, star.flux, "HARPS")

        return(f"Star {idx}/{N-1} finished")

    def simulate_pulsation(self):
        """ Simulate the pulsation spectra."""
        # Get the global parameters
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
        _, time_sample = self.sample_phase(
            P, N_periods=N_periods, N_global=N,
            N_local=(N_local_min, N_local_max),
            random_day_range=(rand_day_min, rand_day_max))

        K_sample = np.zeros(len(time_sample))

        K_sample, bcs, bjds = self.add_barycentric_correction(
            K_sample, time_sample, int(self.conf["hip"]))

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

        print(f"Calculate star {idx}/{N-1} at bjd {bjd}")
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
            # m = int(self.conf[sim]["m"])
            k = int(self.conf[sim]["k"])
            v_p = self.conf[sim]["v_p"]
            dT = self.conf[sim]["dt"]
            T_phase = self.conf[sim]["t_phase"]

            for m in range(-l, l + 1):
                print(
                    f"Add Pulsation {sim}, with P={P}, l={l}, m={m}, v_p={v_p}, k={k}, dT={dT}, T_phase={T_phase}")

                star.add_pulsation(t=bjd, l=l, m=m, nu=1 / P, v_p=v_p, k=k,
                                   T_var=dT, T_phase=T_phase)

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # pulsation
        rest_wavelength, spectrum = star.calc_spectrum(
            self.conf["min_wave"] - 10,
            self.conf["max_wave"] + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelength = rest_wavelength + v / C * rest_wavelength

        array_dict = star.get_arrays()

        if self.instrument in ["CARMENES_VIS", "ALL"]:
            self.saver.save_arrays(array_dict, bjd, "CARMENES_VIS")
            self.saver.save_flux(bjd, star.flux, "CARMENES_VIS")
        if self.instrument in ["HARPS", "ALL"]:
            self.saver.save_arrays(array_dict, bjd, "HARPS")
            self.saver.save_flux(bjd, star.flux, "HARPS")

        self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd)

        return(f"Star {idx}/{N-1} finished")

    def add_barycentric_correction(self, K_array, time_list, star, set_0=True):
        """ At the moment the K_array is not affected.

            TODO: Decide for a consistent way of handling the K array
            We could actually minimize this function since we set the BC to 0
            at the moment.
        """

        # TODO: Fix the exposure time in some way (I don't know at the
        # moment if it matters)
        tmean = 53.0455
        time_list = [t + timedelta(seconds=tmean) for t in time_list]
        jdutc_times = [Time(t, scale="utc") for t in time_list]

        for jdutc in jdutc_times:
            jdutc.format = "jd"

        # Define the Calar Alto Observatory
        caha = observatories.calar_alto
        lat = float(caha["lat"].replace(" N", ""))
        lon = -float((caha["lon"].replace(" W", "")))
        alt = 2168.

        bcs = []
        bjds = []
        for jdutc in jdutc_times:

            result = get_BC_vel(JDUTC=jdutc, hip_id=star,
                                lat=lat, longi=lon,
                                alt=alt, ephemeris='de430')
            # result = get_BC_vel(JDUTC=jdutc,
            #                     ra=225.72515818125,
            #                     dec=2.0913040080555554,
            #                     epoch=2451545.0,
            #                     pmra=-54.89,
            #                     pmdec=13.34,
            #                     px=0.0,
            #                     lat=37.2236,
            #                     longi=-2.5463,
            #                     alt=2168.0)
            # bjd_result = utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdutc,
            #                                      ra=225.72515818125,
            #                                      dec=2.0913040080555554,
            #                                      epoch=2451545.0,
            #                                      pmra=-54.89,
            #                                      pmdec=13.34,
            #                                      px=0.0,
            #                                      lat=37.2236,
            #                                      longi=-2.5463,
            #                                      alt=2168.0)
            bjd_result = utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdutc, hip_id=star,
                                                 lat=lat, longi=lon,
                                                 alt=alt, ephemeris='de430')
            bcs.append(float(result[0]))
            bjds.append(float(bjd_result[0]))

        bcs = np.array(bcs)
        if set_0:
            bcs *= 0
        return K_array - bcs, bcs, bjds


if __name__ == "__main__":
    ticket = "hip73620_ticket.ini"
    SimulationController(ticket)
