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
        shifted_spec, wave = carmenes.interpolate(spectrum, shift_wavelength)

        new_header = carmenes.get_new_header(time, bc, bjd)
        timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
        filename = f"car-{timestr}-sci-fake-vis_A.fits"

        self.saver.save_spectrum(shifted_spec, new_header, filename)

    def sample_phase(self, P, N, N_phases=1):
        """ Return a phase sample and the corresponding time sample.

            Phase ranges from 0 to 1
        """
        # At the moment, fix today as last observation date
        stop = datetime.combine(date.today(), datetime.min.time())

        # Sample one Period
        P_sample = np.linspace(0, N_phases * P, N, dtype=int)
        time_sample = np.array([stop - timedelta(days=int(d))
                                for d in P_sample[::-1]])
        phase_sample = (1 / P * P_sample) % 1

        return phase_sample, time_sample

    def sample_phase_new(self, sample_P, N_phases=1, N_global=30, N_local=(3, 8)):
        """ New a bit more random phase sampling. Really ugly at the moment

            :param float sample_P: Global Period to sample
            :param int N_phase: Number of phases to sample
            :param int N_global: Global Number of local datapoints to draw
                                 to sample
            :param tuple of ints N_local: Min and max number of datapoints to
                                          draw for one global datapoint
        """
        stop = datetime.combine(date.today(), datetime.min.time())

        start = stop - timedelta(days=int(sample_P * N_phases))

        time_sample = []

        global_days = np.linspace(0, int(sample_P * N_phases), N_global)
        local_days = []
        for gday in global_days:
            for i in range(random.randint(N_local[0], N_local[1])):
                # Simulate multiple observations shortly after each other
                day_random = random.randrange(-10, 10)
                local_day = start + \
                    timedelta(days=int(gday)) + timedelta(days=int(day_random))
                time_sample.append(local_day)

        return sorted(time_sample)

    def simulate_planet(self):
        """ Return a list of wavelengths and fluxes for a planetary signal."""
        P = self.conf[self.sim]["period"]
        N = int(self.conf["n"])
        K = self.conf[self.sim]["k"]

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
        shift_wavelengths = []
        spectra = []
        for v, time, bc, bjd in zip(K_sample, time_sample, bcs, bjds):
            vo = v
            ve = 0
            a = (1.0 + vo / C) / (1.0 + ve / C)
            shift_wavelengths = np.exp(np.log(rest_wavelength) + np.log(a))

            self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd)

    def simulate_spot(self):
        """ Simulate the spot spectra."""
        # TODO allow multiple spots
        sim = self.simulation_keys[0]
        N = int(self.conf["n"])
        N_star = int(self.conf["n_star"])
        P = self.conf[sim]["period"]
        limb_darkening = bool(int(self.conf["limb_darkening"]))
        inclination = self.conf["inclination"]
        v_rot = self.conf["v_rot"]

        phase_sample, time_sample = self.sample_phase(P, N)

        # TODO REMOVE
        phase_sample = phase_sample[:-1]
        time_sample = time_sample[:-1]
        # END TODO

        # At the moment assume that there is no planetary signal present
        # But still create K_sample for barycentric correction
        K_sample = np.zeros(len(time_sample))
        K_sample, bcs, bjds = self.add_barycentric_correction(
            K_sample, time_sample, int(self.conf["hip"]))

        shift_wavelengths = []
        spectra = []
        fluxes = []
        i = 0

        for v, phase, time, bjd, bc in zip(K_sample, phase_sample, time_sample,
                                           bjds, bcs):
            print(f"Calculate star {i}/{len(time_sample)-1} at bjd {bjd}")
            i += 1
            star = GridSpectrumSimulator(
                N_star=N_star,
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
            self.saver.save_flux(bjd, star.flux)

    def simulate_pulsation(self):
        """ Simulate the pulsation spectra."""
        # Get the global parameters
        N = int(self.conf["n"])
        hip = int(self.conf["hip"])
        N_processes = int(self.conf["n_processes"])

        # Determine the time sample
        # TODO fix phase sampling
        # At the moment take the first mode as sampling period
        P = self.conf[self.simulation_keys[0]]["period"]
        time_sample = self.sample_phase_new(
            P, N_phases=1, N_global=N, N_local=(1, 1))

        K_sample = np.zeros(len(time_sample))

        K_sample, bcs, bjds = self.add_barycentric_correction(
            K_sample, time_sample, hip)

        idx_list = list(range(len(K_sample)))
        with ProcessPoolExecutor(max_workers=N_processes) as executor:
            results = list(executor.map(
                self._run_pulsation_sim, idx_list, K_sample, time_sample, bjds, bcs))

    def _run_pulsation_sim(self, idx, v, time, bjd, bc):
        N = int(self.conf["n"])
        limb_darkening = bool(int(self.conf["limb_darkening"]))
        Teff = int(self.conf["teff"])
        v_rot = self.conf["v_rot"]
        inclination = self.conf["inclination"]
        N_star = int(self.conf["n_star"])
        min_wave = self.conf["min_wave"]
        max_wave = self.conf["max_wave"]

        print(f"Calculate star {idx}/{N-1} at bjd {bjd}")
        star = GridSpectrumSimulator(
            N_star=N_star, N_border=3, Teff=Teff,
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
            min_wave - 10, max_wave + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelength = rest_wavelength + v / C * rest_wavelength

        self._save_to_disk(shift_wavelength, spectrum, time, bc, bjd)
        self.saver.save_flux(bjd, star.flux)

    def add_barycentric_correction(self, K_array, time_list, star, set_0=True):
        """ Add the barycentric correction to the K_list."""

        tmean = 53.0455
        time_list = [t + timedelta(seconds=tmean) for t in time_list]
        jdutc_times = [Time(t, scale="utc") for t in time_list]

        for jdutc in jdutc_times:
            jdutc.format = "jd"

        caha = observatories.calar_alto
        lat = float(caha["lat"].replace(" N", ""))
        lon = -float((caha["lon"].replace(" W", "")))
        alt = 2168.

        bcs = []
        bjds = []
        for jdutc in jdutc_times:

            # result = get_BC_vel(JDUTC=jdutc, hip_id=star, lat=lat, longi=lon,
            #                     alt=alt, ephemeris='de430')
            result = get_BC_vel(JDUTC=jdutc,
                                ra=225.72515818125,
                                dec=2.0913040080555554,
                                epoch=2451545.0,
                                pmra=-54.89,
                                pmdec=13.34,
                                px=0.0,
                                lat=37.2236,
                                longi=-2.5463,
                                alt=2168.0)
            bjd_result = utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdutc,
                                                 ra=225.72515818125,
                                                 dec=2.0913040080555554,
                                                 epoch=2451545.0,
                                                 pmra=-54.89,
                                                 pmdec=13.34,
                                                 px=0.0,
                                                 lat=37.2236,
                                                 longi=-2.5463,
                                                 alt=2168.0)
            bcs.append(float(result[0]))
            bjds.append(float(bjd_result[0]))

        bcs = np.array(bcs)
        if set_0:
            bcs *= 0
        return K_array - bcs, bcs, bjds
