from datetime import datetime, timedelta, date
import random
import numpy as np
from parse_ini import parse_global_ini


def sample_phase(sample_P, N_global=30, N_periods=1,
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
    # stop = datetime.combine(date.today(), datetime.min.time())
    stop = datetime(2021, 6, 10, hour=0, minute=0, second=0)

    start = stop - timedelta(days=int(sample_P * N_periods))

    time_sample = []

    global_days = np.linspace(0, int(sample_P * N_periods), N_global)
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


def sample_phase_randomuniform(N_global, N_periods, period):
    """ Sample time and phase for a random timesampling with one observation in random nights with additional
        random sampling within the night
    ."""
    stop = datetime.combine(date.today(), datetime.min.time())
    days_int = int(round(N_periods * period))
    start = stop - timedelta(days=days_int)

    # Randomly choose days in between
    lin_days = np.arange(0, days_int, 1)
    random_days = sorted(random.sample(list(lin_days), k=N_global))
    random_days = [int(d) for d in random_days]

    time_sample = np.array([])
    for randday in random_days:
        # Now randomly draw a timestamp
        # TODO: ideally it would be implemented that you can only draw at the night time
        # But that would at the moment require to load the location and calculate that
        # It's a bit too much for now and probably not really important
        randhour = random.uniform(-6, 6)
        dtime = timedelta(days=randday, hours=randhour)
        observed_datetime = start + dtime
        time_sample = np.append(time_sample, observed_datetime)

    phase_sample = (np.mod((time_sample - start) /
                           timedelta(days=1), period)) / period

    return phase_sample.astype(float), time_sample


def presample_spot_phase(N_global, N_periods, period, savename):
    """ Presample the phases of a spot sim for repeated use"""
    phase_sample, time_sample = sample_phase_randomuniform(
        N_global, N_periods, period)

    global_dict = parse_global_ini()
    out_directory = global_dict["datapath"]
    out_file = out_directory / "timesamples" / f"{savename}.dat"

    if out_file.is_file():
        print(f"{out_file} exists already!")
        print("Overwriting is not allowed!")
        exit()

    print(f"Save to {out_file}")

    with open(out_file, "w") as f:
        for p, t in zip(phase_sample, time_sample):
            line = f"{p:.15f}    {t.isoformat()}\n"
            f.write(line)


def load_presampled_spot_phase(savename):
    """ Load a presampled spot phase."""
    global_dict = parse_global_ini()
    load_directory = global_dict["datapath"]

    savename = savename.replace(".dat", "")
    load_file = load_directory / "timesamples" / f"{savename}.dat"

    phase_sample = np.array([])
    time_sample = np.array([])
    with open(load_file, "r") as f:
        for line in f:
            cols = line.strip().split()
            phase_sample = np.append(phase_sample, float(cols[0]))
            time_sample = np.append(
                time_sample, datetime.fromisoformat(cols[1]))

    return phase_sample, time_sample


if __name__ == "__main__":
    pass
   # presample_spot_phase(60, 20, 4.3491, "N60_Np20_p4c3491")
