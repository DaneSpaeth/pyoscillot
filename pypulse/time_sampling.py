from datetime import datetime,  timedelta
import random
import numpy as np

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