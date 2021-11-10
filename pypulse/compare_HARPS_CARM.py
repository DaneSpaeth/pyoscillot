import numpy as np
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
import sys
sys.path.append("/home/dane/Documents/PhD/pyCARM/pyCARM")
from plotter import plot_rv, plot_activity, plot_activity_rv, plot_rvo, plot_rv_lambda

if __name__ == "__main__":
    name = "HARPS_CARM_test"

    rv_dict = load.rv(name)
    crx_dict = load.crx(name, full=True)

    rvo_dict = load.rvo(name)

    plot_rv_lambda(rvo_dict, crx_dict, instrument="HARPS")
    plt.show()
