import numpy as np
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
import sys
sys.path.append("/home/dane/Documents/PhD/pyCARM/pyCARM")
from plotter import plot_rv, plot_activity, plot_activity_rv, plot_rvo, plot_rv_lambda

if __name__ == "__main__":
    name = "fix_CARM"

    rv_dict = load.rv(name)
    crx_dict = load.crx(name, full=True)

    plot_rv(rv_dict)
    plt.show()
