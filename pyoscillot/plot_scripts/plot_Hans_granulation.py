import numpy as np
import idlsave
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dataloader as load

out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")

intensity = load.granulation_map()
index = 0
def updatefig(index):
    im.set_array(intensity[index, :, :])
    print(index)
    t_s = index*200
    t_h = t_s / (3600)
    ax.set_title(f"t={round(t_h, 2)}h")

fig, ax = plt.subplots()
im = ax.imshow(intensity[index,:,:], cmap="hot")
ani = animation.FuncAnimation(
            fig, updatefig, range(intensity.shape[0]), interval=125, blit=False, repeat=False)
#
# im.set_array(images[index])
# plt.imshow(intensity[0,:,:], cmap="hot")
ani.save(out_dir / "longer.gif")