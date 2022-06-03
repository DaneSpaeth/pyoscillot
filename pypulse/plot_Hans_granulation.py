import numpy as np
import idlsave
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dataloader as load


index = 0
def updatefig(index):
    im.set_array(intensity[index, :, :])

fig, ax = plt.subplots()
im = ax.imshow(intensity[index,:,:], cmap="hot")
ani = animation.FuncAnimation(
            fig, updatefig, range(intensity.shape[0]), interval=125, blit=False, repeat=False)
#
# im.set_array(images[index])
# plt.imshow(intensity[0,:,:], cmap="hot")
ani.save("squared.gif")