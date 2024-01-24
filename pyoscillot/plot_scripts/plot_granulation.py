import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
mpl.use('Qt5Agg')

temp_directory = Path("/home/dane/Documents/PhD/pyoscillot/mounted_data/fake_spectra/GRANULATION_MEDRES/arrays/temperature/")
velocity_directory = Path("/home/dane/Documents/PhD/pyoscillot/mounted_data/fake_spectra/GRANULATION_MEDRES/arrays/granulation_velocity/")
temp_files = sorted(list(temp_directory.glob("*npy")))
velocity_files = sorted(list(velocity_directory.glob("*npy")))


previous_temp = None

fig, ax = plt.subplots(2, 2, figsize=(16,9))
axs = ax.flatten()
print(axs)
temp_files = [temp_files[0], temp_files[10], temp_files[20], temp_files[30]]
for a, temp_file in zip(axs, temp_files):
    temp = np.load(temp_file)
    if previous_temp is not None:
        print(np.isclose(temp, previous_temp).all())
    # previous_temp = temp
    a.imshow(temp, vmin=4300, vmax=5300, cmap="hot")
    # a.axis("off")
fig.set_tight_layout(True)
plt.savefig("/home/dane/Documents/PhD/Sabine_overviews/11.05.2022/temperature.png", dpi=300)

for a, temp_file in zip(axs, velocity_files):
    temp = np.load(temp_file)
    if previous_temp is not None:
        print(np.isclose(temp, previous_temp).all())
    # previous_temp = temp
    a.imshow(temp, vmin=-1000, vmax=1000, cmap="seismic")
    # a.axis("off")
fig.set_tight_layout(True)
plt.savefig("/home/dane/Documents/PhD/Sabine_overviews/11.05.2022/velocity.png", dpi=300)

# velocity = np.load(velocity_file)
#

# plt.show()
#
# plt.imshow(velocity, vmin=-1000, vmax=1000, cmap="seismic")
# plt.show()