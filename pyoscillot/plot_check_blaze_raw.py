import numpy as np
import plot_settings
import matplotlib.pyplot as plt
from dataloader import carmenes_template, harps_template


(spec, cont, sig, wave) = carmenes_template()
(spec_h, wave_h, blaze) = harps_template(spec_filename="HARPS_template_ngc4349_127_e2ds_A.fits",
                                                        wave_filename="HARPS_template_ngc4349_127_wave_A.fits",
                                                        blaze_filename="HARPS_template_ngc4349_127_blaze_A.fits")


order = 40
fig, ax = plt.subplots(1)
# ax.plot(wave[order], spec[order])
# ax.plot(wave[order], cont[order])
# ax.plot(wave[order], sig[order])
ax.plot(wave_h[order], spec_h[order])
plt.savefig(f"HARPS_blaze_test_order{order}.png")