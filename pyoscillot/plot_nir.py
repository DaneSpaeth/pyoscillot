from pathlib import Path
from dataloader import carmenes_template
import matplotlib.pyplot as plt


fake_file = sorted(list(Path("/data/dspaeth/simulations/fake_spectra/EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_OPPOSITE/CARMENES_NIR").glob("*.fits")))[0]
template = Path("/data/dspaeth/simulations/CARMENES_NIR_templates/CARMENES_template_EV_Lac.fits")

(templ_spec, templ_cont, templ_sig, templ_wave) = carmenes_template(template)
(spec, cont, sig, wave) = carmenes_template(fake_file)

print(len(spec))

order = 8

fig, ax = plt.subplots(1, figsize=(16,9))
ax.plot(wave[order], spec[order], color="blue")
ax.plot(templ_wave[order], templ_spec[order], color="red", alpha=0.5)
ax.plot()
plt.show()