import numpy as np
from dataloader import carmenes_template
from pathlib import Path
from parse_ini import parse_global_ini
from shutil import copytree
from astropy.io import fits

global_dict = parse_global_ini()
DATAROOT = global_dict["datapath"]

folders = ["2SPOTS_EQU_CLOSER_uniform_template",
           "2SPOTS_EQU_CLOSER_HIP73620_template",
           "2SPOTS_EQU_CLOSER_YZ_CMi_template",
           "2SPOTS_EQU_CLOSER_V_Ori_template"]

templates = ["CARMENES_template_EV_Lac.fits",
             "CARMENES_template_HIP73620.fits",
             "CARMENES_template_YZ_CMi.fits",
             "CARMENES_template_V_Ori.fits"]

for folder, template in zip(folders, templates):
    directory = DATAROOT / "fake_spectra" / folder
    new_folder = DATAROOT / "fake_spectra" / (folder+"_NAN")
    copytree(directory, new_folder)


    for channel in ["CARMENES_VIS", "CARMENES_NIR"]:
        (spec_templ, cont_templ, sig_templ,
         wave_templ) = carmenes_template(f"{channel}_templates/"+template)
        channel_folder = new_folder / channel
        files = channel_folder.glob("*.fits")

        for file in files:
            spec, _, _, wave = carmenes_template(file)

            try:
                if (wave != wave_templ).any():
                    raise ValueError
            except AttributeError:
                print(wave != wave_templ)
                exit()
            for order in range(len(wave_templ)):
                order_spec = spec[order]
                nan_mask = np.isnan(sig_templ[order])
                order_spec[nan_mask] = np.nan
                spec[order] = order_spec

            with fits.open(file, mode="update") as hdul:
                hdul[1].data = spec
                hdul.flush()




