from dataloader import phoenix_spectrum
from utils import normalize_phoenix_spectrum
from pathlib import Path
import matplotlib.pyplot as plt

# First get rid of duplicates
all_phoenix = sorted(list(Path("/home/dspaeth/pypulse/data/phoenix_spectra").glob("*.fits")))
for file in all_phoenix:
    if "(1).fits" in file.name:
        new_file = file.parent / "duplicates" / file.name
        print(f"Move {file} to {new_file}")
        file.rename(new_file)
    
# Now again
all_phoenix = sorted(list(Path("/home/dspaeth/pypulse/data/phoenix_spectra").glob("lte*.fits")))



# Create here only the models that you need
all_phoenix = []
_,_,_, filepath = phoenix_spectrum(Teff=5700, logg=4.5, feh=0.0, return_filepath=True)
all_phoenix.append(filepath)
_,_,_, filepath = phoenix_spectrum(Teff=5600, logg=4.5, feh=0.0, return_filepath=True)
all_phoenix.append(filepath)
_,_,_, filepath = phoenix_spectrum(Teff=5800, logg=4.5, feh=0.0, return_filepath=True)
all_phoenix.append(filepath)


for file in all_phoenix:
    name = file.name
    
    # Get Teff, logg, feh from the filenames
    value_str = name.split("lte")[-1].split(".PHOENIX")[0]
    Teff = int(value_str.split("-")[0])
    logg_feh_str = value_str[6:]
    feh_sign = value_str[-4]
    logg = float(logg_feh_str.split(feh_sign)[0])
    feh = float(logg_feh_str.split(feh_sign)[1]) * int(feh_sign+"1")
    
    if Teff < 4000:
        continue
    
    # Now load the full spectra
    wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(3500, 7200))
    
    _, spec_norm, continuum_interp = normalize_phoenix_spectrum(wave, spec,Teff, logg, feh, run=True)
    file_out = Path("/home/dspaeth/pypulse/pypulse/RASSINE_phoenix_spec_rassine.p")
    out_root = Path("/home/dspaeth/pypulse/data/Rassine_dataframes_for_phoenix")
    new_file = out_root / (file.stem + ".p") 
    
    file_out.rename(new_file)
    
    # Make a debug plot
    fig, ax = plt.subplots(1, figsize=(30,9))
    ax.plot(wave, spec, color="tab:blue")
    ax.plot(wave, continuum_interp, color="tab:red")
    ax.set_xlabel(r"Wavelength [$\AA]")
    ax.set_ylabel("Flux")
    ax.set_title(file.stem)
    fig.set_tight_layout(True)
    plt.savefig(out_root / "plots" / (file.stem + ".png"), dpi=500)
    plt.close()