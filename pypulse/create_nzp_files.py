import pandas as pd
from pathlib import Path
import numpy as np
def read_in_rvs(file):
    """ Read in the rvs for the file. Return as pandas DataFrame."""
    bjd = []
    rvc = []
    rvce = []
    drift = []
    drifte = []
    rv = []
    rve = []
    berv = []
    sadrift = []
    with open(file, "r") as f:
        for line in f:
            columns = line.strip().split()
            bjd.append(float(columns[0]))
            rvc.append(float(columns[1]))
            rvce.append(float(columns[2]))
            drift.append(float(columns[3]))
            drifte.append(float(columns[4]))
            rv.append(float(columns[5]))
            rve.append(float(columns[6]))
            berv.append(float(columns[7]))
            sadrift.append(float(columns[8]))

    df = pd.DataFrame({"bjd": bjd, "rvc": rvc, "rvce": rvce, "drift": drift,
                       "drifte": drifte, "rv": rv, "rve": rve, "berv": berv,
                       "sadrift": sadrift})

    return df

def create_nzp_file(file, raccoon=False):
    """ Create a correction file for the file."""
    sim = True
    # SERVAL
    if not raccoon:
        rv_df = read_in_rvs(file)
        # Now add the RV, RVE tags for easy retrieval
        rv_df.insert(1, "RV", rv_df["rv"])
        rv_df.insert(2, "RVE", rv_df["rve"])
    # RACCOON
    else:
        rv_df = pd.read_csv(file, delimiter=" ")
        rv_df = rv_df.rename(columns={"rv": "rvc", "rverr": "rvce"})
        # Convert from km/s to m/s
        rv_df["rvc"] = rv_df["rvc"] * 1000.
        rv_df["rvce"] = rv_df["rvce"] * 1000.
        # Since for some stupid reason the BJD is not saved in the *.par.dat
        # file but only in the ccfpar.dat file we have to read in the BJD
        # separately
        bjd = np.loadtxt(str(file).replace("par.dat", "ccfpar.dat"))[:, 0]
        rv_df["bjd"] = bjd
        rv_df.insert(1, "RV", rv_df["rvc"])
        rv_df.insert(2, "RVE", rv_df["rvce"])

    if not raccoon:
        corrected_file = file.parent / \
            (file.stem + "_nzp.csv")
    else:
        if file.stem == "None.par":
            name = file.parent.parent.name
            corrected_file = file.parent / (name + ".rvc_nzp.csv")
        else:
            try:
                name = Path(file.stem).stem
            except:
                print(type(file), file)
                exit()
            corrected_file = file.parent / (name + ".rvc_nzp.csv")
    if "par_nzp" in corrected_file.name:
        corrected_file = Path(
            str(corrected_file).replace("par_nzp", "rvc_nzp"))
    if ".par.rvc" in corrected_file.name:
        corrected_file = Path(
            str(corrected_file).replace(".par.rvc", ".rvc"))

    print(f"Save DataFrame to {corrected_file}")
    rv_df.to_csv(corrected_file)

    return corrected_file

def create_ascii_file(file):
    """ Create ascii styled files with the corrected RVs."""

    df = pd.read_csv(file)
    new_file = file.parent / (file.stem + ".ascii")
    bjd = list(df["bjd"])
    rv = list(df["RV"])
    rve = list(df["RVE"])

    print(f"Write file {new_file}")
    with open(new_file, "w") as f:
        for d, r, re in zip(bjd, rv, rve):
            line = f"{round(d,6):16}{round(r,2):10}{round(re,2):10}\n"
            f.write(line)

    #
    #     # Also create the bis file
    #     new_file = file.parent / \
    #         (file.stem.replace(".rvc_nzp", "") + ".bis.ascii")
    #     bis = list(df["bis"])
    #     bise = list(df["biserr"])
    #     print(f"Write file {new_file}")
    #     with open(new_file, "w") as f:
    #         for d, b, be in zip(bjd, bis, bise):
    #             line = f"{round(d,6):16}  {round(b,8):10}  {round(be,8):10}\n"
    #             f.write(line)






if __name__ == "__main__":
    file = Path("/home/dspaeth/pypulse/data/reduced/serval/SIMULATION/NGC4349_Test2/CARMENES_VIS/NGC4349_Test2.rvc.dat")
    file = Path("/home/dspaeth/pypulse/data/reduced/raccoon/SIMULATION/NGC4349_Test2/CARMENES_VIS_CCF/None.par.dat")
    nzp_file = create_nzp_file(file, raccoon=True)
    create_ascii_file(nzp_file)