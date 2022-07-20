#!/bin/bash
DATAPATH=$1
RVLIBPATH=$2
SIMNAME=$3
STAR=$4
INST=$5

sp_type=SIMULATION_73620

# activate the venv for racconf
source /home/dane/Documents/PhD/raccoon/venv/bin/activate


echo "run raccoon"
raccoonccf \
    $DATAPATH/fake_spectra/$SIMNAME/$INST/*.fits \
    CARM_VIS \
    /home/dane/Documents/PhD/raccoon/raccoon/data/mask/CARM_VIS/$sp_type.mas \
    --filtell /home/dane/Documents/PhD/raccoon/raccoon/data/tellurics/CARM_VIS/telluric_mask_carm_short.dat \
    --rvshift none \
    --fcorrorders obshighsnr \
    --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME \
    --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME \
    --plot_sv \
    --bervmax 100 \
    --verbose


# deactivate the venv
deactivate
