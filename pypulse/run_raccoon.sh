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
if [ $INST = "CARMENES_VIS" ]
then
  raccoonccf \
      $DATAPATH/fake_spectra/$SIMNAME/$INST/*.fits \
      CARM_VIS \
      /home/dane/Documents/PhD/raccoon/raccoon/data/mask/CARM_VIS/$sp_type.mas \
      --filtell /home/dane/Documents/PhD/raccoon/raccoon/data/tellurics/CARM_VIS/telluric_mask_carm_short.dat \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME/CARMENES_VIS_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/$INST \
      --plot_sv \
      --bervmax 100 \
      --verbose
elif [ $INST = "CARMENES_NIR" ]
then
  raccoonccf \
      $DATAPATH/fake_spectra/$SIMNAME/$INST/*.fits \
      CARM_NIR \
      /home/dane/Documents/PhD/raccoon/raccoon/data/mask/CARM_NIR/$sp_type.mas \
      --filtell /home/dane/Documents/PhD/raccoon/raccoon/data/tellurics/CARM_NIR/telluric_mask_nir4.dat \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME/CARMENES_NIR_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/$INST \
      --plot_sv \
      --bervmax 100 \
      --verbose
fi


# deactivate the venv
deactivate
