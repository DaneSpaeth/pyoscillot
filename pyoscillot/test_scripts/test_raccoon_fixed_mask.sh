#!/bin/bash
# activate the venv for racconf
source ${PYENV_ROOT}/versions/raccoon-venv/bin/activate

RACCOONDATADIR="/home/dspaeth/raccoon/raccoon/data"
RVLIBPATH="/home/dspaeth/pyoscillot/data/reduced"

i=84
fits_dir="/home/dspaeth/pyoscillot/data/fake_spectra/NGC4349_Test${i}/HARPS/fits"
BLAZETXTFILE=$fits_dir/"blazefiles.txt"

raccoonccf \
      $fits_dir/*.fits \
      HARPS \
      $RACCOONDATADIR/mask/HARPS/ngc4349-127.mas \
      --filtell $RACCOONDATADIR/tellurics/CARM_VIS/telluric_mask_carm_short.dat \
      --filobs2blaze $BLAZETXTFILE \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/NGC4349_Test${i}_ngcmask/HARPS_pre2015_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/NGC4349_Test${i}/HARPS_pre2015 \
      --plot_sv \
      --bervmax 100 \
      --verbose
