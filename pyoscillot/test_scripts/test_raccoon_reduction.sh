RVLIBPATH=/data/dspaeth/pyoscillot_reduced
DATAPATH=/data/dspaeth/pyoscillot_fake_spectra
# activate the venv for racconf
source ${PYENV_ROOT}/versions/raccoon-venv/bin/activate

RACCOONDATADIR=/home/dspaeth/raccoon/raccoon/data
INST="HARPS"

SIMNAME="NGC4349_blindsearch_grid_194"
fits_dir=$DATAPATH/$SIMNAME/$INST/fits
BLAZETXTFILE=$fits_dir/blazefiles.txt


raccoonccf \
      $fits_dir/*.fits \
      HARPS \
      $RACCOONDATADIR/mask/HARPS/ngc4349-127.mas \
      --filobs2blaze $BLAZETXTFILE \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/NGC4349_blindsearch_grid_194_ngcmask/HARPS_pre2015_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/HARPS_pre2015 \
      --plot_sv \
      --bervmax 100 \
      --verbose