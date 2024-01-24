#!/bin/bash
DATAPATH=$1
RVLIBPATH=$2
SIMNAME=$3
STAR=$4
INST=$5

# activate the venv for racconf
source ${PYENV_ROOT}/versions/raccoon-venv/bin/activate

RACCOONDATADIR=/home/dspaeth/raccoon/raccoon/data

echo "run raccoon"
if [ $INST = "CARMENES_VIS" ]
then

  # First create the new mask using the SERVAL template
  raccoonmask $RVLIBPATH/serval/SIMULATION/$SIMNAME/CARMENES_VIS/$SIMNAME.fits serval $STAR \
    --inst CARM_VIS \
    --tplrv 0 \
    --cont poly \
    --contfiltmed 1 \
    --contfiltmax 400 \
    --contpolyord 2 \
    --line_fwhmmax 30.00 \
    --line_contrastminmin 0.06 \
    --line_depthw_percentdeepest 0.10 \
    --line_depthw_depthmaxquantile 0.6 \
    --dirout $RACCOONDATADIR/mask/CARM_VIS/$SIMNAME --verbose
  raccoonccf \
      $DATAPATH/fake_spectra/$SIMNAME/$INST/*.fits \
      CARM_VIS \
      $RACCOONDATADIR/mask/CARM_VIS/$SIMNAME/$SIMNAME.mas \
      --filtell $RACCOONDATADIR/tellurics/CARM_VIS/telluric_mask_carm_short.dat \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME/CARMENES_VIS_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/$INST \
      --plot_sv \
      --bervmax 100 \
      --vsini 6.0 \
      --verbose
elif [ $INST = "CARMENES_NIR" ]
then
  raccoonmask $RVLIBPATH/serval/SIMULATION/$SIMNAME/CARMENES_NIR/$SIMNAME.fits serval $STAR \
    --inst CARM_NIR \
    --tplrv 0 \
    --cont poly \
    --contfiltmed 1 \
    --contfiltmax 400 \
    --contpolyord 2 \
    --line_fwhmmax 30.00 \
    --line_contrastminmin 0.06 \
    --line_depthw_percentdeepest 0.10 \
    --line_depthw_depthmaxquantile 0.6 \
    --dirout $RACCOONDATADIR/mask/CARM_NIR/$SIMNAME --verbose

  raccoonccf \
      $DATAPATH/fake_spectra/$SIMNAME/$INST/*.fits \
      CARM_NIR \
      $RACCOONDATADIR/mask/CARM_NIR/$SIMNAME/$SIMNAME.mas \
      --filtell $RACCOONDATADIR/tellurics/CARM_NIR/telluric_mask_nir4.dat \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME/CARMENES_NIR_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/$INST \
      --plot_sv \
      --bervmax 100 \
      --verbose
elif [ $INST = "HARPS" ]
then
  echo "SKIP MASK CREATION, USE NGCMASK INSTEAD"
  # raccoonmask $RVLIBPATH/serval/SIMULATION/$SIMNAME/HARPS_pre2015/$SIMNAME.fits serval $STAR \
  #   --inst HARPS \
  #   --tplrv 0 \
  #   --cont poly \
  #   --contfiltmed 1 \
  #   --contfiltmax 400 \
  #   --contpolyord 2 \
  #   --line_fwhmmax 30.00 \
  #   --line_contrastminmin 0.06 \
  #   --line_depthw_percentdeepest 0.10 \
  #   --line_depthw_depthmaxquantile 0.6 \
  #   --dirout $RACCOONDATADIR/mask/HARPS/$SIMNAME --verbose

    # Now we need to unzip the fits files
    # Create a directory for the fits files
    fits_dir=$DATAPATH/$SIMNAME/$INST/fits
    mkdir $fits_dir
    cd $fits_dir
    FILES=$(find $DATAPATH/$SIMNAME/$INST/ -name '*.tar')
    for file in $FILES
    do
      tar -xvf $file
    done

    # Now create a blazefile list
    BLAZEFILEPATH=/data/dspaeth/pyoscillot_data/HARPS_template_ngc4349_127_blaze_A.fits
    BLAZETXTFILE=$fits_dir/blazefiles.txt
    FILES=$(find $DATAPATH/$SIMNAME/$INST/ -name '*.fits')
    for file in $FILES
    do
      echo $file $BLAZEFILEPATH >> $BLAZETXTFILE
    done


    raccoonccf \
      $fits_dir/*.fits \
      HARPS \
      $RACCOONDATADIR/mask/HARPS/ngc4349-127.mas \
      --filobs2blaze $BLAZETXTFILE \
      --rvshift none \
      --fcorrorders obshighsnr \
      --dirout $RVLIBPATH/raccoon/SIMULATION/$SIMNAME/HARPS_pre2015_CCF \
      --dirserval $RVLIBPATH/serval/SIMULATION/$SIMNAME/HARPS_pre2015 \
      --plot_sv \
      --bervmax 100 \
      --verbose
fi


# deactivate the venv
deactivate
