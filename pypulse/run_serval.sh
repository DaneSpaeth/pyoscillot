#!/bin/bash
root=/home/dspaeth/Documents/pypulse/data
root=/home/dane/mounted_srv/simulations
# root=/home/dane/Documents/PhD/pypulse/mounted_data

RVLIBPATH=$1
SIMNAME=$2
STAR=$3
INST=$4

SERVALHOME=~/Documents/PhD/serval
SERVAL=$SERVALHOME/serval/
# source ${SERVALHOME}/venv/bin/activate

OUTPATH=$RVLIBPATH/$SIMNAME/$INST
mkdir -p $OUTPATH
cd $OUTPATH


if [ $INST = "CARMENES_VIS" ]
then
    SERVALINST="CARM_VIS"
else
    SERVALINST=$INST
fi

$SERVAL/src/serval.py $SIMNAME ${root}/fake_spectra/$SIMNAME/$INST -inst $SERVALINST -targrv 0 -pspline -targ $STAR -atmmask "" -brvref DRS -safemode 2
mv $SIMNAME/* $OUTPATH
rmdir $SIMNAME
