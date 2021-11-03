#!/bin/bash
root=/home/dspaeth/Documents/pypulse/data
root=/home/dane/Documents/PhD/pypulse/data
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

$SERVAL/src/serval.py $SIMNAME ${root}/fake_spectra/$SIMNAME/$INST -safemode 2 -inst $SERVALINST -targrv 0 -pspline -targ $STAR -atmmask "" -brvref DRS
mv $SIMNAME/* $OUTPATH
rmdir $SIMNAME
