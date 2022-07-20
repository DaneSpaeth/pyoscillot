#!/bin/bash
DATAPATH=$1
RVLIBPATH=$2
SIMNAME=$3
STAR=$4
INST=$5

SERVALHOME=~/Documents/PhD/serval
SERVAL=$SERVALHOME/serval/
# source ${SERVALHOME}/venv/bin/activate

OUTPATH=$RVLIBPATH/serval/SIMULATION/$SIMNAME/$INST
mkdir -p $OUTPATH
cd $OUTPATH


if [ $INST = "CARMENES_VIS" ]
then
    SERVALINST="CARM_VIS"
else
    SERVALINST=$INST
fi

$SERVAL/src/serval.py $SIMNAME $DATAPATH/fake_spectra/$SIMNAME/$INST -inst $SERVALINST -targrv 0 -pspline -targ $STAR -atmmask "" -brvref DRS -safemode 2
mv $SIMNAME/* $OUTPATH
rmdir $SIMNAME
