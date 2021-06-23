#!/bin/bash
root=/home/dane/Documents/PhD/pypulse/data
RVLIBPATH=$1
SIMNAME=$2
STAR=$3

SERVALHOME=~/Documents/PhD/serval
SERVAL=$SERVALHOME/serval/
source ${SERVALHOME}/venv/bin/activate

cd $RVLIBPATH

$SERVAL/src/serval.py $SIMNAME ${root}/fake_spectra/$SIMNAME -safemode 2 -inst CARM_VIS -targrv 0 -pspline -targ $STAR -atmmask "" -brvref DRS
