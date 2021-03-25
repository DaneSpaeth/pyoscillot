#!/bin/bash
root=/home/dane/Documents/PhD/pypulse/data


SERVALHOME=~/Documents/PhD/serval
SERVAL=$SERVALHOME/serval/
source ${SERVALHOME}/venv/bin/activate

root_serval=${root}/fake_serval
mkdir -p $root_serval
cd $root_serval


star="HIP73620"
# -targ $star
# -brvref NoBaryCorr
$SERVAL/src/serval.py ${star}"_vis" ${root}/"fake_spectra" -safemode 2 -inst CARM_VIS  -targ $star

