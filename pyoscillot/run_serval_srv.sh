#!/bin/bash
DATAPATH=$1
RVLIBPATH=$2
SIMNAME=$3
STAR=$4
INST=$5

if [ $INST = "HARPS" ]
then
  OUTINST="HARPS_pre2015"
else
  OUTINST=$INST
fi

OUTPATH=$RVLIBPATH/serval/SIMULATION/$SIMNAME/$OUTINST
mkdir -p $OUTPATH
cd $OUTPATH


if [ $INST = "CARMENES_VIS" ]
then
    SERVALINST="CARM_VIS"
elif [ $INST = "CARMENES_NIR" ]
then
    SERVALINST="CARM_NIR"
else
    SERVALINST=$INST
fi

$SERVAL/src/serval.py ${SIMNAME} $DATAPATH/$SIMNAME/$INST -inst $SERVALINST -targrv 0 -atmmask "" -targ $STAR -brvref DRS -safemode 2
mv $SIMNAME/* $OUTPATH
echo "Move SERVAL results to $OUTPATH"
rmdir $SIMNAME
