#!/bin/bash
cd CMSSW_14_0_0
cmsenv
cd -
eval$(/afs/cern.ch/user/s/scavanau/vetomaps/jetveto_eff.py)
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
echo
echo "Jet veto efficiency"
