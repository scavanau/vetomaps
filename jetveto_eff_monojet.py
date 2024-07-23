#!/usr/bin/python3
import uproot
import correctionlib
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

sample = "/eos/user/s/scavanau/SWAN_projects/JetVeto/GluGluHto2Zto4Nu_PT-150_M-125_TuneCP5_13p6TeV_powheg-pythia8_22.root"
sampleEE = "/eos/user/s/scavanau/SWAN_projects/JetVeto/GluGluHto2Zto4Nu_PT-150_M-125_TuneCP5_13p6TeV_powheg-pythia8.root"

openf = uproot.open(sample)
events = openf["Events"]
openfEE = uproot.open(sampleEE)
eventsEE = openfEE["Events"]

#Monojet Selection Filter
monojet_sel = events['signal_CR_Monojet_selection'].array()
monojet_sel_EE = eventsEE['signal_CR_Monojet_selection'].array()
MET_sel = events['MET_trigger_selection'].array()
MET_sel_EE = eventsEE['MET_trigger_selection'].array()

Monojet = (monojet_sel == True) & (MET_sel == True)
MonojetEE = (monojet_sel_EE == True) & (MET_sel_EE == True)

evtotal = events['event'].array()
evnum = events['event'].array()[Monojet]
luminum = events['luminosityBlock'].array()[Monojet]
runnum = events['run'].array()[Monojet]
veto_pt = events['DetectorMitigation_JetVetoMaps_All_vetojets_pt'].array()[Monojet]
veto_phi = events['DetectorMitigation_JetVetoMaps_All_vetojets_phi'].array()[Monojet]
veto_eta = events['DetectorMitigation_JetVetoMaps_All_vetojets_eta'].array()[Monojet]

evtotalEE = eventsEE['event'].array()
evnumEE = eventsEE['event'].array()[MonojetEE]
luminumEE = eventsEE['luminosityBlock'].array()[MonojetEE]
runnumEE = eventsEE['run'].array()[MonojetEE]
veto_ptEE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_pt'].array()[MonojetEE]
veto_phiEE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_phi'].array()[MonojetEE]
veto_etaEE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_eta'].array()[MonojetEE]

eff_22_monojet = []
eff_22EE_monojet = []
vetosum = []
vetosumEE = []
pt_values = range(0, 100, 2)

for pt_M in pt_values:
    veto_mask_monojet = ~(ak.any(veto_pt > pt_M, axis=1))
    good = evnum[veto_mask_monojet]
    total = evtotal
    efficiency = (len(good)) / (len(total))
    eff_22_monojet.append(efficiency)
    vetosum.append((veto_mask_monojet))

for pt_MEE in pt_values:
    veto_mask_monojetEE = ~(ak.any(veto_ptEE > pt_MEE, axis=1))
    goodEE = evnumEE[veto_mask_monojetEE]
    totalEE = evtotalEE
    efficiencyEE = (len(goodEE)) / (len(totalEE))
    eff_22EE_monojet.append(efficiencyEE)
    vetosumEE.append((veto_mask_monojetEE))

#Plots with Monojet Selections
#22 vs 22EE
plt.plot(pt_values, eff_22_monojet, marker='.', color='orange', label='22 with Monojet')
plt.plot(pt_values, eff_22EE_monojet, marker='.', color='red', label='22EE with Monojet')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$ with Monojet Selections')
plt.legend()
plt.grid()
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/22_vs_22EE_vetoeff_monojet', format='pdf')
plt.show()

#22
plt.plot(pt_values, eff_22_monojet, marker='.', color='orange', label='22 with Monojet')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$ with Monojet Selections')
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/22_vetoeff_monojet', format='pdf')
plt.show()

#22EE
plt.plot(pt_values, eff_22EE_monojet, marker='.', color='red', label='22EE with Monojet')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$')
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/22EE_vetoeff_monojet', format='pdf')
plt.show()
