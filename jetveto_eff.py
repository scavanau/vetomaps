import uproot
import correctionlib
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

sample = "/eos/user/s/scavanau/SWAN_projects/JetVeto/GluGluHto2Zto4Nu_PT-150_M-125_TuneCP5_13p6TeV_powheg-pythia8_22.root"
sample_EE = "/eos/user/s/scavanau/SWAN_projects/JetVeto/GluGluHto2Zto4Nu_PT-150_M-125_TuneCP5_13p6TeV_powheg-pythia8.root"

openf = uproot.open(sample)
events = openf["Events"]
openfEE = uproot.open(sample_EE)
eventsEE = openfEE["Events"]

#branches for Run3Summer22
evnum_ = events['event'].array()
luminum_ = events['luminosityBlock'].array()
runnum_ = events['run'].array()
veto_pt_ = events['DetectorMitigation_JetVetoMaps_All_vetojets_pt'].array()
veto_phi_ = events['DetectorMitigation_JetVetoMaps_All_vetojets_phi'].array()
veto_eta_ = events['DetectorMitigation_JetVetoMaps_All_vetojets_eta'].array()

#branches for Run3Summer22EE
evnum_EE = eventsEE['event'].array()
luminum_EE = eventsEE['luminosityBlock'].array()
runnum_EE = eventsEE['run'].array()
veto_pt_EE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_pt'].array()
veto_phi_EE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_phi'].array()
veto_eta_EE = eventsEE['DetectorMitigation_JetVetoMaps_All_vetojets_eta'].array()

def eff (pt):
    veto_mask = ~(ak.any(veto_pt_ > pt, axis=1))
    good = evnum_[veto_mask]
    total = evnum_
    
    'Gives the fraction of good jets over all jets for 22'
    efficiency_22 = ( (len(good)) / (len(total)) )
    return efficiency_22

def effEE (ptEE):
    veto_maskEE = ~(ak.any(veto_pt_EE > ptEE, axis=1))
    good = evnum_EE[veto_maskEE]
    total = evnum_EE
    
    'Gives the fraction of good jets over all jets for 22EE'
    efficiency_22EE = ( (len(good)) / (len(total)) )
    return efficiency_22EE

#Plotting 22
pt_values_22 = np.linspace(0, 100, 50)
efficiencies = [eff(pt) for pt in pt_values_22]
#Plotting 22EE
pt_values_22EE = np.linspace(0, 100, 50)
efficienciesEE = [effEE(ptEE) for ptEE in pt_values_22EE]

#22 vs 22EE
plt.plot(pt_values_22, efficiencies, marker='.', color='black', label='22')
plt.plot(pt_values_22EE, efficienciesEE, marker='.', color='blue', label='22EE')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$ (22 vs 22EE)')
l1 = 15
l2 = 40
plt.axvline(x=l1, color='red', linestyle='-')
plt.axvline(x=l2, color='red', linestyle='-')
plt.grid()
plt.legend()
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/22_vs_22EE_vetoeff', format='pdf')

#22
plt.plot(pt_values_22, efficiencies, marker='.', color='black', label='22')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$ (22)')
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/22_vetoeff', format='pdf')

#22EE
plt.plot(pt_values_22EE, efficienciesEE, marker='.', color='blue', label='22EE')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Jet Veto Efficiency vs $p_T$ (22EE)')
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/22EE_vetoeff', format='pdf')
