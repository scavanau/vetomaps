#!/usr/bin/python3
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

files = [
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 87.89),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 101.4),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 6.319),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 13.81),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 0.2154),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 0.833),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 0.02587),
    ("/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/Zto2Nu-2Jets_PTNuNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", 0.1574)
]

lumi = 9.69

def run(file_name, xsec, lumi):
    open_file = uproot.open(file_name)
    events = open_file["Events"]

    # Apply monojet and MET selections
    monojet_sel = events['SR_Monojet_jet_central_selection'].array()
    MET_sel = events['MET_trigger_selection'].array()
    monojet_mask = (monojet_sel == True) & (MET_sel == True)

    # Apply selections
    evnum = events['event'].array()[monojet_mask]
    veto_pt = events['DetectorMitigation_JetVetoMaps_All_vetojets_pt'].array()[monojet_mask]
    cleaned_jet_pt = events['cleaned_central_Jet_pt'].array()[monojet_mask]
    cleaned_jet_eta = events['cleaned_central_Jet_eta'].array()[monojet_mask]
    puppimet_pt = events['TypeIPuppiMET_pt'].array()[monojet_mask]

    etamask = (cleaned_jet_eta < 2.4)
    cleaned_jet_pt = cleaned_jet_pt[etamask][:, 0]

    # Define the efficiency function
    def eff(pt):
        veto_mask = ~(ak.any(veto_pt > pt, axis=1))
        good = evnum[veto_mask]
        total = evnum
        return len(good) / len(total)

    # Calculate efficiencies
    pt_range = np.linspace(0, 1000, 200)
    efficiencies = [eff(pt) for pt in pt_range]

    # Calculate weight
    weight = (xsec * lumi) / len(evnum)

    # Return values used to graph
    return pt_range, efficiencies, weight, veto_pt, len(evnum), cleaned_jet_pt, puppimet_pt, cleaned_jet_eta
results = [run(file, xsec, lumi) for file, xsec in files]

# Function to plot histograms of variables
def plot_histograms(bins, combined_hist, combined_weighted_hist, title, path):
    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins=bins, weights=combined_hist, alpha=0.7, color='red', label='No normalization')
    ax.hist(bins[:-1], bins=bins, weights=combined_weighted_hist, color='blue', label='With normalization')
    ax.set_xlabel('pT [GeV]')
    ax.set_ylabel('Events')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(path, format='png')


#Uncomment to run function to plot individual efficiencies
#Function to plot individual efficiencies
#def plot_individual(pt, eff, title, color='b', label='Efficiency', pt_min=None, pt_max=None):
#    plt.plot(pt, eff, marker='.', color=color, label=label)
#    plt.xlabel('Jet $p_T$')
#    plt.ylabel('Efficiency')
#    plt.title(title)
#    plt.grid()
#    plt.legend()
#    if pt_min is not None or pt_max is not None:
#        plt.xlim(pt_min, pt_max)
#    plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff_unweighted.png', format='png')


# Function to plot combined efficiencies
def plot_combined(pt, combined_eff, title, color='b', pt_min=None, pt_max=None):
    plt.plot(pt, combined_eff, marker='.', color=color, label='Zto2Nu')
    plt.xlabel('Jet $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.grid()
    plt.axvline(x=15, color='red', linestyle='-')
    plt.axvline(x=40, color='red', linestyle='-')
    plt.legend()
    if pt_min is not None or pt_max is not None:
        plt.xlim(pt_min, pt_max)
    plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff_weighted.png', format='png')

# Function to plot combined efficiencies with out weights
def plot_combined_unweighted(pt, combined_eff, title, color='r', pt_min=None, pt_max=None):
    plt.plot(pt, combined_eff, marker='.', color=color, label='Unweighted Efficiency')
    plt.xlabel('Jet $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.grid()
    plt.axvline(x=15, color='red', linestyle='-')
    plt.axvline(x=40, color='red', linestyle='-')
    plt.legend()
    if pt_min is not None or pt_max is not None:
        plt.xlim(pt_min, pt_max)
    plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff_unweighted.png', format='png')

# Initialize histograms
bins = np.linspace(0, 1000, 50)
histograms = {'veto_pt': [], 'cleaned_jet_pt': [], 'puppimet_pt': [], 'cleaned_jet_eta': []}
weighted_histograms = {'veto_pt': [], 'cleaned_jet_pt': [], 'puppimet_pt': [], 'cleaned_jet_eta': []}
weights = []

# Fill histograms
for file_name, xsec in files:
    pt_range, efficiencies, weight, veto_pt, evnumb, cleaned_jet_pt, puppimet_pt, cleaned_jet_eta = run(file_name, xsec, lumi)

    ptveto_flat = ak.flatten(veto_pt)
    cleaned_jet_pt_flat = cleaned_jet_pt
    puppimet_pt_flat = puppimet_pt
    cleaned_jet_eta_flat = ak.flatten(cleaned_jet_eta)

    for hist_type, data in zip(['veto_pt', 'cleaned_jet_pt', 'puppimet_pt', 'cleaned_jet_eta'],
                               [ptveto_flat, cleaned_jet_pt_flat, puppimet_pt_flat, cleaned_jet_eta_flat]):
        hist, _ = np.histogram(data, bins=bins)
        weighted_hist, _ = np.histogram(data, bins=bins, weights=np.full(len(data), weight))

        histograms[hist_type].append(hist)
        weighted_histograms[hist_type].append(weighted_hist)

    weights.append(weight)

# Combine histograms of each sample
combined_histograms = {key: np.zeros_like(bins[:-1], dtype=float) for key in histograms.keys()}
combined_weighted_histograms = {key: np.zeros_like(bins[:-1], dtype=float) for key in histograms.keys()}

for hist_type in histograms.keys():
    for hist, weight in zip(histograms[hist_type], weights):
        combined_histograms[hist_type] += hist
        combined_weighted_histograms[hist_type] += hist * weight

# Plot combined histograms for different variable
# If adding other histograms need to update run function and add to return variables
plot_histograms(bins, combined_histograms['veto_pt'], combined_weighted_histograms['veto_pt'], 'Jet Veto pt (Monojet & MET selected)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/JetVeto_pt.png')
plot_histograms(bins, combined_histograms['cleaned_jet_pt'], combined_weighted_histograms['cleaned_jet_pt'], 'Cleaned Leading Jet pt (Monojet & MET selected)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/cleanedleadingjet_pt.png')
plot_histograms(bins, combined_histograms['puppimet_pt'], combined_weighted_histograms['puppimet_pt'], 'PuppiMET pt (Monojet & MET selected)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/PuppiMET_pt.png')
#plot_histograms(bins, combined_histograms['cleaned_jet_pt'], combined_weighted_histograms['cleaned_jet_pt'], 'Cleaned Jet pt (Monojet & MET selected)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/cleanedjet_pt.png')
#plot_histograms(bins, combined_histograms['cleaned_jet_eta'], combined_weighted_histograms['cleaned_jet_eta'], 'cleaned_jet_eta (Monojet & MET selected)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/CleanedJet_eta.png')

"""
#Uncomment to get efficiencies of each sample
# Plot individual efficiencies
for i, (file, xsec) in enumerate(files):
    pt_range, efficiencies, weight, veto_pt, evnumb, cleaned_jet_pt, puppimet_pt = results[i]
    pt_max = 200
    if '200to400' in file:
        pt_max = 400
    elif '400to600' in file:
        pt_max = 600
    elif '600' in file:
        pt_max = 1000

    plot_individual(pt_range, efficiencies, f'Individual Efficiency for {file}', pt_min=0, pt_max=pt_max)
"""

# Combine and plot efficiencies
common_pt_range = np.linspace(0, 100, 50)
combined_efficiencies = np.zeros_like(common_pt_range, dtype=float)
combined_efficiencies_unweighted = np.zeros_like(common_pt_range, dtype=float)

total_weighted_events = np.sum([weight * evnumb for weight, evnumb in zip(weights, [r[4] for r in results])])

for pt_range, efficiencies, weight, _, evnumb, _, _ in results:
    interp_func = interp1d(pt_range, efficiencies, bounds_error=False, fill_value="extrapolate")
    interpolated_efficiencies = interp_func(common_pt_range)

    weighted_efficiencies = interpolated_efficiencies * weight
    combined_efficiencies += weighted_efficiencies * evnumb / total_weighted_events
    combined_efficiencies_unweighted += interpolated_efficiencies / len(results)

plot_combined_unweighted(common_pt_range, combined_efficiencies_unweighted, 'Zto2Nu Jet Veto Efficiency (Unweighted)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff_unweighted.png', pt_min=0, pt_max=100)
plot_combined(common_pt_range, combined_efficiencies, 'Zto2Nu Jet Veto Efficiency', '/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff_weighted.png', pt_min=0, pt_max=100)

# Combined plot of weighted vs unweighted all samples
plt.figure()
plt.plot(common_pt_range, combined_efficiencies_unweighted, marker='.', color='r', label='Unweighted Efficiency')
plt.plot(common_pt_range, combined_efficiencies, marker='.', color='b', label='Weighted Efficiency')
plt.xlabel('Jet $p_T$')
plt.ylabel('Efficiency')
plt.title('Run3Summer23Bpix Zto2Nu Jet Veto Efficiency')
plt.grid()
plt.axvline(x=15, color='red', linestyle='-')
plt.axvline(x=40, color='red', linestyle='-')
plt.legend()
plt.xlim(0, 100)
plt.savefig('/eos/user/s/scavanau/SWAN_projects/JetVeto/output/23Bpix_vetoeff.png', format='png')
