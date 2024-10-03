#!/usr/bin/python3
import ROOT
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

files22 = [
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_100to200_1J.root', 87.89),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_100to200_2J.root', 101.4),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_200to400_1J.root', 6.319),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_200to400_2J.root', 13.81),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_400to600_1J.root', 0.2154),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_400to600_2J.root', 0.833),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_600_1J.root', 0.02587),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22_600_2J.root', 0.1574)
]
files22EE = [
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_100to200_1J.root', 87.89),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_100to200_2J.root', 101.4),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_200to400_1J.root', 6.319),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_200to400_2J.root', 13.81),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_400to600_1J.root', 0.2154),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_400to600_2J.root', 0.833),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_600_1J.root', 0.02587),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer22EE_600_2J.root', 0.1574)
]
files23 = [
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_100to200_1J.root', 87.89),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_100to200_2J.root', 101.4),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_200to400_1J.root', 6.319),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_200to400_2J.root', 13.81),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_400to600_1J.root', 0.2154),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_400to600_2J.root', 0.833),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_600_1J.root', 0.02587),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23_600_2J.root', 0.1574)
]
files23BPix = [
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_100to200_1J.root', 87.89),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_100to200_2J.root', 101.4),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_200to400_1J.root', 6.319),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_200to400_2J.root', 13.81),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_400to600_1J.root', 0.2154),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_400to600_2J.root', 0.833),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_600_1J.root', 0.02587),
    ('/eos/user/s/scavanau/SWAN_projects/JetVeto/Zto2Nu/andreas_updates/for_phietascan/Run3Summer23BPix_600_2J.root', 0.1574)
]
lumi= 1

def plot_combined(pt, combined_eff, title, path, color='b', pt_min=None, pt_max=None):
    '''
    This plots each file normalized for ONE era.
    The file path needs to be changed before running if saving somewhere different.
    '''
    plt.plot(pt, combined_eff, marker='.', color=color, label='Zto2Nu')
    plt.xlabel('Jet $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.grid()
    plt.axvline(x=40, color='red', linestyle='-')
    plt.legend()
    if pt_min is not None or pt_max is not None:
        plt.xlim(pt_min, pt_max)
    plt.savefig(path, format='png')
    plt.close()


def run(file_name, xsec, lumi):
    '''
    Grabs the arrays from the samples branches
    Applies monojet selection to the jet veto maps and calculates efficiency
    Calculates the negative weights to create a weight array later applied in plotting
    '''
    open_file = uproot.open(file_name)
    events = open_file["Events"]

    # Apply monojet and MET selections
    monojet_sel = events['Monojet_jet_inclusive_jec_Nominal_selection'].array()
    MET_sel = events['MET_trigger_selection'].array()
    monojet_mask = (monojet_sel == True) & (MET_sel == True)

    # Apply selections
    # ADD [monojet_mask] for monojet selections
    negweight = events['weight_generator_nominal'].array()[monojet_mask]
    evnum = events['event'].array()[monojet_mask]
    veto_pt = events['JetVetoMaps_veto_jets_pt'].array()[monojet_mask]

    # Calculate Negative Weights
    weights_sign = np.sign(negweight)
    pos_count = np.sum(weights_sign == 1)
    neg_count = np.sum(weights_sign == -1)
    weight_scalar = (xsec * lumi) / (pos_count - neg_count)
    scaled_weights = weights_sign * weight_scalar

    # Define the efficiency function
    def eff(pt):
        veto_mask = ~(ak.any(veto_pt > pt, axis=1))
        good = evnum[veto_mask]
        total = evnum
        return (len(good) / len(total))

    # Calculate efficiencies
    pt_range = np.linspace(0, 1000, 200)
    efficiencies = [eff(pt) for pt in pt_range]

    return pt_range, efficiencies, scaled_weights, len(evnum)


#Storing the data to apply weights for 22
results22 = [run(file, xsec, lumi) for file, xsec in files22]
common_pt_range = np.linspace(0, 100, 50)
combined_efficiencies22 = np.zeros_like(common_pt_range, dtype=float)
total_weighted_events = 0
# Summing the efficiencies weighted by event numbers
for pt_range, efficiencies, scaled_weights, evnumb in results22:
    total_weighted_events += np.sum(scaled_weights * evnumb)
    combined_efficiencies22 += np.interp(common_pt_range, pt_range, efficiencies) * np.sum(scaled_weights * evnumb)
# Normalize combined efficiency
combined_efficiencies22 /= total_weighted_events
#plot_combined(common_pt_range, combined_efficiencies22, 'Zto2NuSummer22 Jet Veto Efficiency (Combined)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/22.png', pt_min=0, pt_max=100)

#Storing data for 22EE
results22EE = [run(file, xsec, lumi) for file, xsec in files22EE]
common_pt_range = np.linspace(0, 100, 50)
combined_efficiencies22EE = np.zeros_like(common_pt_range, dtype=float)
total_weighted_events = 0
# Summing the efficiencies weighted by event numbers
for pt_range, efficiencies, scaled_weights, evnumb in results22EE:
    total_weighted_events += np.sum(scaled_weights * evnumb)
    combined_efficiencies22EE += np.interp(common_pt_range, pt_range, efficiencies) * np.sum(scaled_weights * evnumb)
# Normalize combined efficiency
combined_efficiencies22EE /= total_weighted_events
#plot_combined(common_pt_range, combined_efficiencies22EE, 'Zto2NuSummer22EE Jet Veto Efficiency (Combined)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/22EE.png', pt_min=0, pt_max=100)

#Storing data for 23
results23 = [run(file, xsec, lumi) for file, xsec in files23]
common_pt_range = np.linspace(0, 100, 50)
combined_efficiencies23 = np.zeros_like(common_pt_range, dtype=float)
total_weighted_events = 0
# Summing the efficiencies weighted by event numbers
for pt_range, efficiencies, scaled_weights, evnumb in results23:
    total_weighted_events += np.sum(scaled_weights * evnumb)
    combined_efficiencies23 += np.interp(common_pt_range, pt_range, efficiencies) * np.sum(scaled_weights * evnumb)
# Normalize combined efficiency
combined_efficiencies23 /= total_weighted_events
#plot_combined(common_pt_range, combined_efficiencies23, 'Zto2NuSummer23 Jet Veto Efficiency (Combined)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/23.png', pt_min=0, pt_max=100)

#Storing data for 23BPix
results23BPix = [run(file, xsec, lumi) for file, xsec in files23BPix]
common_pt_range = np.linspace(0, 100, 50)
combined_efficiencies23BPix = np.zeros_like(common_pt_range, dtype=float)
total_weighted_events = 0
# Summing the efficiencies weighted by event numbers
for pt_range, efficiencies, scaled_weights, evnumb in results23BPix:
    total_weighted_events += np.sum(scaled_weights * evnumb)
    combined_efficiencies23BPix += np.interp(common_pt_range, pt_range, efficiencies) * np.sum(scaled_weights * evnumb)
# Normalize combined efficiency
combined_efficiencies23BPix /= total_weighted_events
#plot_combined(common_pt_range, combined_efficiencies23BPix, 'Zto2NuSummer23BPix Jet Veto Efficiency (Combined)', '/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/23BPix.png', pt_min=0, pt_max=100)

def plot_all(pt_range, efficiencies_list, labels, title, path, pt_min=0, pt_max=100):
    '''
    This function plots the normalized efficiency plot for all eras on the same graph
    '''
    plt.figure()
    for efficiencies, label in zip(efficiencies_list, labels):
        plt.plot(pt_range, efficiencies, label=label)

    plt.xlabel('Jet $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.xlim(pt_min, pt_max)
    plt.ylim(.91, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(path, format='png')
    plt.close()

#Plotting with mathplot the graph with all eras efficiencies
labels = [
    'Zto2NuSummer22',
    'Zto2NuSummer22EE',
    'Zto2NuSummer23',
    'Zto2NuSummer23BPix'
]
efficiencies_list = [
    combined_efficiencies22,
    combined_efficiencies22EE,
    combined_efficiencies23,
    combined_efficiencies23BPix
]
#plot_all(common_pt_range, efficiencies_list, labels, 'Jet Veto Efficiency All Eras', '/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/All.png', pt_min=0, pt_max=100)


#Plotting with ROOT the graph with all eras efficiencies
colors = {
    'Zto2NuSummer22': ROOT.kBlue,
    'Zto2NuSummer22EE': ROOT.kGreen,
    'Zto2NuSummer23': ROOT.kPink,
    'Zto2NuSummer23BPix': ROOT.kMagenta
}
# Create the canvas
t1_canvas = ROOT.TCanvas("t1_canvas", "Jet Veto Efficiency All Eras", 800, 600)
t1_canvas.SetGrid()
# Create the legend
legend = ROOT.TLegend(0.6, 0.1, 0.85, 0.3)
legend.SetTextSize(0.03)
pt_bins = len(common_pt_range)
x_values = np.array(common_pt_range, dtype='double')

efficiencies_dict = {
    'Zto2NuSummer22': combined_efficiencies22,
    'Zto2NuSummer22EE': combined_efficiencies22EE,
    'Zto2NuSummer23': combined_efficiencies23,
    'Zto2NuSummer23BPix': combined_efficiencies23BPix
}

graphs_efficiencies = []
for label, efficiencies in efficiencies_dict.items():
    y_values = np.array(efficiencies, dtype='double')
    
    graph_efficiencies = ROOT.TGraph(pt_bins, x_values, y_values)
    graph_efficiencies.SetLineColor(colors[label])
    graph_efficiencies.SetLineWidth(2)
    graph_efficiencies.SetTitle("")
    graph_efficiencies.GetXaxis().SetTitle("Jet p_{T} [GeV]")
    graph_efficiencies.GetYaxis().SetTitle("Efficiency")
    graphs_efficiencies.append(graph_efficiencies)
    legend.AddEntry(graph_efficiencies, label, "l")
# Draw the graphs
first = True
for graph in graphs_efficiencies:
    if first:
        graph.SetMaximum(1.0)
        graph.SetMinimum(0.91)
        graph.GetXaxis().SetLimits(0, 100)
        graph.Draw("AL")  # "AL" for axis and line
        first = False
    else:
        graph.Draw("L SAME")  # "L" for line
legend.Draw()
# Draw a vertical line at a specific pt value
line = ROOT.TLine(40, 0.91, 40, 1.0)
line.SetLineColor(ROOT.kRed)
line.SetLineStyle(7)
line.Draw()
# Update the canvas and save it
t1_canvas.Update()
t1_canvas.Print("/eos/user/s/scavanau/SWAN_projects/JetVeto/Git/AllROOT.png")
