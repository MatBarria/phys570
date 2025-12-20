import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)
import os
import time

# print("esto")
# print("Program starts now.")
# time.sleep(3)  # Pause execution for 3 seconds

def get_canvas(draw_ratio=False):
    if draw_ratio:
        fig, axs = plt.subplots(2, 1, height_ratios=[10, 2])
        fig.subplots_adjust(hspace=0.1)
        return fig, axs

    fig, axs = plt.subplots(1, 1)
    return fig, axs


def save_figure(fig, outputDirectory, name):

    os.makedirs(outputDirectory, exist_ok=True)
    # fig.savefig(outputDirectory + name + ".pdf", bbox_inches="tight")
    # fig.savefig(outputDirectory + name + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(outputDirectory + name + ".pdf")
    fig.savefig(outputDirectory + name + ".png", dpi=300)
    print(outputDirectory + name + " Has been created")


samples = [
    "bbHtoGG",
    "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
    "GluGlutoHH_kl-1p00_kt-1p00_c2-0p00",
    "VBFHToGG",
    "GGJets_MGG-40to80",
    "GJet_PT-40_DoubleEMEnriched_MGG-80",
    "VHtoGG",
    "GGJets_MGG-80",
    "GluGluHtoGG",
    "ttHToGG",
    # "",
]

samples_name = {
        "bbHtoGG":"bbHtoGG",
        "GJet_PT-20to40_DoubleEMEnriched_MGG-80":"GJet_pt20-40",
    "GluGlutoHH_kl-1p00_kt-1p00_c2-0p00":"Signal",
    "VBFHToGG":"VBFHToGG",
    "GGJets_MGG-40to80":"GGJets_MGG40-80",
    "GJet_PT-40_DoubleEMEnriched_MGG-80":"GGJets_pt80",
    "VHtoGG":"VH",
    "GGJets_MGG-80":"GGJets_MGG-80",
    "GluGluHtoGG":"ggHToGG",
    "ttHToGG":"ttH",
    # "",
    }

variables_name = ["Signal_prob", "ttH_prob", "QCD_prob", "VH_prob", "DttH", "D_QCD", "mass_bb"]

var_index = 0;
variable = "MultiBDT_output"
# variable = "Res_HHbbggCandidate_mass"
todas_las_variables = []

colores = [
    '#1f77b4',  # azul
    '#ff7f0e',  # naranja
    '#2ca02c',  # verde
    '#d62728',  # rojo
    '#9467bd',  # púrpura
    '#8c564b',  # marrón
    '#e377c2',  # rosa
    '#7f7f7f',  # gris
    '#bcbd22',  # lima/amarillo-verdoso
    '#17becf'   # cian
]

fig, ax = get_canvas()
for idx, sample in enumerate(samples):
    df = pd.read_parquet(
        "./data/sim/" + sample + "/nominal/" +sample+ "_merged_MultiBDT_output.parquet",
        columns=[variable]
    )
    if var_index <=3 :
        var = df[variable].apply(lambda lista: lista[var_index])
        var = var[var !=0]
    
    elif var_index == 4:
        sig  = df[variable].apply(lambda lista: lista[0])
        tth  = df[variable].apply(lambda lista: lista[1])
        var = sig/(sig+tth)
    
    elif var_index == 5:
        sig  = df[variable].apply(lambda lista: lista[0])
        qcd = df[variable].apply(lambda lista: lista[2])
        vh  = df[variable].apply(lambda lista: lista[3])
        tth  = df[variable].apply(lambda lista: lista[1])
        mask = (sig + vh + qcd != 0) & (sig + tth!= 0)
        var = sig[mask] / (sig[mask]+vh[mask]+qcd[mask])
        print(len(var))
        # var = sig/(sig+vh+qcd)
    elif var_index == 6:
        var  = df[variable]
    else:
        print("wrong var_index")
        exit()

    histogram, bins = np.histogram(
        var,
        40,
        # (0, 0.7, 1),
        (0, 1),
    )

    hep.histplot(
        histogram / np.sum(histogram),
        bins,
        yerr=False,
        ax=ax,
        color=colores[idx],
        linewidth=2,
        label=samples_name[sample],
    )


ax.set_ylabel("Events / Total events")
ax.set_xlabel(variables_name[var_index])
# ax.set_yscale("log")
ax.legend(loc="best")
# ax.set_ylim(0,0.1)
# plt.show()

save_figure(fig, "./plots/", variable + "_" + variables_name[var_index])
# save_figure(fig, "./", variable)
