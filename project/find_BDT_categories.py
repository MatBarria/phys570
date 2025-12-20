import uproot as ur
import mplhep as hep
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from utils.labels import luminosity, x_range, n_bins, x_labels
from utils.helper import get_canvas, save_figure, get_histograms_ratio


def find_bdt_categories(era, bdt_categories, bdf_cut_max=1, iteration=0):
    print("Iteration:", iteration)
    dataset_base_dir = "../"

    bkg_samples = [
        "bbHtoGG",
        "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
        "VBFHToGG",
        "GGJets_MGG-40to80",
        "GJet_PT-40_DoubleEMEnriched_MGG-80",
        "VHtoGG",
        "GGJets_MGG-80",
        "GluGluHtoGG",
        "ttHToGG",
    ]

    signal_samples = ["GluGlutoHH_kl-1p00_kt-1p00_c2-0p00"]

    variable = "MultiBDT_output"

# BDT-based discriminants and weights
    bkg_values_dqcd = np.array([], dtype=np.float32)
    bkg_values_dtth = np.array([], dtype=np.float32)
    bkg_weights     = np.array([], dtype=np.float32)

    sig_values_dqcd = np.array([], dtype=np.float32)
    sig_values_dtth = np.array([], dtype=np.float32)
    sig_weights     = np.array([], dtype=np.float32)


# ---------- Background ----------
    for sample in bkg_samples:
        df = pd.read_parquet(
            f"./data/sim/{sample}/nominal/{sample}_merged_MultiBDT_output.parquet",
            columns=[variable, "eventWeight"]
        )

        # Extract event weights
        w_evt = df["eventWeight"].to_numpy(dtype=np.float32)

        # Extract BDT outputs from the list column
        sig = df[variable].apply(lambda lista: lista[0]).to_numpy(dtype=np.float32)
        tth = df[variable].apply(lambda lista: lista[1]).to_numpy(dtype=np.float32)
        qcd = df[variable].apply(lambda lista: lista[2]).to_numpy(dtype=np.float32)
        vh  = df[variable].apply(lambda lista: lista[3]).to_numpy(dtype=np.float32)

        # Your discriminants
        # d_qcd = sig /(sig + vh + qcd)
        # d_tth = sig / (sig + tth)

        mask = (sig + vh + qcd != 0) & (sig + tth!= 0)
        d_qcd = sig[mask] / (sig[mask]+vh[mask]+qcd[mask])
        d_tth = sig[mask] / (sig[mask]+tth[mask])
        w_evt = w_evt[mask]*(170.97/26.67)

        bkg_values_dqcd = np.concatenate([bkg_values_dqcd, d_qcd])
        bkg_values_dtth = np.concatenate([bkg_values_dtth, d_tth])
        bkg_weights     = np.concatenate([bkg_weights,     w_evt])

# ---------- Signal ----------
    print(sig_values_dqcd)
    for sample in signal_samples:
        df = pd.read_parquet(
            f"./data/sim/{sample}/nominal/{sample}_merged_MultiBDT_output.parquet",
            columns=[variable, "eventWeight"]
        )

        w_evt = df["eventWeight"].to_numpy(dtype=np.float32)

        sig = df[variable].apply(lambda lista: lista[0]).to_numpy(dtype=np.float32)
        tth = df[variable].apply(lambda lista: lista[1]).to_numpy(dtype=np.float32)
        qcd = df[variable].apply(lambda lista: lista[2]).to_numpy(dtype=np.float32)
        vh  = df[variable].apply(lambda lista: lista[3]).to_numpy(dtype=np.float32)

        # d_qcd = sig / (sig + vh + qcd)
        # d_tth = sig / (sig + tth)

        mask = (sig + vh + qcd != 0) & (sig + tth!= 0)
        d_qcd = sig[mask] / (sig[mask]+vh[mask]+qcd[mask])
        d_tth = sig[mask] / (sig[mask]+tth[mask])
        w_evt = w_evt[mask]*(170.97/26.67)
        sig_values_dqcd = np.concatenate([sig_values_dqcd, d_qcd])
        sig_values_dtth = np.concatenate([sig_values_dtth, d_tth])
        sig_weights     = np.concatenate([sig_weights,     w_evt])

    print(sig_values_dqcd)


    signal = []
    bkg_sqrt = []
    significance = []
    bdf_cut_min = 0.0
    # bdf_cut_max = 1.0
    cut_step = bdf_cut_max / 1000
    #if channel_US == "VBF":
    #    cut_step = bdf_cut_max / k50
    # if iteration == 0:
        # cut_step = bdf_cut_max / 130
        # if channel_US == "VBF":
            # cut_step = bdf_cut_max / 200
    bins = []
    ttH85= 0.938
    qcdGOD= 0.998
    # qcdGOD= 0.958
    while bdf_cut_min < bdf_cut_max:
        print("----------------------------------------------------------------")
        print("min cut:", bdf_cut_min)
        print("max cut:", bdf_cut_max)
        # print("----------------------------------------------------------------")
        bkg_bool_list = (
            (bkg_values_dtth > ttH85 )
            & (bkg_values_dqcd> bdf_cut_min)
            & (bkg_values_dqcd < bdf_cut_max)
            # (bkg_values_dqcd > qcdGOD )
            # & (bkg_values_dtth > bdf_cut_min)
            # & (bkg_values_dtth < bdf_cut_max)
        )
        signal_bool_list = (
            (sig_values_dtth > ttH85 )
            & (sig_values_dqcd > bdf_cut_min)
            & (sig_values_dqcd < bdf_cut_max)
            # (sig_values_dqcd > qcdGOD)
            # & (sig_values_dtth > bdf_cut_min)
            # & (sig_values_dtth < bdf_cut_max)
        )

        signal_events = np.sum(
            sig_weights[signal_bool_list]
        )
        bkg_events = np.sum(
            bkg_weights[bkg_bool_list]
        )
        if bkg_events <= 0:
           break
        signal.append(signal_events)
        bkg_sqrt.append(math.sqrt(bkg_events))
        # if bkg_events/np.sum(
            # bkg_weights) < .155  and bkg_events/np.sum(
            # bkg_weights) > .145:

            # print("eff", bkg_events/np.sum(
            # bkg_weights))
            # print("sig", signal_events/np.sum(
            # sig_weights))
            # print("max: ",  bdf_cut_max)
            # print("max: ",  bdf_cut_min)
        bins.append(bdf_cut_min)
        significance.append(signal_events / math.sqrt(bkg_events))
        print("Significance:", signal_events / math.sqrt(bkg_events))
        print("----------------------------------------------------------------")

        bdf_cut_min += cut_step

    fig, ax = get_canvas()

    signal = np.array(signal)
    bkg_sqrt = np.array(bkg_sqrt)
    ey = (signal/bkg_sqrt)*(1/signal + 0.25/bkg_sqrt**2)**(0.5)
    print("significanse lend: ", len(significance))
    print("ey  lend: ", len(ey))
    ax.errorbar(
        bins,
        significance,
        #ey,
        marker="o",
        linestyle="",
        markerfacecolor="black",
        color="black",
        markersize=5,
        # label=labelList[j],
    )

    # hep.cms.label(
    # data="True", label="", year=era, com="13.6", lumi=luminosity[era], ax=ax
    # )
    hep.cms.label(data="True", ax=ax, com="13.6")

    max_height = max(significance)
    best_cut = bins[significance.index(max(significance))]
    print("MAX significance: ", max(significance))
    # ax.axvline(
        # x=best_cut,
        # # ymin=0.0,
        # # ymax=max_height,
        # color="red",
        # linestyle="--",
        # alpha=0.5,
    # )

    # print("BDT > ", best_cut)
    # ax.set_yscale("log")
    # ax.set_ylim(
    # 0.001,
    # 10 * max_height,
    # )
    ax.set_ylim(0.0, 1.3 * max_height)
    # ax.set_xlim(bins[0], bins[-1])
    ax.set_xlim(0.0, 1.01)
    ax.set_ylabel(r"S/$\sqrt{B}$", loc="center")
    # ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("r$D_{ttH}$")

    output_directory = "../plots_scores_dd/"
    save_name = "BDT_cuts_ttHfirst_Cat"
    save_figure(fig, output_directory, save_name)

    bdt_categories.append(best_cut)
    #bdt_categories.append(round(best_cut, 1))
    # if max(significance) > 0.05:
    if iteration < 0:
        find_bdt_categories(era, bdt_categories, best_cut, iteration + 1)
    print(bdt_categories)

find_bdt_categories("2",[])

