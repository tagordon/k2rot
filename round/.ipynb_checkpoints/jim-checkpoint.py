from round import lc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import numpy as np



epic_ids = ["211972086", "212002525", "211946007", "212009427"]
campaigns = ["05", "05", "05", "05"]
nf = len(epic_ids)
outfile = "test.dat"

for i, eid in enumerate(epic_ids):
    url = "https://archive.stsci.edu/hlsps/everest/v2/c" + campaigns[i] + "/" + eid[:4] + "00000/" + eid[4:] + "/hlsp_everest_k2_llc_" + eid + "-c" + campaigns[i] + "_kepler_v2.0_lc.fits"
    print("computing {0} of {1} lightcurves\n EPIC {2}".format(i+1, nf, eid))
    summaryfile = eid + "_summary.png"
    cornerfile = eid + "_corner.png"
    light_curve = lc.LightCurve.everest(url)
    light_curve.compute(mcmc=True, mcmc_draws=500, tune=500, 
                        target_accept=0.9, prior_sig=3.0, 
                        with_SHOTerm=False, cores=28)
    
    fig = plt.figure(figsize=(20, 10))
    really_outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
    outer = gridspec.GridSpecFromSubplotSpec(1, 2, 
                                             subplot_spec=really_outer[0], wspace=0.2, 
                                             hspace=0.2)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                             subplot_spec=outer[0], wspace=0.1, 
                                             hspace=0.3)
    
    # raw light curve and polynomial fit 
    ax = plt.Subplot(fig, inner[0])
    light_curve.plot_raw(ax, 'k.', label="everest flux")
    light_curve.plot_trend(ax, linewidth=3, color="#f55649", label="third order polynomial fit")
    ax.plot(light_curve.raw_t[light_curve.masked], light_curve.raw_flux[light_curve.masked], 
            'r.', 
            label="masked outliers")
    ax.set_title("Raw Everest Light Curve", fontsize=20)
    ax.set_xlabel("Time (BJD - 2454833)", fontsize=15)
    ax.set_ylabel("Flux", fontsize=15)
    ax.legend()
    fig.add_subplot(ax)

    # autocorrelation
    ax = plt.Subplot(fig, inner[1])
    mcmc_period = np.exp(light_curve.mcmc_summary["mean"]["logperiod"])
    light_curve.plot_autocor(ax, "k", linewidth=3)
    ax.axvline(light_curve.map_soln["period"], 
               color="#f55649", 
               label="maximum likelihood GP period: {:<3.3f}".format(light_curve.map_soln["period"]), 
               linewidth=3, 
               linestyle="--")
    ax.axvline(mcmc_period, color="#53bff5", label="mean period from MCMC: {:<3.3f}".format(mcmc_period), linewidth=3, linestyle="--")
    ax.set_title("Autocorrelation", fontsize=20)
    ax.set_xlabel("Lag (BJD)", fontsize=15)
    ax.set_ylabel("ACF", fontsize=15)
    ax.legend()
    fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], 
                                             wspace=0.1, hspace=0, 
                                             height_ratios=[3, 1])

    # GP prediction
    ax = plt.Subplot(fig, inner[0])
    light_curve.plot(ax, 'k.', label="normalized everest flux")
    light_curve.plot_map_soln(ax, t=np.linspace(light_curve.t[0], light_curve.t[-1], 1000), 
                              linewidth=3, 
                              color="#f55649", 
                              label="GP prediction")
    ax.set_title("Maximum-likelihood GP Prediction", fontsize=20)
    ax.set_ylabel("Normalized Flux", fontsize=15)
    fig.add_subplot(ax)
    
    # residuals
    ax = plt.Subplot(fig, inner[1])
    #ax.plot(light_curve.t, light_curve.flux - light_curve.map_soln["mu"], 'k.')
    light_curve.plot_residuals(ax, 'k.')
    ax.set_ylabel("Residuals", fontsize=15)
    ax.set_xlabel("Time (BJD - 2454833)", fontsize=15)
    fig.add_subplot(ax)

    fig.suptitle("EPIC {0}".format(light_curve.ident), fontsize=30)
    fig.savefig("{0}/{1}".format(outdir, summaryfile), dpi=200)
    light_curve.write_summary_string(outfile, 0)

    light_curve.plot_corner(smooth=True, 
                            truths=light_curve.mcmc_summary["mean"].values, 
                            truth_color="#f55649");
    plt.savefig("{0}/{1}".format(outdir, cornerfile), dpi=200)
    plt.clf()

