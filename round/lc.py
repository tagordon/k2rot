import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import corner
import os
import datetime
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.stats import sigmaclip
from astropy.stats import sigma_clip
from copy import deepcopy

class LightCurve:
    def __init__(self, t, flux, ident, telescope, yerr=None):
        self.t = t
        self.raw_t = t
        self.raw_flux = flux
        self.varnames = ["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2"]
        self.ident = ident
        self.telescope = telescope
        self.flux = None
        self.masked = None
        self.yerr = None
        self.trend = None
        self.model = None
        self.map_soln = None
        self.trace = None
        self.mcmc_summary = None
        self.computed = False
        self.hasmcmc = False
        self.normalized = False
        self.acfpeaks = None
        self.maxpeak = None
        self.lags = None
        self.power = None
        
    def __getitem__(self, key):
        new_lc = deepcopy(self)
        new_lc.__init__(self.t[key], self.raw_flux[key], self.ident, self.telescope)
        return new_lc
    
    @classmethod
    def everest(cls, everest_fits):
        with fits.open(everest_fits) as hdus:
            data = hdus[1].data
            hdr = hdus[1].header
        t = data["TIME"]
        flux = data["FLUX"]
        m = (data["QUALITY"] == 0) & np.isfinite(t) & np.isfinite(flux)
        t = np.ascontiguousarray(t[m], dtype=np.float64)
        flux = np.ascontiguousarray(flux[m], dtype=np.float64)
        ident = hdr["KEPLERID"]
        return cls(t, flux, ident, "KEPLER")
    
    @classmethod 
    def TESS(cls, tess_fits):
        with fits.open(tess_fits) as hdus:
            data = hdus[1].data
            hdr = hdus[1].header
        t = data["TIME"]
        flux = data["PDCSAP_FLUX"]
        m = (data["QUALITY"] == 0) & np.isfinite(t) & np.isfinite(flux)
        t = np.ascontiguousarray(t[m], dtype=np.float64)
        flux = np.ascontiguousarray(flux[m], dtype=np.float64)
        ident = hdr["TICID"]
        return cls(t, flux, ident, "TESS")
    
    @classmethod 
    def concatenate(cls, lcs):
        if any([lc.normalized is None for lc in lcs]):
            raise Exception("All light curves must be normalized before concatenation")
        t, flux = np.array([]), np.array([])
        ident = lcs[0].ident
        telescope = ""
        for lc in lcs:
            np.append(t, lc.t)
            np.append(t, lc.flux)
            telescope = telescope + lc.telescope
        return cls(t, flux, ident, telescope)
            
            
    def normalize(self, trendgaps=False):
        if trendgaps == False:
            self.trend = self.get_trend_nogaps(3)
        else:
            self.trend = self.get_trend(3)
        self.flux = (self.raw_flux-self.trend)/np.median(self.raw_flux)
        self.masked, self.yerr = self.estimate_yerr()
        self.flux = self.flux[self.masked == False]
        self.t = self.t[self.masked == False]
        self.yerr = self.yerr*np.ones(len(self.t))
        self.normalized = True
    
    def compute(self, trendgaps=False, mcmc=False, mcmc_draws=500, tune=500, target_accept=0.9, prior_sig=5.0, maxper=50.0, with_SHOTerm=False, cores=4):
        self.normalize(trendgaps=trendgaps)
        self.model, self.map_soln = self.build_model(prior_sig=prior_sig, maxper=maxper, with_SHOTerm=with_SHOTerm)
        if mcmc:
            self.trace = self.mcmc(draws=mcmc_draws, tune=tune, target_accept=target_accept, cores=cores)
            self.mcmc_summary = pm.summary(self.trace, varnames=self.varnames)
            self.hasmcmc=True
        self.computed = True
        
    def build_mcmc_summary(self):
        if not self.hasmcmc:
            raise Exception("Must first run mcmc")
        nvar = len(self.varnames)
        ncols = 7
        columns = ["variable\n", "mean\n", "sd\n", "mc_error\n", "hpd_2.5\n", "hpd_97.5\n", "n_eff\n", "Rhat\n"]
        for i in range(nvar):
            columns[0] += "\n{0}".format(self.varnames[i])
        for i in range(ncols):
            for j in range(nvar):
                columns[i+1] += "\n{0:0.3f}".format(self.mcmc_summary.values[j, i])
        return columns
    
    def write_summary_string(self, file, campaignnum):
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("Generated with round.py version 0.1 on {0}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
                f.write("Campaign: {0}\n".format(campaignnum))
                f.write("EPIC number   \tvariable name\tmean    \tsd      \tmc_error\thpd_2.5 \thpd_97.5\tn_eff   \tRhat    \n\n")
        if not self.hasmcmc:
            raise Exception("Must first run mcmc")
        with open(file, "a+") as f:
            epicnum = ["EPIC " + str(self.ident)]
            sumstring = [np.hstack((epicnum, self.varnames[i], self.mcmc_summary.values[i])) for i in range(len(self.mcmc_summary.values))]
            fmtarray = ["%-8.8s"]*len(self.mcmc_summary.values[0])
            fmtarray = ["%s", "%-13s"] + fmtarray
            np.savetxt(f, sumstring, fmt=fmtarray, delimiter="\t")
            sumstring = np.hstack((epicnum, ["acfpeak"], str(self.maxpeak)))
            np.savetxt(f, [sumstring], fmt=fmtarray[:3], delimiter="\t")
    
    def build_det_summary(self):
        if not self.hasmcmc:
            raise Exception("Must first run mcmc")
        names = ["period", "amplitude", "variance"]
        summary = pm.summary(self.trace, varnames=names)
        nvar = 3
        ncols = 3
        columns = ["variable\n", "mean\n", "sd\n", "mc_error\n"]
        for i in range(nvar):
            columns[0] += "\n{0}".format(names[i])
        for i in range(ncols):
            for j in range(nvar):
                columns[i+1] += "\n{0:0.03e}".format(summary.values[j, i])
        return columns
    
    def plot(self, ax, *args, **kwargs):
        ax.plot(self.t, self.flux, *args, **kwargs)
        return ax
    
    def plot_raw(self, ax, *args, **kwargs):
        ax.plot(self.raw_t, self.raw_flux, *args, **kwargs)
        return ax
    
    def plot_trend(self, ax, *args, **kwargs):
        ax.plot(self.t, self.trend[self.masked == False], *args, **kwargs)
        return ax
    
    def plot_map_soln(self, ax, t=None, *args, **kwargs):
        if not self.computed:
            raise Exception("Must first call compute()")
        mu, var = self.predict(t=t, return_var=True)
        ax.plot(t, mu, *args, **kwargs)
        ax.fill_between(t, mu+np.sqrt(var), mu-np.sqrt(var), *args, alpha=0.3, **kwargs)
        return ax
    
    def plot_residuals(self, ax, *args, **kwargs):
        if not self.computed:
            raise Exception("Must first call compute()")
        mu = self.predict(t=self.t, return_var=False)
        ax.plot(self.t, self.flux-mu, *args, **kwargs)
    
    def plot_corner(self, *args, **kwargs):
        if not self.hasmcmc:
            raise Exception("Must first run mcmc by calling mcmc() or compute(mcmc=True) with mcmc=True")
        samples = pm.trace_to_dataframe(self.trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2"])
        columns = self.build_mcmc_summary()
        corn = corner.corner(samples, *args, **kwargs)
        for i in range(len(columns)):
            plt.annotate(columns[i], xy=(0.38+0.08*i, 0.7), xycoords="figure fraction", fontsize=12)
        columns = self.build_det_summary()
        for i in range(len(columns)):
            plt.annotate(columns[i], xy=(0.55+0.08*i, 0.6), xycoords="figure fraction", fontsize=12)
        plt.annotate("EPIC {0}".format(self.ident), xy=(0.4, 0.95), xycoords="figure fraction", fontsize=30)
        return corn
    
    def write_summary_line(self):
        if not self.hasmcmc:
            raise Exception("Must first run mcmc by calling mcmc() or compute(mcmc=True) with mcmc=True")
        samples = pm.trace_to_dataframe(self.trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2"])
        
    
    def get_trend(self, n):
        gaps = np.where((np.diff(self.t) > 10*(self.t[1]-self.t[0])) == True)[0]
        gaps = np.concatenate([[0], gaps, [-1]])
        res = np.zeros((len(gaps), n+1))
        trend = []
        for i in range(len(gaps)-1):
            if gaps[i+1] == -1:
                temp_t = self.t[gaps[i]:]
                temp_flux = self.raw_flux[gaps[i]:]
            else:
                temp_t = self.t[gaps[i]:gaps[i+1]]
                temp_flux = self.raw_flux[gaps[i]:gaps[i+1]]
            res[i, :] = np.polyfit(temp_t, temp_flux, n)
            trend = np.concatenate([trend, sum([c*(temp_t**i) for (i, c) in enumerate(res[i,:][::-1])])])
        return trend
    
    def get_trend_nogaps(self, n):
        res = np.polyfit(self.t, self.raw_flux, n)
        trend = sum([c*(self.t**i) for (i, c) in enumerate(res[::-1])])
        return trend
        
    def autocor(self, max_peaks=1, min_period=0.5):
        results = xo.autocorr_estimator(self.t, self.flux, 
                                        max_peaks=max_peaks, 
                                        min_period=min_period)
        self.lags, self.power = results["autocorr"]
    
    def get_peaks(self, max_peaks=1, min_period=0.5):
        peak_ind = np.array(range(1, len(self.power)-1))[[(self.power[i+1] < self.power[i]) & (self.power[i-1] < self.power[i]) for i in range(1, len(self.power)-1)]]
        trough_ind = np.array(range(1, len(self.power)-1))[[(self.power[i+1] > self.power[i]) & (self.power[i-1] > self.power[i]) for i in range(1, len(self.power)-1)]]

        peaks = self.lags[1:][peak_ind]
        troughs = self.lags[1:][trough_ind]

        if (len(troughs) == 0) or (len(peaks) == 0):
            return [], [], [] 
        if len(troughs) == len(peaks):
            if troughs[0] < peaks[0]:
                troughs = np.append(troughs, self.power[-1])
                trough_ind = np.append(trough_ind, len(self.power)-1)
            if troughs[0] > peaks[0]:
                troughs = np.insert(troughs, 0, self.power[0])
                trough_ind = np.insert(trough_ind, 0, 0)
        heights = np.array([2*self.power[peak_ind[i]] - self.power[trough_ind[i]] - self.power[trough_ind[i+1]] for i in range(len(troughs)-1)])
        return peaks, troughs, heights
    
    def plot_autocor(self, ax, *args, max_peaks=1, min_period=0.5, maxper=50.0, **kwargs):
        if not self.computed:
            raise Exception("Must first call compute()")
        ax.plot(self.lags[self.lags < maxper], self.power[self.lags < maxper], *args, **kwargs)
        if self.maxpeak is not None:
            ax.axvline(self.maxpeak, color="#f55649", lw=5, alpha=0.6, label="chosen ACF peak: {:<3.3f}".format(self.maxpeak))
        return ax        
    
    def estimate_yerr(self, kernel_size=21, sigma=3):
        filt = medfilt(self.flux, kernel_size=kernel_size)
        longfilt = medfilt(self.flux, kernel_size=5*kernel_size)
        clipped_arr = sigma_clip(self.flux-filt, sigma=sigma)
        long_clipped_arr = sigma_clip(self.flux-longfilt, sigma=sigma)
        return long_clipped_arr.mask, np.std(clipped_arr.data[clipped_arr.mask == 0])
    
    def build_model(self, prior_sig=5.0, maxper=50.0, with_SHOTerm=True):
        self.autocor(min_period=0.5)
        peaks, troughs, heights = self.get_peaks()
        self.acfpeaks = peaks
        if len(peaks) == 0:
            startperiod = np.mean(self.t)
        else:
            halfper = (self.t[-1]-self.t[0])/2.0
            if halfper < maxper:
                maxper = (self.t[-1]-self.t[0])/2.0
            searchpeaks = peaks[peaks < maxper]
            searchheights = heights[peaks < maxper]
            if len(searchheights) > 0:
                self.maxpeak = searchpeaks[searchheights == max(searchheights)][0]
                startperiod = self.maxpeak
            else:
                startperiod = np.mean(self.t)
        with pm.Model() as model:

            #mean = pm.Normal("mean", mu=0.0, sd=10.0)
            logs2 = pm.Normal("logs2", mu=2*np.log(self.yerr[0]), sd=prior_sig)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux)), sd=prior_sig)
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=np.log(maxper))
            logperiod = BoundedNormal("logperiod", mu=np.log(startperiod), sd=prior_sig)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=2*prior_sig)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=2*prior_sig)
            mix = pm.Uniform("mix", lower=0, upper=1.0)
            
            # track deterministics
            amplitude = pm.Deterministic("amplitude", tt.exp(logamp))
            period = pm.Deterministic("period", tt.exp(logperiod))
            variance = pm.Deterministic("variance", tt.exp(logs2))

            # Set up the Gaussian Process model
            kernel = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=period,
                log_Q0=logQ0,
                log_deltaQ=logdeltaQ,
                mix=mix
            )
            
            if with_SHOTerm:
                
                # parameters of the SHOTerm kernel 
                logSHOamp = pm.Normal("logSHOamp", mu=np.log(np.var(self.flux)), sd=prior_sig)
                w0Bound = pm.Bound(pm.Normal, lower=-4, upper=4)
                logSHOw0 = w0Bound("logSHOw0", mu=-1.0, sd=2*prior_sig)
                logS0 = pm.Deterministic("logS0", logSHOamp - logSHOw0)
                
                kernel = kernel + xo.gp.terms.SHOTerm(
                    log_S0=logS0,
                    log_Q=1/tt.sqrt(2),
                    log_w0=logSHOw0
                )
                J = 6
            else:
                J = 4
            
            gp = xo.gp.GP(kernel, self.t, self.yerr**2 + tt.exp(logs2), J=J)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(self.flux))

            # Compute the mean model prediction for plotting purposes
            #pm.Deterministic("mu", gp.predict())
            map_soln = xo.optimize(start=model.test_point, verbose=False)
            return model, map_soln
        
    def mcmc(self, draws=500, tune=500, target_accept=0.9, progressbar=False, cores=4):
        sampler = xo.PyMC3Sampler(finish=200)
        with self.model:
            #sampler.tune(tune=tune, start=self.map_soln, step=xo.get_dense_nuts_step(), step_kwargs=dict(target_accept=target_accept), progressbar=progressbar, cores=cores)
            sampler.tune(tune=tune, start=self.map_soln, step_kwargs=dict(target_accept=target_accept), progressbar=progressbar, cores=cores)
            trace = sampler.sample(draws=draws, progressbar=progressbar, cores=cores)
        return trace
    
    def predict(self, t=None, return_var=True):
        if t is None:
            t = self.t
        with self.model:
            kernel = xo.gp.terms.RotationTerm(
                log_amp=self.map_soln["logamp"],
                period=self.map_soln["period"],
                log_Q0=self.map_soln["logQ0"],
                log_deltaQ=self.map_soln["logdeltaQ"],
                mix=self.map_soln["mix"]
            )
            gp = xo.gp.GP(kernel, self.t, self.yerr**2 + tt.exp(self.map_soln["logs2"]), J=4)
            gp.log_likelihood(self.flux)
            if return_var:
                mu, var = xo.eval_in_model(gp.predict(t, return_var=True), self.map_soln)
                return mu, var
            else:
                mu = xo.eval_in_model(gp.predict(t, return_var=False), self.map_soln)
                return mu
