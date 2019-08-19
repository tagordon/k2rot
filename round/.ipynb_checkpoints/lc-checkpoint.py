import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import corner
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.stats import sigmaclip

class LightCurve:
    def __init__(self, t, flux, yerr=None):
        self.t = t
        self.raw_flux = flux
        self.varnames = ["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2", "mean"]
        self.flux = None
        self.yerr = None
        self.trend = None
        self.model = None
        self.map_soln = None
        self.trace = None
        self.mcmc_summary = None
        self.computed = False
        self.hasmcmc = False
    
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
        return cls(t, flux)
    
    def compute(self, mcmc=False, mcmc_draws=500, tune=500, target_accept=0.9, prior_sig=5.0):
        self.trend = self.get_trend(3)
        self.flux = (self.raw_flux-self.trend)/np.median(self.raw_flux)
        self.yerr = self.estimate_yerr()*np.ones(len(self.t))
        self.model, self.map_soln = self.build_model(prior_sig=prior_sig)
        if mcmc:
            self.trace = self.mcmc(draws=mcmc_draws, tune=tune, target_accept=target_accept)
            self.mcmc_summary = pm.summary(self.trace, varnames=self.varnames)
            self.hasmcmc=True
        self.computed = True
    
    def plot(self, ax, *args, **kwargs):
        ax.plot(self.t, self.flux, *args, **kwargs)
        return ax
    
    def plot_raw(self, ax, *args, **kwargs):
        ax.plot(self.t, self.raw_flux, *args, **kwargs)
        return ax
    
    def plot_trend(self, ax, *args, **kwargs):
        ax.plot(self.t, self.trend, *args, **kwargs)
        return ax
    
    def plot_map_soln(self, ax, t=None, *args, **kwargs):
        if not self.computed:
            raise Exception("Must first call compute()")
        mu, var = self.predict(t=t, return_var=True)
        ax.plot(t, mu, *args, **kwargs)
        ax.fill_between(t, mu+np.sqrt(var), mu-np.sqrt(var), *args, alpha=0.3, **kwargs)
        return ax
    
    def plot_corner(self):
        if not self.hasmcmc:
            raise Exception("Must first run mcmc by calling mcmc() or compute(mcmc=True) with mcmc=True")
        samples = pm.trace_to_dataframe(self.trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2", "mean"])
        return corner.corner(samples)
    
    def get_trend(self, n):
        res = np.polyfit(self.t, self.raw_flux, n)
        return sum([c*(self.t**i) for (i, c) in enumerate(res[::-1])])
        
    def autocor(self, max_peaks=1, min_period=0.5, max_period=100):
        results = xo.autocorr_estimator(self.t, self.flux, 
                                        max_peaks=max_peaks, 
                                        min_period=min_period, 
                                        max_period=max_period)
        lags, power = results["autocorr"]
        peaks = results["peaks"]
        return lags, power, peaks
    
    def plot_autocor(self, ax, *args, max_peaks=1, min_period=0.5, max_period=100, **kwargs):
        lags, power, peaks = self.autocor(max_peaks=max_peaks, min_period=min_period, max_period=max_period)
        fig = plt.figure()
        ax.plot(lags, power, "k")
        ax.axvline(peaks[0]["period"], color="k", lw=4, alpha=0.3)
        return ax
    
    def estimate_yerr(self, kernel_size=21, sigma=3):
        filt = medfilt(self.flux, kernel_size=kernel_size)
        return np.std(sigmaclip(self.flux-filt, low=sigma, high=sigma)[0])
    
    def build_model(self, prior_sig=5.0):
        lags, power, peaks = self.autocor()
        with pm.Model() as model:

            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            logs2 = pm.Normal("logs2", mu=2*np.log(self.yerr[0]), sd=prior_sig)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux)), sd=prior_sig)
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=np.log(50))
            logperiod = BoundedNormal("logperiod", mu=np.log(peaks[0]["period"]), sd=prior_sig)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=2*prior_sig)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=2*prior_sig)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
            period = pm.Deterministic("period", tt.exp(logperiod))

            # Set up the Gaussian Process model
            kernel = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=period,
                log_Q0=logQ0,
                log_deltaQ=logdeltaQ,
                mix=mix
            )
            gp = xo.gp.GP(kernel, self.t, self.yerr**2 + tt.exp(logs2), J=4)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(self.flux - mean))

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("mu", gp.predict())
            map_soln = xo.optimize(start=model.test_point)
            return model, map_soln
        
    def mcmc(self, draws=500, tune=500, target_accept=0.9):
        sampler = xo.PyMC3Sampler(finish=200)
        with self.model:
            sampler.tune(tune=tune, start=self.map_soln, step_kwargs=dict(target_accept=target_accept))
            trace = sampler.sample(draws=draws)
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
            gp.log_likelihood(self.flux - self.map_soln["mean"])
            mu, var = xo.eval_in_model(gp.predict(t, return_var=return_var), self.map_soln)
            return mu, var