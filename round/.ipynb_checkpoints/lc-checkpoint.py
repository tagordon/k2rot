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
        self.flux = flux
        self.raw_flux = flux
        self.varnames = ["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2", "mean"]
        self.model = None
        self.map_soln = None
        self.trace = None
        self.mcmc_summary = None
        self.computed = False
    
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
        flux = (flux-np.median(flux))/np.median(flux)
        return cls(t, flux)
    
    def compute(self):
        self.subtract_trend(3)
        self.model, self.map_soln = self.build_model()
        self.trace = self.mcmc()
        self.mcmc_summary = pm.summary(self.trace, varnames=self.varnames)
        self.computed = True
    
    def plot(self):
        fig = plt.figure()
        plt.plot(self.t, self.flux, 'k.')
        return fig
    
    def plot_raw(self):
        fig = plt.figure()
        plt.plot(self.t, self.raw_flux, 'k.')
        return fig
    
    def plot_map_soln(self):
        if not self.computed:
            raise Exception("Must first call compute()")
        fig = plt.figure()
        plt.plot(self.t, self.flux, 'k.', alpha=0.2)
        plt.plot(self.t, self.map_soln["pred"])
        return fig
    
    def plot_corner(self):
        if not self.computed:
            raise Exception("Must first call compute()")
        samples = pm.trace_to_dataframe(self.trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2", "mean"])
        return corner.corner(samples)
    
    def get_trend(self, n):
        res = np.polyfit(self.t, self.flux, n)
        return sum([c*(self.t**i) for (i, c) in enumerate(res[::-1])])
    
    def subtract_trend(self, n):
        trend = self.get_trend(n)
        self.flux = self.flux - trend
        
    def autocor(self, max_peaks=1, min_period=0.5, max_period=100):
        results = xo.autocorr_estimator(self.t, self.flux, 
                                        max_peaks=max_peaks, 
                                        min_period=min_period, 
                                        max_period=max_period)
        lags, power = results["autocorr"]
        peaks = results["peaks"]
        return lags, power, peaks
    
    def plot_autocor(self, max_peaks=1, min_period=0.5, max_period=100):
        lags, power, peaks = self.autocor(max_peaks=max_peaks, min_period=min_period, max_period=max_period)
        fig = plt.figure()
        plt.plot(lags, power, "k")
        plt.axvline(peaks[0]["period"], color="k", lw=4, alpha=0.3)
        return fig
    
    def estimate_yerr(self, kernel_size=21, sigma=3):
        filt = medfilt(self.flux, kernel_size=kernel_size)
        return np.std(sigmaclip(self.flux-filt, low=sigma, high=sigma)[0])
    
    def build_model(self):
        lags, power, peaks = self.autocor()
        yerr = self.estimate_yerr()*np.ones(len(self.t))
        with pm.Model() as model:

            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            logs2 = pm.Normal("logs2", mu=2*np.log(yerr[0]), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux)), sd=5.0)
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=np.log(50))
            logperiod = BoundedNormal("logperiod", mu=np.log(peaks[0]["period"]), sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
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
            gp = xo.gp.GP(kernel, self.t, yerr**2 + tt.exp(logs2), J=4)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(self.flux - mean))

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())
            map_soln = xo.optimize(start=model.test_point)
            return model, map_soln
        
    def mcmc(self):
        sampler = xo.PyMC3Sampler(finish=200)
        with self.model:
            sampler.tune(tune=500, start=self.map_soln, step_kwargs=dict(target_accept=0.9))
            trace = sampler.sample(draws=500)
        return trace