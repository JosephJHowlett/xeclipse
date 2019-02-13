import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize
import scipy.stats
import matplotlib.pyplot as plt
import emcee
import pickle
import sys

from slow_control.plot_from_mysql import *

import corner

class MCMCPlotMaker(object):

    """Auxialliary object to make plots based on an ELifetimeFitter's chain

    This object holds the ELifetimeFitter as an attribute, and vice versa.
    To make plots from an instance of ELifetimeFitter (e.g. "fitter = ELifetimeFitter()"),
    do things like "fitter.plotter.plot_corner(**kwargs)".

    """

    def __init__(self, fitter):
        self.fitter = fitter

    def plot_corner(self, nb_iters=500, nb_samples=1000, filename='corner.png', show=True, range=None):
        # chain is (walkers, steps, pars)
        samples = self.fitter.chain[:,-nb_iters:,:].reshape(-1, self.fitter.chain.shape[-1])
        names = [self.fitter.p0[name].get('latex_name', name) for name in self.fitter.p0.keys()]
        if range:
            range=[range]*len(names)
        corner.corner(
            samples,
            labels=names,
            label_kwargs={'fontsize': 24},
            range=range,
            weights=[1.0]*len(samples),
            )
        plt.savefig(self.fitter.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_marginalized_posterior(self, par_name, nb_iters=200, filename='marginal_posterior.png', show=True):
        ind = np.unravel_index(np.argmax(self.fitter.lnprobability, axis=None), self.fitter.lnprobability.shape)
        par_sets = []
        tot_iters = np.shape(self.fitter.chain)[1]
        # chain is (walkers, steps, pars)
        for walker in range(np.shape(self.fitter.chain)[0]):
            for step in range(nb_iters):
                if not np.isinf(self.fitter.lnprobability[walker][tot_iters-nb_iters+step]):
                    par_sets.append(self.fitter.chain[walker][tot_iters-nb_iters+step])
        par_sets = np.asarray(par_sets)
        par_meds = self.fitter.p_vector_to_dict(np.percentile(
            par_sets,
            50.0,
            axis=0))
        par_ups = self.fitter.p_vector_to_dict(np.percentile(
            par_sets,
            84.0,
            axis=0))
        par_lows = self.fitter.p_vector_to_dict(np.percentile(
            par_sets,
            16.0,
            axis=0))
        par_quantiles = (par_lows[par_name], par_meds[par_name], par_ups[par_name])
        par_mode = self.fitter.p_vector_to_dict(self.fitter.chain[ind[0], ind[1], :])[par_name]
        par_vals = par_sets[:, self.fitter.get_par_i(par_name)]
        #par_vals = par_vals[np.where(par_vals<np.percentile(par_vals, 999.))[0]]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, _ = plt.hist(par_vals, histtype='step', bins=50, color='k')#, weights=[1.0/len(par_vals)]*len(par_vals))
        centers = (bins - 0.5*(bins[1]-bins[0]))[1:]
        def gaus(x, const, mean, sigma):
            return const/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mean)**2.0/(2.0*sigma**2.0))
        popt, popcov = curve_fit(gaus, centers, n, p0=[np.max(n), np.mean(n), np.std(n)])
        print(popt)
        plt.plot(centers, gaus(centers, *popt), 'r--')
        for par_quantile in par_quantiles:
            plt.axvline(x=par_quantile, linestyle='--', color='k')
        plt.axvline(x=par_mode, linestyle='-', color='k', label='Global Mode: %.2f' % par_mode)
        plt.text(0.1, 0.5, self.fitter.p0[par_name]['latex_name']+' = $%.2f^{+%.2f}_{-%.2f}$' % (
                par_quantiles[1],
                par_quantiles[2]-par_quantiles[1],
                par_quantiles[1]-par_quantiles[0]
            ),
            fontsize=24,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
        )
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(self.fitter.name + '_' + filename)
        plt.show()

    def plot_best_fit(self, times, taus, initial_values, nb_iters=50, filename='best_fit.png', show=True, get_meds=True, t0=False, verbose=False, odes_latex=[], text_pos='top'):
        ind = np.unravel_index(np.argmax(self.fitter.lnprobability, axis=None), self.fitter.lnprobability.shape)
        par_sets = []
        tot_iters = np.shape(self.fitter.chain)[1]
        if get_meds:
            # chain is (walkers, steps, pars)
            for walker in range(np.shape(self.fitter.chain)[0]):
                for step in range(nb_iters):
                    if not np.isinf(self.fitter.lnprobability[walker][tot_iters-nb_iters+step]):
                        par_sets.append(self.fitter.chain[walker][tot_iters-nb_iters+step])
            par_sets = np.asarray(par_sets)
            par_meds = self.fitter.p_vector_to_dict(np.percentile(
                par_sets,
                50.0,
                axis=0))
            par_ups = self.fitter.p_vector_to_dict(np.percentile(
                par_sets,
                84.0,
                axis=0))
            par_lows = self.fitter.p_vector_to_dict(np.percentile(
                par_sets,
                16.0,
                axis=0))
        else:
            # use mode
            par_meds = self.fitter.p_vector_to_dict(self.fitter.chain[ind[0], ind[1], :])
            par_ups = par_meds
            par_lows = par_meds
        for key, val in par_meds.items():
            print('%s: %.2e + %.2e - %.2e' % (key, val, par_ups[key]-val, val-par_lows[key]))
        sol, _ = self.fitter.solve_ODEs(times, taus, par_meds, initial_values, verbose=verbose)
        print('Log-Likelihood:')
        print(self.fitter.chi2_from_pars(par_meds.values(), times, taus, initial_values))
#        print(self.fitter.m_l/par_meds['eff_tau']/self.fitter.flow_l/self.fitter.LXe_density)
#        print(self.fitter.flow_l*self.fitter.LXe_density*(1.0/taus[-1])*par_meds['eff_tau']/self.fitter.m_l)
        fig = plt.figure(figsize=(10, 6))
        if t0:
            # convert back to epoch
            ax = fig.add_subplot(111)
            datetimes = dates.date2num([datetime.fromtimestamp((3600.0*time)+t0) for time in times])
            ax.plot_date(datetimes, taus, 'k.')
            ax.plot_date(datetimes, 1.0/sol[:,0], 'r--', linewidth=2)
            date_format = dates.DateFormatter('%Y/%m/%d\n%H:%M')
            ax.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
            plt.xlabel('Time')
        else:
            ax = fig.add_subplot(111)
            plt.plot(times, taus, 'k.')
            plt.plot(times, 1.0/sol[:, 0], 'r--', linewidth=2)
            plt.xlabel('Time [hours]')

        # print ODEs (model) if given
        if len(odes_latex):
            if text_pos=='top':
                for i, ode_string in enumerate(odes_latex):
                    plt.text(
                        0.01,
                        0.95 - (0.15*i),
                        ode_string,
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes,
                        fontsize=20, color='blue'
                    )
            else:
                for i, ode_string in enumerate(odes_latex):
                    plt.text(
                        0.05,
                        0.01 + (0.15*i),
                        ode_string,
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes,
                        fontsize=20, color='blue', fontweight='bold'
                    )

        # print parameter values
        par_string = r''
        for key, val in par_meds.items():
            par_string += self.fitter.p0[key].get('latex_name', '$%s$' % key)
            par_string += '$\ =\ $'
            par_string += '$%.2e\ %s$\n' % (val, self.fitter.p0[key].get('unit', ''))
        plt.text(0.5, 0.95, par_string,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=20, color='blue',
        )

        plt.ylabel('Electron Lifetime [us]')
        plt.savefig(self.fitter.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_lnprobability(self, filename='chi2.png', show=True):
        if np.all(np.isinf(self.fitter.lnprobability)):
            print('something wrong with init')
        fig = plt.figure()
        for i in range(self.fitter.nb_walkers):
            plt.plot(range(self.fitter.nb_steps), self.fitter.lnprobability[i])
        plt.savefig(self.fitter.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_burn_in(self, filename='burn_in', show=True):
        YTitles = self.fitter.p0.keys()
        NumColumn = 0
        if self.fitter.nb_dof % 2 == 0:
            NumColumn = int(self.fitter.nb_dof / 2)
        else:
            NumColumn = int(self.fitter.nb_dof / 2) + 1
        fig, Axes = plt.subplots(NumColumn, 2, sharex=True, sharey=False, figsize=(15, 8))
        # reshape axes
        axes1 = []
        axes2 = []
        for axes in Axes:
            for j, axis in enumerate(axes):
                if j == 0:
                    axes1.append(axis)
                else:
                    axes2.append(axis)
        axes = axes1 + axes2
        for i, (ytitle, ax) in enumerate(zip(YTitles, axes)):
            if (i == int(self.fitter.nb_dof / 2) - 1) or (i == len(axes)-1):
                ax.set_xlabel('step')
            for j in range(self.fitter.nb_walkers):
                ax.plot(range(self.fitter.nb_steps), self.fitter.chain[j,:,i], linewidth=0.3)
            # ax.plot(Iterators, SamplesForPlots[i][0], draw_opt)
            ax.set_ylabel(ytitle)
        plt.savefig(self.fitter.name + '_' + filename)
        if show:
            plt.show()
        plt.close()
        return
