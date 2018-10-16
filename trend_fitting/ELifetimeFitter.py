import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import emcee
from collections import OrderedDict
from click import progressbar
import pickle
import sys

from slow_control.plot_from_mysql import *
import pdb
from operator import itemgetter

from pars import p0

import corner
import CoolProp.CoolProp as CP


class ELifetimeFitter(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'ELifetimeFit')
        self.flow_g = kwargs.get('gas_flow_slph', 10.0*60)  # SL/hour
        self.flow_l = kwargs.get(
            'liquid_flow_lph',
            (5.894/1000/2.942)*30.0*60
            )  # liters of liquid / hour
        #self.LXe_density = 2.942  # kg/liter
        #self.GXe_density = 5.894/1000  # kg/SL (depends on PT01)
        self.M_tot = kwargs.get('M_tot', 34.0)  # kg
        self.V_PM = kwargs.get('V_PM', 26.8)  # liters
        self.setup_thermodynamics()

        self.get_RHSs = kwargs.get('get_RHSs', self.standard_model)
        self.times = kwargs.get('times', None)
        self.taus = kwargs.get('taus', None)
        self.initial_values = kwargs.get('initial_values', None)

        self.fit_initial_values = np.asarray(
            kwargs.get('fit_initial_values', [None]*10))

        self.odeint_kwargs = kwargs.get('odeint_kwargs', {})

        self.name_to_function_map = {
            'chi2': self.chi2_from_pars,
            'lnl': self.lnl_from_pars,
            }

        # either start from a pickled dictionary
        if kwargs.get('start_from_dict', False):
            self.sampler_dict = self.load_self_dict(kwargs['start_from_dict'])
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.start_from_dict()
        # or start in a gaussian ball around p0
        else:
            self.nb_steps = 0
            self.nb_walkers = kwargs.get('nb_walkers', 80)
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.nb_dof = len(self.p0)
            self.setup_pos0_from_p0()

        # default to simple chi2
        self.function_to_explore = kwargs.get('function_to_explore', 'chi2')
        return

    def setup_thermodynamics(self, **kwargs):
        detector_pressure = 1.700  # bar
        standard_temperature = 273.15  # K
        standard_pressure = 101325  # Pa
        xenon_standard_density = 1e-3*CP.PropsSI('D', 'P', standard_pressure, 'T', standard_temperature, 'Xenon')  # kg/L
        self.LXe_density = 1e-3*CP.PropsSI('D', 'P', 1.e5*detector_pressure, 'Q', 0, 'Xenon')  # kg/L
        self.GXe_density = 1e-3*CP.PropsSI('D', 'P', 1.e5*detector_pressure, 'Q', 1, 'Xenon')  # kg/L
        self.V_gas = (((self.LXe_density * self.V_PM) - self.M_tot)
            / (self.LXe_density - self.GXe_density)
        )
        self.V_liquid = self.V_PM - self.V_gas
        print('liquid mass: %.2f kg ' % (self.LXe_density*self.V_liquid))
        print('gas mass: %.2f kg ' % (self.GXe_density*self.V_gas))
        return

    def setup_ranges(self):
        lower_ranges = np.zeros(len(self.p0.keys()))
        upper_ranges = np.zeros(len(self.p0.keys()))
        for i, par_name in enumerate(self.p0.keys()):
            lower_ranges[i] = self.p0[par_name]['range'][0]
            upper_ranges[i] = self.p0[par_name]['range'][1]
        return lower_ranges, upper_ranges

    def standard_model(self, ns, t, p, verbose=False):
        """The model itself.

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        n_g = ns[0]
        n_l = ns[1]
        outgassing_liquid = p['O_l']*np.exp(-t/p['tau_ol'])
        outgassing_gas = p['O_g']*np.exp(-t/p['tau_og'])
        migration_lg = (1.0/p['tau_mig'])*( (p['alpha']*n_l / (1.0 + ((p['alpha'] - 1.0)*n_l))) - n_g )
        migration_gl = (1.0/p['tau_mig'])*( (p['alpha']*n_g / (p['alpha'] - ((p['alpha'] - 1.0)*n_g))) - n_l )
        dn_g_dt = ( (-self.flow_g * self.GXe_density * n_g) + # SLPH*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
                    migration_gl +
                    outgassing_liquid
                    ) / self.m_l
        RHSs = [dn_l_dt, dn_g_dt]
        return RHSs

    def solve_ODEs(self, times, taus, p, initial_values, verbose=False):
        # set fit values to their pars
        if np.any(self.fit_initial_values):
            inds_to_fit = np.where(self.fit_initial_values)[0]
            initial_values[inds_to_fit] = itemgetter(*self.fit_initial_values[inds_to_fit])(p)
        sol, verb = odeint(self.get_RHSs, initial_values, times, args=(p, verbose), full_output=True, **self.odeint_kwargs)
        return sol, verb['message']

    def get_chi2(self, observation, expectation):
        chi2 = np.sum((observation - expectation)**2.0/(2*expectation**2.0))
        if np.isnan(chi2):
            print('ch2 is nan!')
            chi2=np.inf
        return chi2

    def get_lnl(self, observation, expectation, uncertainty):
        chi2 = np.sum((observation - expectation)**2.0/(2*uncertainty**2.0))
        if np.isnan(chi2):
            print('ch2 is nan!')
            chi2=np.inf
        normalization = np.sum(np.log(1.0/np.sqrt(2*np.pi)/uncertainty))
        return normalization - chi2

    def p_vector_to_dict(self, p_vector):
        p_dict = OrderedDict()
        for i, par_name in enumerate(self.p0.keys()):
            p_dict[par_name] = p_vector[i]
        return p_dict

    def get_par_i(self, par_name):
        for i, name in enumerate(self.p0.keys()):
            if par_name==name:
                return i
        return None

    def check_pars(self, p):
        return np.all(np.greater(p, self.lower_ranges)) and np.all(np.less(p, self.upper_ranges))

    def chi2_from_pars(self, p, times, taus, initial_values):
        """The chi-2 function for MCMC

        Takes in a list of parameter values, which needs to match
        the ordering of pars elsewhere.
        """
        if not self.check_pars(p):
            return -np.inf
        p = self.p_vector_to_dict(p)
        if np.any(np.asarray(p.values())<0):
            return -np.inf
        else:
            sol, message = self.solve_ODEs(times, taus, p, initial_values)
            if 'Excess work' in message:
                return -np.inf
            # calculate chi2 based on liquid impurities observed
#            chi2 = self.get_chi2(1.0/taus, sol[:, 0])
            chi2 = self.get_chi2(taus, 1.0/sol[:, 0])
            return -chi2

    def lnl_from_pars(self, p, times, taus, initial_values):
        """The log-likelihood function for MCMC

        Takes in a list of parameter values, which needs to match
        the ordering of pars elsewhere.
        """
        if not self.check_pars(p):
            return -np.inf
        p = self.p_vector_to_dict(p)
        if np.any(np.asarray(p.values())<0):
            return -np.inf
        else:
            sol, message = self.solve_ODEs(times, taus, p, initial_values)
            if 'Excess work' in message:
                return -np.inf
            # calculate chi2 based on lifetime observed
            return self.get_lnl(taus, 1.0/sol[:, 0], 0.15*taus)

    def run_sampler(self, nb_steps, fixed_args=None):
        if (fixed_args==None):
            fixed_args = (self.times, self.taus, self.initial_values)
        for fixed_arg in fixed_args:
            if not hasattr(fixed_arg, '__len__'):
                raise ValueError(
                    'You need to give the sampler (times, taus, initial_values)\
                    or give these as args when you instantiate the class.'
                )
        sampler = emcee.EnsembleSampler(
            self.nb_walkers,
            self.nb_dof,
            self.name_to_function_map[self.function_to_explore],
            args=fixed_args,
            )
        with progressbar(
                sampler.sample(p0=self.pos0, iterations=nb_steps),
                length=nb_steps
                )  as mcmc_sampler:
            for mcmc in mcmc_sampler:
                pass
        self.update_self_from_sampler(sampler)
        return

    def update_self_from_sampler(self, sampler):
        self.pos0 = sampler.chain[:,-1,:]
        if self.nb_steps:
            self.chain = np.concatenate((self.chain, sampler.chain), axis=1)
            self.lnprobability = np.concatenate((self.lnprobability, sampler.lnprobability), axis=1)
        else:
            self.chain = sampler.chain
            self.lnprobability = sampler.lnprobability
        self.nb_walkers = np.shape(sampler.chain)[0]
        self.nb_steps += np.shape(sampler.chain)[1]
        self.nb_dof = np.shape(sampler.chain)[2]
        return

    def save_current_self(self, output_name='temp.pkl'):
        output_dict = {}
        output_dict['chain'] = self.chain
        output_dict['lnprobability'] = self.lnprobability
        with open(output_name, 'wb') as f:
            pickle.dump(output_dict, f)

    def load_self_dict(self, infile):
        with open(infile, 'rb') as f:
            x = pickle.load(f)
        return x

    def start_from_dict(self):
        # setup pos0
        # chain is (walkers, steps, pars)
        sampler_dict = self.sampler_dict
        self.pos0 = sampler_dict['chain'][:,-1,:]
        self.chain = sampler_dict['chain']
        self.lnprobability = sampler_dict['lnprobability']
        self.nb_walkers = np.shape(sampler_dict['chain'])[0]
        self.nb_steps = np.shape(sampler_dict['chain'])[1]
        self.nb_dof = np.shape(sampler_dict['chain'])[2]
        return

    def setup_pos0_from_p0(self):
        # setup pos0
        self.pos0 = []
        for i in range(self.nb_walkers):
            walker_start = []
            for j, par in enumerate(self.p0.keys()):
                walker_start.append(np.random.normal(
                    loc=self.p0[par]['guess'],
                    scale=self.p0[par]['uncertainty'],
                    ))
            self.pos0.append(walker_start)
        return

    ##### Plotting #####
    def plot_corner(self, nb_iters=500, nb_samples=1000, filename='corner.png', show=True):
        # chain is (walkers, steps, pars)
        samples = self.chain[:,-nb_iters:,:].reshape(-1, self.chain.shape[-1])
        names = [self.p0[name].get('latex_name', name) for name in self.p0.keys()]
        corner.corner(
            samples,
            labels=names,
            label_kwargs={'fontsize': 24},
            range=[.9]*len(names),
            weights=[1.0]*len(samples),
            )
        plt.savefig(self.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_marginalized_posterior(self, par_name, nb_iters=200, filename='marginal_posterior.png', show=True):
        ind = np.unravel_index(np.argmax(self.lnprobability, axis=None), self.lnprobability.shape)
        par_sets = []
        tot_iters = np.shape(self.chain)[1]
        # chain is (walkers, steps, pars)
        for walker in range(np.shape(self.chain)[0]):
            for step in range(nb_iters):
                if not np.isinf(self.lnprobability[walker][tot_iters-nb_iters+step]):
                    par_sets.append(self.chain[walker][tot_iters-nb_iters+step])
        par_sets = np.asarray(par_sets)
        par_meds = self.p_vector_to_dict(np.percentile(
            par_sets,
            50.0,
            axis=0))
        par_ups = self.p_vector_to_dict(np.percentile(
            par_sets,
            84.0,
            axis=0))
        par_lows = self.p_vector_to_dict(np.percentile(
            par_sets,
            16.0,
            axis=0))
        par_quantiles = (par_lows[par_name], par_meds[par_name], par_ups[par_name])
        par_mode = self.p_vector_to_dict(self.chain[ind[0], ind[1], :])[par_name]
        par_vals = par_sets[:, self.get_par_i(par_name)]
        par_vals = par_vals[np.where(par_vals<np.percentile(par_vals, 99.))[0]]
        plt.figure()
        plt.hist(par_vals, histtype='step', bins=50, color='k', weights=[1.0/len(par_vals)]*len(par_vals))
        for par_quantile in par_quantiles:
            plt.axvline(x=par_quantile, linestyle='--', color='k')
        plt.axvline(x=par_mode, linestyle='-', color='k', label='Mode: %.2f' % par_mode)
        plt.text(2000, 0.1, self.p0[par_name]['latex_name']+' = $%.2f^{+%.2f}_{-%.2f}$' % (
            par_quantiles[1],
            par_quantiles[2]-par_quantiles[1],
            par_quantiles[1]-par_quantiles[0]
        ), fontsize=24)
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('alpha_dist.png')
        plt.show()

    def plot_best_fit(self, times, taus, initial_values, nb_iters=50, filename='best_fit.png', show=True, get_meds=True, t0=False, verbose=False, odes_latex=[], text_pos='top'):
        ind = np.unravel_index(np.argmax(self.lnprobability, axis=None), self.lnprobability.shape)
        par_sets = []
        tot_iters = np.shape(self.chain)[1]
        if get_meds:
            # chain is (walkers, steps, pars)
            for walker in range(np.shape(self.chain)[0]):
                for step in range(nb_iters):
                    if not np.isinf(self.lnprobability[walker][tot_iters-nb_iters+step]):
                        par_sets.append(self.chain[walker][tot_iters-nb_iters+step])
            par_sets = np.asarray(par_sets)
            par_meds = self.p_vector_to_dict(np.percentile(
                par_sets,
                50.0,
                axis=0))
            par_ups = self.p_vector_to_dict(np.percentile(
                par_sets,
                84.0,
                axis=0))
            par_lows = self.p_vector_to_dict(np.percentile(
                par_sets,
                16.0,
                axis=0))
        else:
            # use mode
            par_meds = self.p_vector_to_dict(self.chain[ind[0], ind[1], :])
            par_ups = par_meds
            par_lows = par_meds
        for key, val in par_meds.items():
            print('%s: %.2e + %.2e - %.2e' % (key, val, par_ups[key]-val, val-par_lows[key]))
        sol, _ = self.solve_ODEs(times, taus, par_meds, initial_values, verbose=verbose)
        print('Log-Likelihood:')
        print(self.chi2_from_pars(par_meds.values(), times, taus, initial_values))
#        print(self.m_l/par_meds['eff_tau']/self.flow_l/self.LXe_density)
#        print(self.flow_l*self.LXe_density*(1.0/taus[-1])*par_meds['eff_tau']/self.m_l)
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
            ode_string = r''
            for ode in odes_latex:
                ode_string += ode + '\n'
            if text_pos=='top':
                plt.text(
                    0.05,
                    0.95,
                    ode_string,
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes,
                    fontsize=20, color='blue', fontweight='bold'
                )
            else:
                plt.text(
                    0.05,
                    0.01,
                    ode_string,
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    fontsize=20, color='blue', fontweight='bold'
                )

        # print parameter values
        par_string = r''
        for key, val in par_meds.items():
            par_string += self.p0[key].get('latex_name', '$%s$' % key)
            par_string += '$\ =\ $'
            par_string += '$%.2e\ %s$\n' % (val, self.p0[key].get('unit', ''))
        plt.text(0.5, 0.95, par_string,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=20, color='blue',
        )

        plt.ylabel('Electron Lifetime [us]')
        plt.savefig(self.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_lnprobability(self, filename='chi2.png', show=True):
        if np.all(np.isinf(self.lnprobability)):
            print('something wrong with init')
        fig = plt.figure()
        for i in range(self.nb_walkers):
            plt.plot(range(self.nb_steps), self.lnprobability[i])
        plt.savefig(self.name + '_' + filename)
        if show:
            plt.show()
        plt.close('all')
        return

    def plot_burn_in(self, filename='burn_in', show=True):
        YTitles = self.p0.keys()
        NumColumn = 0
        if self.nb_dof % 2 == 0:
            NumColumn = int(self.nb_dof / 2)
        else:
            NumColumn = int(self.nb_dof / 2) + 1
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
            if (i == int(self.nb_dof / 2) - 1) or (i == len(axes)-1):
                ax.set_xlabel('step')
            for j in range(self.nb_walkers):
                ax.plot(range(self.nb_steps), self.chain[j,:,i], linewidth=0.3)
            # ax.plot(Iterators, SamplesForPlots[i][0], draw_opt)
            ax.set_ylabel(ytitle)
        plt.savefig(self.name + '_' + filename)
        if show:
            plt.show()
        plt.close()
        return

class ELifetimeModeler(ELifetimeFitter):

    """Subclass for playing with the model being fit to ELifetime data.

    Each function forms the argument to scipy's odeint, taking in a 
    set of x and y values (time and impurity concentration in GXe and LXe) 
    to numerically integrate via a model for their derivatives.

    Just write a function or choose one below, and when you initialize
    the class instance, feed a kwarg called `model_name` - a string of
    the function's name.

    """

    def __init__(self, **kwargs):
        super(ELifetimeModeler, self).__init__(**kwargs)
        self.get_RHSs = getattr(self, kwargs.get('model_name', 'standard_model'))
        return

    # Some basic models we may want to use
    def liquid_only_fixed_outgassing(self, ns, t, p, verbose=False):
        """Simplified (liquid-only, fixed outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        n_l = ns[0]
        outgassing_liquid = p['O_l']
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau']) + # liters/hour*kg/liter
                    outgassing_liquid
                    ) / self.m_l
        RHSs = [dn_l_dt]
        eff_tau = self.m_l/(self.flow_l*self.LXe_density*p['eff_tau'])
        if verbose:
            print(eff_tau)
        return RHSs

    def liquid_only_exp_outgassing(self, ns, t, p, verbose=False):
        """Simplified (liquid-only, exponentially-decreasing outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        n_l = ns[0]
        outgassing_liquid = p['O_l']
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau'] ) + # liters/hour*kg/liter
                    (outgassing_liquid * np.exp(-t/p['tau_ol']))
                    ) / self.m_l
        RHSs = [dn_l_dt]
        return RHSs

    def liquid_only_linear_outgassing(self, ns, t, p, verbose=False):
        """Simplified (liquid-only, linearly-decreasing outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        n_l = ns[0]
        outgassing_liquid = p['O_l']
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau'] ) + # liters/hour*kg/liter
                    (outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))
                    ) / self.m_l
        if verbose:
            print((outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))/self.m_l)
        eff_tau = self.m_l/(self.flow_l*self.LXe_density*p['eff_tau'])
        if verbose:
            print(eff_tau)
        RHSs = [dn_l_dt]
        return RHSs

    def liquid_gas_fixed_outgassing(self, ns, t, p, verbose=False):
        """Simplified (liquid and gas vol, fixed outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        n_l = ns[0]
        n_g = ns[1]
        outgassing_liquid = p['O_l']#*np.exp(-t/p['tau_ol'])
        outgassing_gas = p['O_g']#*np.exp(-t/p['tau_og'])
        migration_lg = (1.0/p['tau_mig'])*( (p['alpha']*n_l / (1.0 + ((p['alpha'] - 1.0)*n_l))) - n_g )
        migration_gl = (1.0/p['tau_mig'])*( (p['alpha']*n_g / (p['alpha'] - ((p['alpha'] - 1.0)*n_g))) - n_l )
        dn_g_dt = ( (-self.flow_g * self.GXe_density * n_g) + # SLPS*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
                    migration_gl +
                    outgassing_liquid
                    ) / self.m_l
        if verbose:
            print('Entering liquid [us^-1/hour]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dn_l_dt, dn_g_dt]
        return RHSs

    def liquid_wall(self, ns, t, p, verbose=False):
        n_l = ns[0]
        n_w = ns[1]
        dn_l_dt = ( (-self.flow_l * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
                    (n_w/(p['beta']*p['tau_wl'])) - (n_l/(p['tau_lw'])) # wall-liquid migration
                    ) / self.m_l
        dn_w_dt = (p['beta']*n_l/p['tau_lw']) - (n_w/p['tau_wl']) # liquid-wall migration
        if verbose:
            print('Entering liquid [us^-1/hour]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dn_l_dt, dn_w_dt]
        return RHSs


if __name__=='__main__':
    # Fit last 24 hours to nominal model
    # Note: it's typically better to make another program and instantiate
    # your own ELifetimeModeler to play around.

    t1 = time.time()
    t0 = t1 - 24.0

    lifetime = get_data_from_mysql('daq0', 'ElectronLifetime', t0, t1)
    times = np.array([row[0]  for row in lifetime])
    taus = np.array([row[1] for row in lifetime])

    times = times - times[0]

    print(p0)

    liquid_flow_lph = 0.5*60 # liters per hour

    nb_walkers = 200
    nb_steps = 50
    nb_dof = len(p0.keys())
    pickle_filename = os.path.join('saved_trends', 'test.pkl')
    fit = False
    new = False
    initial_values = [1.0/taus[0], p0['n_g_0']['guess']]
    if new:
        fitter = ELifetimeFitter(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            p0=p0,
            function_to_explore='chi2',
            liquid_flow_lph=liquid_flow_lph,
            )
    else:
        fitter = ELifetimeFitter(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            function_to_explore='chi2',
            start_from_dict=pickle_filename,
            p0=p0,
            liquid_flow_lph=liquid_flow_lph,
            )
    if fit:
        fitter.run_sampler(nb_steps, (times, taus, initial_values))
    print(np.shape(fitter.chain))
    fitter.save_current_self(pickle_filename)

    fitter.plot_best_fit(times, taus, initial_values, show=False)
    fitter.plot_lnprobability(show=False)
    fitter.plot_burn_in(show=False)
