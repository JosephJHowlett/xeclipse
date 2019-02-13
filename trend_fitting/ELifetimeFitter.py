import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize
import scipy.stats
import matplotlib.pyplot as plt
import emcee
from collections import OrderedDict
from click import progressbar
import pickle
import sys

from slow_control.plot_from_mysql import *
import pdb
from operator import itemgetter

import corner
import CoolProp.CoolProp as CP

import MCMCPlotMaker


class ELifetimeFitter(object):
    def __init__(self, **kwargs):
        slpm_per_llpm = 2942/5.894
        self.flow_gg = kwargs.get('gas_getter_flow_slph', 10.0*60)  # SL/hour
        self.flow_lg = kwargs.get(
            'liquid_getter_flow_slpm',
            30.0
            )/slpm_per_llpm*60  # liters of liquid / hour
        self.flow_cl = kwargs.get(
            'cryogenic_liquid_flow_lpm',
            0.5
            )*60  # liters of liquid / hour
        self.odeint_kwargs = kwargs.get('odeint_kwargs', {})
        self.times = kwargs.get('times', None)
        self.taus = kwargs.get('taus', None)
        self.initial_values = kwargs.get('initial_values', None)
        self.fit_initial_values = np.asarray(
            kwargs.get('fit_initial_values', [None]*10))


        self.name = kwargs.get('name', 'ELifetimeFit')
        self.M_tot = kwargs.get('M_tot', 34.0)  # kg
        self.V_PM = kwargs.get('V_PM', 26.8)  # liters
        self.setup_thermodynamics()

        # set the differential equation function
        self.get_RHSs = self.standard_model

        self.name_to_function_map = {
            'chi2': self.chi2_from_pars,
            'lnl': self.lnl_from_pars,
            }

        # either start from the last step from a pickled dictionary
        if kwargs.get('start_from_dict', False):
            self.sampler_dict = self.load_self_dict(kwargs['start_from_dict'])
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.start_from_dict()
        # or start in a gaussian ball around kwarg p0
        else:
            self.nb_steps = 0
            self.nb_walkers = kwargs.get('nb_walkers', 80)
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.nb_dof = len(self.p0)
            self.setup_pos0_from_p0()

        # default to lnl
        self.function_to_explore = kwargs.get('function_to_explore', 'lnl')

        # setup plotmaker
        self.plotter = MCMCPlotMaker.MCMCPlotMaker(self)
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
        dn_g_dt = ( (-self.flow_gg * self.GXe_density * n_g) + # SLPH*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
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
        normalization = np.sum(-1*np.log(np.sqrt(2*np.pi)) - np.log(uncertainty))
        if np.isnan(normalization):
            print('norm is nan!')
            normalization = -np.inf
        lnl = normalization - chi2
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
            lnl = self.get_lnl(taus, 1.0/sol[:, 0], (0.15*taus)+10.0)
        if np.isnan(lnl):
            return -np.inf
        lnl += self.priors_from_pars(p)
        return lnl

    def priors_from_pars(self, p):
        logprior = 0.0
        for key, val in p.items():
            if self.p0[key].get('prior', None):
                distribution = getattr(scipy.stats, self.p0[key]['prior']['type'])
                logprior += distribution.logpdf(val, **self.p0[key]['prior']['args'])
        return logprior

    def reset_walkers(self, perc=99., nb_iters=500):
        """Update last position of all walkers outside a given percentile"""
        # get all means
        tot_iters = np.shape(self.chain)[1]
        par_sets = []
        for walker in range(np.shape(self.chain)[0]):
            for step in range(nb_iters):
                if not np.isinf(self.lnprobability[walker][tot_iters-nb_iters+step]):
                    par_sets.append(self.chain[walker][tot_iters-nb_iters+step])
        par_sets = np.asarray(par_sets)
        par_val_cutoffs_high = np.percentile(par_sets, perc, axis=0)
        par_val_cutoffs_low = np.percentile(par_sets, 100 - perc, axis=0)
        # loop over walkers
        for walker in range(np.shape(self.chain)[0]):
            par_vals = self.chain[walker][-1]
            inds_above = np.where(par_vals > par_val_cutoffs_high)[0]
            for ind in inds_above:
                self.chain[walker][-1][ind] = par_val_cutoffs_high[ind]
            inds_below = np.where(par_vals < par_val_cutoffs_low)[0]
            for ind in inds_below:
                self.chain[walker][-1][ind] = par_val_cutoffs_low[ind]
        self.pos0 = self.chain[:,-1,:]
        print('Reset walkers outside [%f, %f] percentile.' % (perc, 1-perc))

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


class ELifetimeModeler(ELifetimeFitter):

    """Subclass for playing with the model being fit to ELifetime data.

    Each function forms the argument to scipy's odeint, taking in a 
    set of x and y values (time and impurity concentration in GXe and LXe) 
    to numerically integrate via a model for their derivatives.

    Just write a function or choose one below, and when you initialize
    the class instance, feed a kwarg called `model_function_name` - a string of
    the function's name.

    """

    def __init__(self, **kwargs):
        # initialize as the super-class first
        super(ELifetimeModeler, self).__init__(**kwargs)
        # overwrite the model with custom model
        self.get_RHSs = getattr(self, kwargs.get('model_function_name', 'standard_model'))
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
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau']) + # liters/hour*kg/liter
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
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau'] ) + # liters/hour*kg/liter
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
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau'] ) + # liters/hour*kg/liter
                    (outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))
                    ) / self.m_l
        if verbose:
            print((outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))/self.m_l)
        eff_tau = self.m_l/(self.flow_lg*self.LXe_density*p['eff_tau'])
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
        dn_g_dt = ( (-self.flow_gg * self.GXe_density * n_g) + # SLPS*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
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
        dn_l_dt = ( (-self.flow_lg * self.LXe_density * n_l * p['eff_tau']) + #liters/hour*kg/liter
                    (n_w/(p['beta']*p['tau_wl'])) - (n_l/(p['tau_lw'])) # wall-liquid migration
                    ) / self.m_l
        dn_w_dt = (p['beta']*n_l/p['tau_lw']) - (n_w/p['tau_wl']) # liquid-wall migration
        if verbose:
            print('Entering liquid [us^-1/hour]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dn_l_dt, dn_w_dt]
        return RHSs

class MultipleModeler(ELifetimeModeler):
    """Under Construction..."""
    def __init__(self, **kwargs):
        # initialize as the super-class first
        super(MultipleModeler, self).__init__(**kwargs)
        # check if simultaneously fitting multiple ranges
        self.nb_time_ranges = kwargs.get('nb_time_ranges', 1)
        if self.nb_time_ranges < 2:
            raise ValueError(
                'Please specify nb_time_ranges > 2 or use a different class'
            )
        # make sure above things have the appropriate dimension
        for range_specific_attr in [
            'flow_g',
            'flow_l',
            'odeint_kwargs',
            'times',
            'taus',
            'initial_values',
            'fit_initial_values',
            'model_function_name',
        ]:
            attr_value = getattr(self, range_specific_attr)
            if len(attr_value) != self.nb_time_ranges:
                print('Copying %s for multiple ranges' % range_specific_attr)
                setattr(
                    self,
                    range_specific_attr,
                    [attr_value]*self.nb_time_ranges
                )
        return

    def lnl_from_pars(self, p, times, taus, initial_values):
        """The log-likelihood function for MCMC

        Takes in a list of parameter values, which needs to match
        the ordering of pars elsewhere.
        """
        total_lnl = 0.0
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
