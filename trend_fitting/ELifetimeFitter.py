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


class ELifetimeFitter(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'ELifetimeFit')
        self.flow_g = kwargs.get('gas_flow_slps', 10.0/60)  # SL/sec
        self.flow_l = kwargs.get(
            'liquid_flow_lps',
            (5.894/1000/2.942)*30.0/60
            )  # liters/sec
        self.LXe_density = 2.942  # kg/liter
        self.GXe_density = 5.894/1000  # kg/SL (depends on PT01)
        self.m_l = kwargs.get('m_l', 30.47)  # kg
        self.m_g = kwargs.get('m_g', .240)  # kg
        self.get_RHSs = kwargs.get('get_RHSs', self.standard_model)

        self.fit_initial_values = np.asarray(
            kwargs.get('fit_initial_values', [None]*10))

        self.odeint_kwargs = kwargs.get('odeint_kwargs', {})

        self.name_to_function_map = {
            'chi2': self.chi2_from_pars
            }

        # either start from a pickled dictionary
        if kwargs.get('start_from_dict', False):
            self.sampler_dict = self.load_self_dict(kwargs['start_from_dict'])
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.start_from_dict()
        # or start from a bunch of kwargs
        else:
            self.nb_steps = 0
            self.nb_walkers = kwargs.get('nb_walkers', 80)
            self.p0 = kwargs['p0']
            self.lower_ranges, self.upper_ranges = self.setup_ranges()
            self.nb_dof = len(self.p0)
            self.setup_pos0_from_p0()
        if kwargs.get('function_to_explore', False):
            self.function_to_explore = kwargs['function_to_explore']
        else:
            raise ValueError('Give me a function to use!')
        return

    def setup_ranges(self):
        lower_ranges = np.zeros(len(self.p0.keys()))
        upper_ranges = np.zeros(len(self.p0.keys()))
        for i, par_name in enumerate(self.p0.keys()):
            lower_ranges[i] = self.p0[par_name]['range'][0]
            upper_ranges[i] = self.p0[par_name]['range'][1]
        return lower_ranges, upper_ranges

    def standard_model(self, Is, t, p, verbose=False):
        """The model itself.

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        I_g = Is[0]
        I_l = Is[1]
        outgassing_liquid = p['O_l']*np.exp(-t/p['tau_ol'])
        outgassing_gas = p['O_g']*np.exp(-t/p['tau_og'])
        migration_lg = (1.0/p['tau_mig'])*( (p['alpha']*I_l / (1.0 + ((p['alpha'] - 1.0)*I_l))) - I_g )
        migration_gl = (1.0/p['tau_mig'])*( (p['alpha']*I_g / (p['alpha'] - ((p['alpha'] - 1.0)*I_g))) - I_l )
        dI_g_dt = ( (-self.flow_g * self.GXe_density * I_g) + # SLPS*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau']) + #liters/sec*kg/liter
                    migration_gl +
                    outgassing_liquid
                    ) / self.m_l
        RHSs = [dI_l_dt, dI_g_dt]
        return RHSs

    def solve_ODEs(self, times, taus, p, initial_values, verbose=False):
        # set fit values to their pars
        if np.any(self.fit_initial_values):
            inds_to_fit = np.where(self.fit_initial_values)[0]
            initial_values[inds_to_fit] = itemgetter(*self.fit_initial_values[inds_to_fit])(p)
        sol, verb = odeint(self.get_RHSs, initial_values, times, args=(p, verbose), full_output=True, **self.odeint_kwargs)
        return sol, verb['message']

    def get_chi2(self, observation, expectation):
        chi2 = np.sum((observation - expectation)**2.0/(expectation**2.0))
        if np.isnan(chi2):
            print('ch2 is nan!')
            chi2=np.inf
        return chi2

    def p_vector_to_dict(self, p_vector):
        p_dict = OrderedDict()
        for i, par_name in enumerate(self.p0.keys()):
            p_dict[par_name] = p_vector[i]
        return p_dict

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

    def run_sampler(self, nb_steps, fixed_args):
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
    def plot_corner(self, nb_iters=100, nb_samples=1000, filename='corner.png', show=True):
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

    def plot_best_fit(self, times, taus, initial_values, nb_iters=50, filename='best_fit.png', show=True, get_meds=True, t0=False, verbose=False):
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
        else:
            # use mode
            par_meds = self.p_vector_to_dict(self.chain[ind[0], ind[1], :])
        for key, val in par_meds.items():
            print('%s: %.2e' % (key, val))
        sol, _ = self.solve_ODEs(times, taus, par_meds, initial_values, verbose=verbose)
        print(self.chi2_from_pars(par_meds.values(), times, taus, initial_values))
#        print(self.m_l/par_meds['eff_tau']/self.flow_l/self.LXe_density)
#        print(self.flow_l*self.LXe_density*(1.0/taus[-1])*par_meds['eff_tau']/self.m_l)
        fig = plt.figure()
        if t0:
            ax = fig.add_subplot(111)
            datetimes = dates.date2num([datetime.fromtimestamp(time+t0) for time in times])
            ax.plot_date(datetimes, taus, 'k.')
            ax.plot_date(datetimes, 1.0/sol[:,0], 'r--', linewidth=2)
            date_format = dates.DateFormatter('%Y/%m/%d\n%H:%M')
            ax.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
            plt.xlabel('Time')
        else:
            plt.plot(times/3600.0, taus, 'k.')
            plt.plot(times/3600.0, 1.0/sol[:, 0], 'r--', linewidth=2)
            #plt.plot(times/3600.0, 1.0/taus, 'k.')
            #plt.plot(times/3600.0, sol[:, 0], 'k--', linewidth=2)
            #plt.text(8, 0.012, r'$\tau_{eff}\ =\ 2.7\ hours$'+'\n'+r'$\Lambda_{eff}\ =\ 0.13\ mg/day \times\ e^{-t/29\ hours} $', fontsize=20)
            #plt.ylim([0, .03])
            plt.xlabel('Time [hours]')
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
    def liquid_only_fixed_outgassing(self, Is, t, p, verbose=False):
        """Simplified (liquid-only, fixed outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        I_l = Is[0]
        outgassing_liquid = p['O_l']
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau']) + # liters/sec*kg/liter
                    outgassing_liquid
                    ) / self.m_l
        RHSs = [dI_l_dt]
        eff_tau = self.m_l/(self.flow_l*self.LXe_density*p['eff_tau'])/3600.0
        if verbose:
            print(eff_tau)
        return RHSs

    def liquid_only_exp_outgassing(self, Is, t, p, verbose=False):
        """Simplified (liquid-only, exponentially-decreasing outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        I_l = Is[0]
        outgassing_liquid = p['O_l']
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau'] ) + # liters/sec*kg/liter
                    (outgassing_liquid * np.exp(-t/p['tau_ol']))
                    ) / self.m_l
        RHSs = [dI_l_dt]
        return RHSs

    def liquid_only_linear_outgassing(self, Is, t, p, verbose=False):
        """Simplified (liquid-only, linearly-decreasing outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        I_l = Is[0]
        outgassing_liquid = p['O_l']
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau'] ) + # liters/sec*kg/liter
                    (outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))
                    ) / self.m_l
        if verbose:
            print((outgassing_liquid * 1.0/(1.0 + (t/p['tau_ol'])))/self.m_l)
        eff_tau = self.m_l/(self.flow_l*self.LXe_density*p['eff_tau'])/3600.0
        if verbose:
            print(eff_tau)
        RHSs = [dI_l_dt]
        return RHSs

    def liquid_gas_fixed_outgassing(self, Is, t, p, verbose=False):
        """Simplified (liquid and gas vol, fixed outgassing) model

        Takes in the list of impurity concentrations (1/lifetimes),
        The time since zero, for time-dependent pars (does this make sense?),
        and a dictionary of free parameters.
        """
        I_l = Is[0]
        I_g = Is[1]
        outgassing_liquid = p['O_l']#*np.exp(-t/p['tau_ol'])
        outgassing_gas = p['O_g']#*np.exp(-t/p['tau_og'])
        migration_lg = (1.0/p['tau_mig'])*( (p['alpha']*I_l / (1.0 + ((p['alpha'] - 1.0)*I_l))) - I_g )
        migration_gl = (1.0/p['tau_mig'])*( (p['alpha']*I_g / (p['alpha'] - ((p['alpha'] - 1.0)*I_g))) - I_l )
        dI_g_dt = ( (-self.flow_g * self.GXe_density * I_g) + # SLPS*kg/SL*us-1
                    migration_lg +
                    outgassing_gas
                    ) / self.m_g
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau']) + #liters/sec*kg/liter
                    migration_gl +
                    outgassing_liquid
                    ) / self.m_l
        if verbose:
            print('Entering liquid [us^-1/sec]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dI_l_dt, dI_g_dt]
        return RHSs

    def liquid_wall(self, Is, t, p, verbose=False):
        I_l = Is[0]
        I_w = Is[1]
        dI_l_dt = ( (-self.flow_l * self.LXe_density * I_l * p['eff_tau']) + #liters/sec*kg/liter
                    (I_w/(p['beta']*p['tau_wl'])) - (I_l/(p['tau_lw'])) # wall-liquid migration
                    ) / self.m_l
        dI_w_dt = (p['beta']*I_l/p['tau_lw']) - (I_w/p['tau_wl']) # liquid-wall migration
        if verbose:
            print('Entering liquid [us^-1/sec]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dI_l_dt, dI_w_dt]
        return RHSs


if __name__=='__main__':
    # Fit last 24 hours to nominal model
    # Note: it's typically better to make another code and instantiate
    # your own ELifetimeModeler to play around.

    t1 = time.time()
    t0 = t1 - 24.0*3600

    lifetime = get_data_from_mysql('daq0', 'ElectronLifetime', t0, t1)
    times = np.array([row[0]  for row in lifetime])
    taus = np.array([row[1] for row in lifetime])

    times = times - times[0]

    print(p0)

    liquid_flow_lps = 0.5/60. # liters per second

    nb_walkers = 200
    nb_steps = 50
    nb_dof = len(p0.keys())
    pickle_filename = os.path.join('saved_trends', 'test.pkl')
    fit = False
    new = False
    initial_values = [1.0/taus[0], p0['I_g_0']['guess']]
    if new:
        fitter = ELifetimeFitter(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            p0=p0,
            function_to_explore='chi2',
            liquid_flow_lps=liquid_flow_lps,
            )
    else:
        fitter = ELifetimeFitter(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            function_to_explore='chi2',
            start_from_dict=pickle_filename,
            p0=p0,
            liquid_flow_lps=liquid_flow_lps,
            )
    if fit:
        fitter.run_sampler(nb_steps, (times, taus, initial_values))
    print(np.shape(fitter.chain))
    fitter.save_current_self(pickle_filename)

    fitter.plot_best_fit(times, taus, initial_values, show=False)
    fitter.plot_lnprobability(show=False)
    fitter.plot_burn_in(show=False)
