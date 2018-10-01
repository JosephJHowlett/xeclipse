from ELifetimeFitter import ELifetimeFitter, ELifetimeModeler
from slow_control.plot_from_mysql import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pdb


class getter_bypass_modeler(ELifetimeModeler):
    def temperature_shift_model(self, ns, t, p, verbose=False):
        """takes in time where getter is bypassed and fits
        the full two-volume model to all the data."""
        t_change = 56.25
        n_l = ns[0]
        n_g = ns[1]
        outgassing_liquid = p['O_l']
        outgassing_gas = p['O_g'] * 1.0/(1.0 + (t/p['tau_og']))
        migration_lg = (p['alpha'] * n_l / p['tau_lg']) - (n_g / p['tau_gl'])
        migration_gl = (n_g / p['alpha'] / p['tau_gl']) - (n_l / p['tau_lg'])
        flow_l = self.flow_l
        flow_g = self.flow_g
        if t<t_change:
            flow_g = 0.0
        dn_g_dt = (
                    (-flow_g * self.GXe_density * n_g * p['eff_tau_g']) / self.m_g + # slph*kg/sl*us-1
                    migration_lg +
                    outgassing_gas / (self.m_l/self.LXe_density)
                    )
        dn_l_dt = ( (-flow_l * self.LXe_density * n_l * p['eff_tau']) / self.m_l + #liters/hour*kg/liter
                    migration_gl +
                    outgassing_liquid / (self.m_l/self.LXe_density)
                    ) 
        RHSs = [dn_l_dt, dn_g_dt]
        return RHSs


# for liquid_only_exp_outgassing model
p0 = OrderedDict(
    O_l = {
        'guess': 0.02/24, # Impurities/sec
        'range': [0, 1e10],
        'uncertainty': 0.2/24,
        'latex_name': r'$\Lambda_L$',
        },
    O_g = {
        'guess': 2.148/24, # Impurities/sec
        'range': [0, 1e10],
        'uncertainty': 2.148/24,
        'latex_name': r'$\Lambda_G$',
        },
    tau_og = {
        'guess': 1.45*24,
        'range': [-1e10, 1e10],
        'uncertainty': 5.0*24,
        'latex_name': r'$t_{1/2,\Lambda_G}$',
        },
    eff_tau = {
        'guess': 0.6907,
        'range': [0, 1e10],
        'uncertainty': 0.69,
        'latex_name': r'$f$',
        },
    eff_tau_g = {
        'guess': 0.9856,
        'range': [0, 1e10],
        'uncertainty': 1.0,
        'latex_name': r'$f_G$',
        },
    alpha = {
        'guess': 15.0,
        'range': [0, 1e10],
        'uncertainty': 40.0,
        'latex_name': r'$\alpha$',
        },
    tau_lg = {
        'guess': 3.7*24,
        'range': [0, 1e10],
        'uncertainty': 1.0,
        'latex_name': r'$tau_{LG}$',
        },
    tau_gl = {
        'guess': 0.0037*24,
        'range': [0, 1e10],
        'uncertainty': 0.03*24,
        'latex_name': r'$tau_{GL}$',
        },
    n_g_0 = {
        'guess': 0.0082,
        'range': [0, 1e10],
        'uncertainty': 0.09,
        'latex_name': r'$n_{G0}$',
        },
    )

start_time = "180505_060000"
stop_time = "180508_090000"
#stop_time = "180910040000"
name = 'temperature_shift_180505_model'
model = 'temperature_shift_model'


start_from_dict = False
plot_guess = False
filter_taus = False

fit = True
plot = True 

head_dir = '/Users/josephhowlett/research/xeclipse/analysis/tracked_analysis/'
output_dir = os.path.join(
    head_dir,
    'trend_fitting',
    'saved_trends',
    name
    )
if not start_from_dict:
    try:
        os.makedirs(output_dir)
    except OSError:
        print('Not making dir, dir exists - overwriting files within...')


t0 = datenum_to_epoch(start_time)
if stop_time=="now":
    t1 = time.time()
else:
    t1 = datenum_to_epoch(stop_time)

lifetime = get_data_from_mysql('daq0', 'ElectronLifetime', t0, t1)
times = np.array([row[0]  for row in lifetime])
taus = np.array([row[1] for row in lifetime])
t0 = times[0]
times = (times - t0)/3600.0
if filter_taus:
    kernel_size = 31
    buffer_remove = 5
    taus = medfilt(taus, kernel_size=kernel_size)[:-buffer_remove]
    times = times[:-buffer_remove]

print(p0)

nb_walkers = 200
nb_steps = 100
nb_dof = len(p0.values())
pickle_filename = os.path.join(output_dir, name + '.pkl')
if start_from_dict:
    start_from_dict = pickle_filename

slpm_per_llpm = 2942/5.894

fitter = getter_bypass_modeler(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            p0=p0,
            function_to_explore='chi2',
#            liquid_flow_lps=0.0,
            liquid_flow_lph=30.0/slpm_per_llpm*60, # liters per minute * minutes per hour
#            liquid_flow_lps=0.3/60, # liters per minute / seconds per minute
            gas_flow_slph=5.0*60,
            start_from_dict=start_from_dict,
            model_name=model,
            )

initial_values = [1.0/taus[0], p0['n_g_0']['guess']]
if plot_guess:
    par_array = [par['guess'] for par in p0.values()]
    print(fitter.chi2_from_pars(par_array, times, taus, initial_values))


    sol = fitter.solve_ODEs(times, taus, fitter.p_vector_to_dict(par_array), initial_values)[0]

    fig = plt.figure()
    plt.plot(times, taus, 'k.')
    plt.plot(times, 1.0/sol[:,0], 'b-', linewidth=2)
    plt.xlabel('Time [hours]')
    plt.ylabel('Lifetime [us]')
    plt.ylim([0, 2000])
    plt.show()


if fit:
    fitter.run_sampler(nb_steps, (times, taus, initial_values))
    fitter.save_current_self(pickle_filename)
if plot:
    fitter.plot_lnprobability(show=False)
    fitter.plot_burn_in(show=False)
    fitter.plot_best_fit(times, taus, initial_values, show=False, get_meds=False, t0=t0, verbose=False)
    fitter.plot_corner(show=False)




