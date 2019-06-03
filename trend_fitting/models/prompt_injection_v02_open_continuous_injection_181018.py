from ELifetimeFitter import ELifetimeFitter, ELifetimeModeler
from slow_control.plot_from_mysql import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pdb

t_dump = 18.0
dump_dt = 0.5
#injected_o2 = 4*1e-9 # kg
t_gas_circ = 13.0
t_stop = 34.0
t_dump_2 = 39.5
t_inject_1 = 42.0

def gaussian(x, mu, sigma, amp):
    return amp/sigma/np.sqrt(2*np.pi)*np.exp(-1*(x-mu)**2.0/(2*(sigma**2.0)))

class pm_modeler(ELifetimeModeler):
    def __init__(self, **kwargs):
        super(pm_modeler, self).__init__(**kwargs)
        self.odes_latex = [
        ]
        return

    def pm_model_prompt_injection(self, ns, t, p, verbose=False):
        """Test.
        """
        n_l = ns[0]
        n_g = ns[1]
        injection = 0.0
        if t > t_dump:
            injection = gaussian(
                t,
                t_dump,
                dump_dt,
                (
                9.7 * 2 # 2 for asym gaus
                * 1e-9 # ug/kg
                * 1e9 # 1 billion
                * 131 / 32
                / 257
                / self.GXe_density
                )
            )
        liquid_gas_ratio = (self.LXe_density*self.V_PM_liquid)/(self.GXe_density*self.V_PM_gas)
        migration_lg = (liquid_gas_ratio * n_l / p['tau_lg']) - (n_g / p['tau_gl'])
        migration_gl = (n_g / liquid_gas_ratio / p['tau_gl']) - (n_l / p['tau_lg'])
        gas_circ = 0.0
        time_dependence = 0.0
        if t > t_gas_circ:
            gas_circ  = 1.0
            time_dependence = 1/(1+((t-t_gas_circ)/p['tau_lambda_g']))
        if t > t_stop:
            gas_circ = 0.0
        if t > t_dump_2:
            injection = gaussian(
                t,
                t_dump_2,
                dump_dt,
                (
                9.7 * 2 # 2 for asym gaus
                * 1e-9 # ug/kg
                * 1e9 # 1 billion
                * 131 / 32
                / 257
                / self.GXe_density
                )
            )
        if t > t_inject_1:
            injection = (
                50 * 3.3 / 24
                * 1e-9 # ug/kg
                * 1e9 # 1 billion
                * 131 / 32
                / 257
                / self.GXe_density
                )
        dn_l_dt = (
            (p['lambda_0'] * (1/(1+((t-t_gas_circ)/p['tau_lambda_0']))) / self.V_PM_liquid)
            - (p['f'] * n_l / self.tau_l)
            + migration_gl
            + (injection * p['condensed_fraction'] / self.V_PM_liquid)
        )
        dn_g_dt = (
            (p['lambda_g'] * time_dependence / self.V_PM_gas)
            - (p['f_g'] * n_g / self.tau_g * gas_circ)
            + migration_lg
            + (injection * (1 - p['condensed_fraction']) / self.V_PM_gas)
        )
        RHSs = [dn_l_dt, dn_g_dt]
        return RHSs


p0 = OrderedDict(
    n_g_0 = {
        'guess': .9,
        'range': [0, 1e10],
        'uncertainty': 0.2,
        'latex_name': r'$n_{G,0}$',
        'unit': 'us^{-1}',
        },
    f = {
        'guess': 1.47,
        'range': [0, 1e10],
        'uncertainty': 2,
        'latex_name': r'$f$',
        'prior': {
            'type': 'norm',
            'args': dict(
                loc=0.937,
                scale=0.05,
                ),
            },
        },
    f_g = {
        #'guess': 2.41,
        'guess': 0.5,
        'range': [0.9, 2.5],
        'uncertainty': 3,
        'latex_name': r'$f_g$',
        },
    lambda_0 = {
        'guess': 0.00402,
        'range': [0, 1e10],
        'uncertainty': 0.005,
        'latex_name': r'$\Lambda_0$',
        'unit': 'us^{-1}*l/h',
        },
    tau_lambda_0 = {
        'guess': 19.9,
        'range': [0, 1e10],
        'uncertainty': 100.0,
        'latex_name': r'$\tau_{\lambda_G}$',
        'unit': 'h',
        },
    lambda_g = {
        #'guess': 11.2,
        'guess': 0.5,
        'range': [0, 1e10],
        'uncertainty': 1.5,
        'latex_name': r'$\Lambda_{G}$',
        'unit': 'us^{-1}*l/h',
        },
    tau_lambda_g = {
        'guess': 9.11,
        'range': [0, 1e10],
        'uncertainty': 100.0,
        'latex_name': r'$\tau_{\lambda_G}$',
        'unit': 'h',
        },
    tau_lg = {
        'guess': 1.9,
        'range': [0, 1e10],
        'uncertainty': 100.0,
        'latex_name': r'$\tau_{LG}$',
        'unit': 'h',
        },
    tau_gl = {
        #'guess': 0.0037*24,
        'guess': 2.45,
        'range': [0, 1e10],
        'uncertainty': 100.0,
        'latex_name': r'$\tau_{GL}$',
        'unit': 'h',
        },
    condensed_fraction = {
        #'guess': 0.0037*24,
        'guess': 0.01,
        'range': [0, 1.0],
        'uncertainty': 1.0,
        'latex_name': r'$m_{inj}$',
        'unit': 'ug',
        },
    )

start_time = "181018_000000"
stop_time = "181020_000000"
#stop_time = "181019_100000"
#stop_time = "181018_170000"
omit_start = "181018_170000"
omit_stop = "181018_170000"
#omit_stop = "181019_000000"
name = 'prompt_injection_v02_open_continuous_injection_181018_python3'
model_function_name = 'pm_model_prompt_injection'


start_from_dict = False
guessing = False
filter_taus = False

fit = True
plot = True

if guessing:
    fit = False
    plot = False

head_dir = '/Users/josephhowlett/research/xeclipse/tracked_analysis/'
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
omit_start = datenum_to_epoch(omit_start)
omit_stop = datenum_to_epoch(omit_stop)


lifetime = get_data_from_mysql('daq0', 'ElectronLifetime', t0, omit_start)
lifetime2 = get_data_from_mysql('daq0', 'ElectronLifetime', omit_stop, t1)
times = np.array([row[0]  for row in lifetime] + [row[0]  for row in lifetime2])
taus = np.array([row[1] for row in lifetime] + [row[1] for row in lifetime2])

t0 = times[0]
times = (times - t0)/3600.0
if filter_taus:
    kernel_size = 31
    buffer_remove = 5
    taus = medfilt(taus, kernel_size=kernel_size)[:-buffer_remove]
    times = times[:-buffer_remove]

print(p0)

nb_walkers = 50
nb_steps = 500
nb_dof = len(p0.values())
pickle_filename = os.path.join(output_dir, name + '.pkl')
if start_from_dict:
    start_from_dict = pickle_filename

slpm_per_llpm = 2942/5.894
# adjust initial
initial_nl = np.mean(1.0/np.asarray(taus)[:10])
#p0['n_g_0']['guess'] = (initial_nl) + (p0['lambda_g']['guess'] * (0.5/p0['f_g']['guess']))

initial_values = [initial_nl, p0['n_g_0']['guess']]

fitter = pm_modeler(
    name=pickle_filename.split('.pkl')[0],
    nb_walkers=nb_walkers,
    p0=p0,
    function_to_explore='lnl',
    liquid_getter_flow_slpm=30.0, # slpm
    DPT31=0.2, # mbar
    gas_getter_flow_slph=5.0*60,  # can't have gas circulation while injecting
    start_from_dict=start_from_dict,
    model_function_name=model_function_name,
    times=times,
    taus=taus,
    initial_values=initial_values,
    fit_initial_values=[None, 'n_g_0'],
    odeint_kwargs={'tcrit': []},
)

if guessing:
    par_array = [par['guess'] for par in p0.values()]
    print(fitter.chi2_from_pars(par_array, times, taus, initial_values))

    plot_times = np.linspace(0, times[-1]+24, 200)
    sol = fitter.solve_ODEs(
        plot_times,
        fitter.p_vector_to_dict(par_array),
        initial_values
    )[0]

    fig = plt.figure()
    plt.plot(times, taus, 'k.')
    plt.plot(plot_times, 1.0/sol[:,0], 'b-', linewidth=2)
    plt.xlabel('Time [hours]')
    plt.ylabel('Lifetime [us]')
    plt.show()


#fitter.reset_walkers(90) 
if fit:
    fitter.run_sampler(nb_steps)
    fitter.save_current_self(pickle_filename)


if plot:
    fitter.plotter.plot_lnprobability(show=False)
    fitter.plotter.plot_burn_in(show=False)
    fitter.plotter.plot_best_fit(
        times,
        taus,
        initial_values,
        show=False,
        get_meds=False,
        t0=t0,
        verbose=False,
        odes_latex=fitter.odes_latex,
        text_pos='bottom',
        extrapolate_time_hrs=24,
    )
    #fitter.plotter.plot_corner(show=False, range=0.9)
    #fitter.plotter.plot_marginalized_posterior('alpha')
