from ELifetimeFitter import ELifetimeFitter, ELifetimeModeler
from slow_control.plot_from_mysql import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pdb

t_stop = 14.2

class lxepump_modeler(ELifetimeModeler):
    def __init__(self, **kwargs):
        super(lxepump_modeler, self).__init__(**kwargs)
        self.odes_latex = [
        ]
        return
    def lxepump_model(self, ns, t, p, verbose=False):
        """Test.
        """
        n_l = ns[0]
        n_lpb = ns[1]
        lxe_flow = self.flow_cl
        if t > t_stop:
            lxe_flow = 0
        dn_l_dt = (
            (p['lambda_0'] / self.V_liquid)
            - (p['f'] * n_l * self.flow_lg / self.V_liquid)
            - (p['f_c'] * n_l * lxe_flow / self.V_liquid)
            + ((1.0 - p['epsilon']) * (p['f_c']* n_lpb * lxe_flow / self.V_liquid))
        )
        dn_lpb_dt = (
            (p['lambda_lpb'] / self.V_liquid)
            + (p['f_c'] * n_l * lxe_flow / self.V_liquid)
            - (p['f_c'] * n_lpb * lxe_flow / self.V_liquid)
        )
        RHSs = [dn_l_dt, dn_lpb_dt]
        return RHSs


p0 = OrderedDict(
    n_lpb_0 = {
        'guess': 0.017,
        'range': [0, 1e10],
        'uncertainty': 0.05,
        'latex_name': r'$n_{LPB,0}$',
        'unit': 'us^{-1}',
        },
    f = {
        'guess': 0.937,
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
    f_c = {
        'guess': 0.937,
        'range': [0, 1e10],
        'uncertainty': 0.2,
        'latex_name': r'$f_c$',
        },
    lambda_0 = {
        'guess': 0.0008,
        'guess': 0.0,
        'range': [0, 1e10],
        'uncertainty': 0.001,
        'latex_name': r'$\Lambda_0$',
        'unit': 'us^{-1}*l/h',
        },
    lambda_lpb = {
        'guess': 0.0008,
#        'guess': 0.0012/24,
        'range': [0, 1e10],
        'uncertainty': 0.001,
        'latex_name': r'$\Lambda_{L,PB}$',
        'unit': 'us^{-1}*l/h',
        },
    epsilon = {
        'guess': 0.1,
#        'guess': 0.0012/24,
        'range': [0.0, 1.0],
        'uncertainty': 0.2,
        'latex_name': r'$\epsilon$',
        },
    )

start_time = "190130_120000"
stop_time = "190131_113000"
#stop_time = "190131_022000"
name = 'pump_stop_190131'
model_function_name = 'lxepump_model'


start_from_dict = True
plot_guess = False
filter_taus = False

fit = False
plot = True

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
nb_steps = 500
nb_dof = len(p0.values())
pickle_filename = os.path.join(output_dir, name + '.pkl')
if start_from_dict:
    start_from_dict = pickle_filename

slpm_per_llpm = 2942/5.894
# adjust initial
initial_nl = np.mean(1.0/np.asarray(taus)[:10])
p0['n_lpb_0']['guess'] = (initial_nl) + (p0['lambda_lpb']['guess'] * (0.5/p0['f_c']['guess']))

initial_values = [initial_nl, p0['n_lpb_0']['guess']]

fitter = lxepump_modeler(
    name=pickle_filename.split('.pkl')[0],
    nb_walkers=nb_walkers,
    p0=p0,
    function_to_explore='lnl',
    liquid_getter_flow_slpm=15.0, # slpm
    cryogenic_liquid_flow_lpm=0.5,
    gas_getter_flow_slph=0.0,  # can't have gas circulation while injecting
    start_from_dict=start_from_dict,
    model_function_name=model_function_name,
    times=times,
    taus=taus,
    initial_values=initial_values,
    fit_initial_values=[None, 'n_lpb_0'],
    odeint_kwargs={'tcrit': [t_stop]}
)

if plot_guess:
    par_array = [par['guess'] for par in p0.values()]
    print(fitter.chi2_from_pars(par_array, times, taus, initial_values))


    sol = fitter.solve_ODEs(
        fitter.times,
        fitter.taus,
        fitter.p_vector_to_dict(par_array),
        initial_values
    )[0]

    fig = plt.figure()
    plt.plot(times, taus, 'k.')
    plt.plot(times, 1.0/sol[:,0], 'b-', linewidth=2)
    plt.xlabel('Time [hours]')
    plt.ylabel('Lifetime [us]')
    #plt.ylim([0, 500])
    plt.show()


#fitter.reset_walkers(90) 
if fit:
    fitter.run_sampler(nb_steps)
    fitter.save_current_self(pickle_filename)


if plot:
    fitter.plotter.plot_lnprobability(show=False)
    fitter.plotter.plot_burn_in(show=False)
    fitter.plotter.plot_best_fit(times, taus, initial_values, show=True, get_meds=False, t0=t0, verbose=False, odes_latex=fitter.odes_latex, text_pos='bottom')
    fitter.plotter.plot_corner(show=False)
    #fitter.plotter.plot_marginalized_posterior('alpha')
