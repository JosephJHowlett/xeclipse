from ELifetimeFitter import ELifetimeFitter, ELifetimeModeler
from slow_control.plot_from_mysql import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pdb

t_dump = 18.0
dump_dt = 0.15
#injected_o2 = 4*1e-9 # kg
t_gas_circ = 13.0
t_stop = 34.0
t_inject_1 = 42.0

t_dump_2 = 39.5
#t_dump = 200

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
                9.7*2
                * 1e-9 # ug/kg
                * 1e9 # 1 billion
                * 131 / 32 # LXe mm / O2 mm
                / 257 # ppb * us
                / self.GXe_density # kg / m^3
                )
            )
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
        liquid_gas_ratio = (self.LXe_density*self.V_PM_liquid)/(self.GXe_density*self.V_PM_gas)
        #migration_lg = (liquid_gas_ratio * n_l / p['tau_lg']) - (n_g / p['tau_gl'])
        #migration_gl = (n_g / liquid_gas_ratio / p['tau_gl']) - (n_l / p['tau_lg'])
        migration_gl = - (1.0/p['tau_m']) * ((p['alpha'] * n_l) - n_g)
        migration_lg = (liquid_gas_ratio / p['tau_m']) * ((p['alpha'] * n_l) - n_g)
        gas_circ = 0.0
        extra_term = 0.0
        time_dependence = 0.0
        if t > t_gas_circ:
            gas_circ  = 1.0
            #time_dependence = 1/(1+((t-t_gas_circ)/p['tau_lambda_g']))
        #if t > t_dump:
        #    extra_term = p['extra_term']
        if t > t_stop:
            gas_circ = 0.0
        dn_l_dt = (
            (p['lambda_0'] * (1/(1+((t-t_gas_circ)/p['tau_lambda_0']))) / self.V_PM_liquid)
            - (p['f'] * n_l / self.tau_l)
            + migration_gl
            + (extra_term / self.V_PM_liquid)
        )
        dn_g_dt = (
            (p['lambda_g'] * time_dependence / self.V_PM_gas)
            - (p['f_g'] * n_g / self.tau_g * gas_circ)
            + migration_lg
            + (injection / self.V_PM_gas)
        )
        RHSs = [dn_l_dt, dn_g_dt]
        return RHSs


p0 = OrderedDict(
    n_g_0 = {
        'guess': .25,
        'range': [0, 1e10],
        'uncertainty': 0.1,
        'latex_name': r'$n_{G,0}$',
        'unit': 'us^{-1}',
        },
    f = {
        'guess': .921,
        'range': [0, 1e10],
        'uncertainty': 0.5,
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
        'guess': 1.,
        'range': [0.5, 2.5],
        'uncertainty': 2,
        'latex_name': r'$f_g$',
        },
#    extra_term = {
#        'guess': 0.0001,
#        'range': [0, 1e10],
#        'uncertainty': 0.001,
#        'latex_name': r'$\Lambda_0$',
#        'unit': 'us^{-1}*l/h',
#        },
    lambda_0 = {
        'guess': 0.0028,
        'range': [0, 1e10],
        'uncertainty': 0.001,
        'latex_name': r'$\Lambda_0$',
        'unit': 'us^{-1}*l/h',
        },
    tau_lambda_0 = {
        'guess': 21.4,
        'range': [0, 1e10],
        'uncertainty': 365.0,
        'latex_name': r'$\tau_{\lambda_0}$',
        'unit': 'h',
        },
    lambda_g = {
        #'guess': 11.2,
        'guess': 0.0638,
        'range': [0, 1e10],
        'uncertainty': 1.5,
        'latex_name': r'$\Lambda_{G}$',
        'unit': 'us^{-1}*l/h',
        },
#    tau_lambda_g = {
#        'guess': 5000,
#        'range': [0, 1e10],
#        'uncertainty': 5000.0,
#        'latex_name': r'$\tau_{\lambda_G}$',
#        'unit': 'h',
#        },
    alpha = {
        'guess': 82.7,
        'range': [0, 1e10],
        'uncertainty': 200,
        'latex_name': r'$\alpha$',
        },
    tau_m = {
        'guess': 209.,
#        'guess': 1.0,
        'range': [0, 1e10],
        'uncertainty': 200,
        'latex_name': r'$\tau_M$',
        },
    )

start_time = "181018_000000"
stop_time = "181020"
#stop_time = "181019_180000"
#stop_time = "181019_153000"
#stop_time = "181019_100000"
#stop_time = "181018_170000"
omit_start = "181018_170000"
omit_stop = "181018_170000"
#omit_stop = "181019_000000"
name = 'prompt_injection_v02_open_181018_with_injection'
model_function_name = 'pm_model_prompt_injection'


guessing = False
filter_taus = False

start_from_dict = True
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
    odeint_kwargs={'tcrit': [t_dump]},
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


fitter.reset_walkers(90) 
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
        show=True,
        get_meds=True,
        t0=t0,
        verbose=False,
        odes_latex=fitter.odes_latex,
        text_pos='bottom',
        extrapolate_time_hrs=24,
    )
    fitter.plotter.plot_corner(show=False, range=0.9)


def plot_best_fit_twin(fitter, times, taus, initial_values, nb_iters=50, filename='best_fit.png', show=True, get_meds=True, t0=False, verbose=False, odes_latex=[], text_pos='top', **kwargs):
    ind = np.unravel_index(np.argmax(fitter.lnprobability, axis=None), fitter.lnprobability.shape)
    par_sets = []
    tot_iters = np.shape(fitter.chain)[1]
    if get_meds:
        # chain is (walkers, steps, pars)
        for walker in range(np.shape(fitter.chain)[0]):
            for step in range(nb_iters):
                if not np.isinf(fitter.lnprobability[walker][tot_iters-nb_iters+step]):
                    par_sets.append(fitter.chain[walker][tot_iters-nb_iters+step])
        par_sets = np.asarray(par_sets)
        par_meds = fitter.p_vector_to_dict(np.percentile(
            par_sets,
            50.0,
            axis=0))
        par_ups = fitter.p_vector_to_dict(np.percentile(
            par_sets,
            84.0,
            axis=0))
        par_lows = fitter.p_vector_to_dict(np.percentile(
            par_sets,
            16.0,
            axis=0))
    else:
        # use mode
        par_meds = fitter.p_vector_to_dict(fitter.chain[ind[0], ind[1], :])
        par_ups = par_meds
        par_lows = par_meds
    for key, val in par_meds.items():
        print('%s: %.2e + %.2e - %.2e' % (key, val, par_ups[key]-val, val-par_lows[key]))
    solve_times = times
    solve_times = np.linspace(times[0], times[-1] + kwargs.get('extrapolate_time_hrs', 0.0), len(times))
    sol, _ = fitter.solve_ODEs(solve_times, par_meds, initial_values, verbose=verbose)
    print('Log-Likelihood:')
    print(fitter.chi2_from_pars(par_meds.values(), times, taus, initial_values))
    fig = plt.figure(figsize=(10, 6))
    if t0:
        # convert back to epoch
        ax = fig.add_subplot(111)
        datetimes = dates.date2num([datetime.fromtimestamp((3600.0*time)+t0) for time in times])
        plot_datetimes = dates.date2num([datetime.fromtimestamp((3600.0*time)+t0) for time in solve_times])
        ax.plot_date(datetimes, taus, 'k.')
        ax.plot_date(plot_datetimes, 1.0/sol[:,0], 'r--', linewidth=2)
        date_format = dates.DateFormatter('%Y/%m/%d\n%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_ylabel('Electron Lifetime [us]')
        fig.autofmt_xdate()
        plt.xlabel('Time')
        ax2 = ax.twinx()
        ax2.plot_date(plot_datetimes, sol[:,0], 'b-', linewidth=2, label='liquid conc')


        ax2.plot_date(plot_datetimes, sol[:,1], 'g-', linewidth=2, label='gas conc')
        ax2.set_ylabel('O2 Concentration')
        ax2.set_yscale('log')

    plt.legend(loc='best')
    plt.savefig(fitter.name + '_' + 'gas_liquid_conc' + '_'  + filename)
    if show:
        plt.show()
    plt.close('all')
    return


def plot_marginalized_epsilon(fitter, nb_iters=200, filename='marginal_posterior.png', show=True):
    ind = np.unravel_index(np.argmax(fitter.lnprobability, axis=None), fitter.lnprobability.shape)
    par_sets = []
    tot_iters = np.shape(fitter.chain)[1]
    # chain is (walkers, steps, pars)
    for walker in range(np.shape(fitter.chain)[0]):
        for step in range(nb_iters):
            if not np.isinf(fitter.lnprobability[walker][tot_iters-nb_iters+step]):
                par_sets.append(fitter.chain[walker][tot_iters-nb_iters+step])
    par_sets = np.asarray(par_sets)
    name_arr = np.array([key for key in fitter.p0.keys()])
    alphas = par_sets[:, fitter.get_par_i('alpha')]
    f_gs = par_sets[:, fitter.get_par_i('f_g')]
    fs = par_sets[:, fitter.get_par_i('f')]
    tau_ms = par_sets[:, fitter.get_par_i('tau_m')]
    R = (fitter.V_PM_liquid*fitter.V_PM_gas / (fitter.V_CPS_liquid**2.0))
    tau_mlgs = (fitter.GXe_density*fitter.V_PM_gas) / (fitter.LXe_density*fitter.V_PM_liquid) * tau_ms
    tau_cl = 0.5
    print(alphas*R*tau_cl/fs)

    epsilon_primes = (R*alphas*tau_cl/fs) / ( (R*alphas*tau_cl/fs) + (fitter.tau_g / f_gs) + (tau_mlgs) )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, _ = plt.hist(epsilon_primes, histtype='step', bins=50, color='k')#, weights=[1.0/len(par_vals)]*len(par_vals))
    centers = (bins - 0.5*(bins[1]-bins[0]))[1:]
    for pc in (16, 50, 84):
        plt.axvline(x=np.percentile(epsilon_primes, pc), linestyle='--', color='k')
#    plt.text(0.1, 0.5, fitter.p0[par_name]['latex_name']+' = $%.2f^{+%.2f}_{-%.2f}$' % (
#            par_quantiles[1],
#            par_quantiles[2]-par_quantiles[1],
#            par_quantiles[1]-par_quantiles[0]
#        ),
#        fontsize=24,
#        verticalalignment='bottom', horizontalalignment='left',
#        transform=ax.transAxes,
#    )
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(fitter.name + '_' + filename)
    plt.show()

plot_best_fit_twin(
    fitter,
    times,
    taus,
    initial_values,
    show=True,
    get_meds=False,
    t0=t0,
    verbose=False,
    odes_latex=fitter.odes_latex,
    text_pos='bottom',
    extrapolate_time_hrs=24,
)
plot_marginalized_epsilon(fitter)
