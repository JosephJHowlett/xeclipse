from ELifetimeFitter import ELifetimeFitter, ELifetimeModeler
from slow_control.plot_from_mysql import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pdb


class getter_flow_model(ELifetimeModeler):
    def get_flow_from_time(self, time):
        result = self.liquid_flows[np.logical_and((time>self.run_starts), (time<self.run_stops))]
        if len(result):
            return result[0]
        return 0.0

    def liquid_wall_bypass(self, Is, t, p, verbose=False):
        I_l = Is[0]
        I_w = Is[1]
        flow_l = self.get_flow_from_time(t)
        dI_l_dt = ( ((-flow_l * self.LXe_density * I_l * p['eff_tau'])/ self.m_l) + #liters/sec*kg/liter
                    (I_w/(p['beta']*p['tau_wl'])) - (I_l/(p['tau_lw'])) # wall-liquid migration
                    ) 
        factor = 1.0
        if not flow_l:
            dI_l_dt += p['bypass_impurities']/(self.m_l/self.LXe_density)
        dI_w_dt = (p['beta']*I_l/p['tau_lw']) - (I_w/p['tau_wl']) # liquid-wall migration
        if verbose:
            print('Entering liquid [us^-1/sec]:')
            print((migration_gl + outgassing_liquid)/self.m_l)
        RHSs = [dI_l_dt, dI_w_dt]
        return RHSs

# for liquid_only_exp_outgassing model
p0 = OrderedDict(
    eff_tau = {
        'guess': .58,
        'range': [0, 1e10],
        'uncertainty': 0.2,
        },
    beta = {
        'guess': 0.6542,
        'range': [0.0, 1e10],
        'uncertainty': 0.5,
        },
    tau_wl = {
        'guess': 3.1445*3600*24,
        'range': [0.0, 1e10],
        'uncertainty': 1.0*3600*24,
        },
    tau_lw = {
        'guess': 0.778*3600*24,
        'range': [0.0, 1e10],
        'uncertainty': 1.0*3600*24,
        },
    bypass_impurities = {
        'guess': 0.65/3600/24, # us-1*L/sec
        'range': [0.0, 1e10],
        'uncertainty': 1.0/3600/24,
        },
    I_w_0 = {
        'guess': 0.0049,
        'range': [0.0, 1e10],
        'uncertainty': 0.005,
        },
    )

def parse_logbook_file(filename):
    run_starts = []
    run_stops = []
    liquid_flows = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        arr = line.split()
        run_starts.append(arr[0])
        run_stops.append(arr[1])
        liquid_flows.append(arr[2])
    return run_starts, run_stops, liquid_flows

start_from_dict = True
plot_guess = False
filter_taus = True

fit = False
plot = True
name = 'testing_wall_model'
model = 'liquid_wall_bypass'

head_dir = '/Users/josephhowlett/research/xeclipse/'
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

logbook_filename = os.path.join(
    head_dir,
    'getter_tests',
    'logbook_getter_info.dat'
    )
run_starts, run_stops, liquid_flows = parse_logbook_file(logbook_filename)

t0 = datenum_to_epoch(run_starts[0])
t1 = datenum_to_epoch(run_stops[-3])

lifetime = get_data_from_mysql('daq0', 'ElectronLifetime', t0, t1)
times = np.array([row[0]  for row in lifetime])
taus = np.array([row[1] for row in lifetime])
t0 = times[0]
times = times - t0

print(run_starts)
run_starts = np.asarray([datenum_to_epoch(run) for run in run_starts]) - t0
run_stops = np.asarray([datenum_to_epoch(run) for run in run_stops]) - t0

if filter_taus:
    kernel_size = 31
    buffer_remove = 5
    taus = medfilt(taus, kernel_size=kernel_size)[:-buffer_remove]
    times = times[:-buffer_remove]

print(p0)

nb_walkers = 200
nb_steps = 300
nb_dof = len(p0.values())
pickle_filename = os.path.join(output_dir, name + '.pkl')
if start_from_dict:
    start_from_dict = pickle_filename

slpm_per_llpm = 2942/5.894

fitter = getter_flow_model(
            name=pickle_filename.split('.pkl')[0],
            nb_walkers=nb_walkers,
            p0=p0,
            function_to_explore='chi2',
#            liquid_flow_lps=0.0,
            liquid_flow_lps=30.0/slpm_per_llpm/60, # liters per minute / seconds per minute
#            liquid_flow_lps=0.3/60, # liters per minute / seconds per minute
            gas_flow_slps=5.0/60,
            start_from_dict=start_from_dict,
            model_name=model,
            fit_initial_values = [None, 'I_w_0'] # tell it what initial conditions to fit
            )
fitter.run_starts = run_starts
fitter.run_stops = run_stops
fitter.liquid_flows = np.asarray(liquid_flows, dtype=int)/slpm_per_llpm/60.0
initial_values = [1.0/taus[0], 0.0]  # initial wall input doesn't matter

if plot_guess:
    par_array = [par['guess'] for par in p0.values()]
    print(fitter.chi2_from_pars(par_array, times, taus, initial_values))

    sol = fitter.solve_ODEs(times, taus, fitter.p_vector_to_dict(par_array), initial_values)[0]

    fig = plt.figure()
    plt.plot(times/3600.0, taus, 'k.')
    plt.plot(times/3600.0, 1.0/sol[:,0], 'b-', linewidth=2)
    plt.xlabel('Time [hours]')
    plt.ylabel('Lifetime [us]')
    plt.ylim([0, 5000])
    plt.show()


if fit:
    fitter.run_sampler(nb_steps, (times, taus, initial_values))
    fitter.save_current_self(pickle_filename)
if plot:
    #fitter.plot_lnprobability(show=False)
    #fitter.plot_burn_in(show=False)
    fitter.plot_best_fit(times, taus, initial_values, show=True, get_meds=False, t0=t0, verbose=False)
    fitter.plot_corner(show=False)




