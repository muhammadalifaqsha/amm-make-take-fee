import numpy as np
import utils as utils_aux
nsims = 100

colors = ['tab:red', 'tab:orange', 'skyblue', 'tab:green', 'tab:blue','black']

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import matplotlib
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')
dpi = 300
fs = 14
fsL = 19
plt.rcParams["font.size"] = fs



def stress_two_parameters_two_sets_and_show_nu(environment = utils_aux.environment, parameter1 = 'fee', parameter2 = 'xi', values1 = [1,2,4,8,16], values2 = [1,2,4,8,16], 
                                               params = [], to_print1 = 'a', to_print2 = 'b', logScale = False, showZero = True, showSZ = False,
                                               savefig = False, filename = 'check', nsim = 0, colors = colors, strag_vn = '1', strag_LP = '1'):
    stressed_values1 = np.array(values1) * params[parameter1]
    stressed_values2 = np.array(values2) * params[parameter2]
    Nt = params['Nt']
    nu_all = np.zeros((Nt+1, len(stressed_values1)))
    for it, tc in enumerate(stressed_values1):
        np.random.seed(123)
        params_aux = params.copy()
        tc2 = stressed_values2[it]
        params_aux[parameter1] = tc
        params_aux[parameter2] = tc2
        env_aux = environment(**params_aux)
        S, Y, Z, X, nu, wealth, perf, rew, venue_wealth, A_a_hist, A_b_hist, A_W_hist, A_B_hist, rew_drift_hist = env_aux.simulate_venue(nsims=nsims, strag_LP=strag_LP, strag_vn=strag_vn, verbose = True)
        nu_all[:,it] = nu[:,nsim]
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6,4))
    for i in range(len(stressed_values1)):
        plt.plot(env_aux.timesteps, nu_all[:,i], label= to_print1 +  f'$* {(values1[i])}$'+r'$\,\,\,\&\,\,\,$'+ to_print2 +  f'$* {(values2[i])}$', color = colors[i])
    plt.xlabel(r'$t$', fontsize = fsL)
    plt.ylabel(r'$\nu^*_t$', fontsize = fsL)
    if showZero:
        plt.hlines(y = 0, xmin = 0, xmax = 1, linestyle = '--', alpha = 0.4, color='tab:red')
    if logScale:
        plt.yscale("log")
    plt.legend(fontsize = fs, loc='upper right')

    if savefig:
        plt.tight_layout()
        plt.savefig('figures/'+filename+'.pdf', format="pdf", dpi=dpi)
    plt.show()
    
    if showSZ:
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6,4))
        plt.plot(env_aux.timesteps, -(S[:,nsim]+ Z[:,nsim]), label= r'$-(S_t+Z_t)$')
        plt.xlabel(r'$t$', fontsize = 18)
        if showZero:
            plt.hlines(y = 0, xmin = 0, xmax = 1, linestyle = '--', alpha = 0.4, color='tab:red')
        if logScale:
            plt.yscale("log")
        plt.legend(fontsize = fs, loc='upper right')
        plt.show()
        
def params_show_nu(environment = utils_aux.environment, params = [], showZero = True, nsim =0, colors = colors, strag_vn = '1', strag_LP = '1'):
    Nt = params['Nt']
    nu_all = np.zeros((Nt+1,))
    np.random.seed(123)
    env_aux = environment(**params)
    S, Y, Z, X, nu, wealth, perf, rew, venue_wealth, A_a_hist, A_b_hist, A_W_hist, A_B_hist, rew_drift_hist = env_aux.simulate_venue(nsims=nsims, strag_LP=strag_LP, strag_vn=strag_vn, verbose = True)
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6,4))
    plt.plot(env_aux.timesteps, nu[:,nsim], color = colors[0])
    plt.xlabel(r'$t$', fontsize = 18)
    plt.ylabel(r'$\nu^*_t$', fontsize = 22)
    if showZero:
        plt.hlines(y = 0, xmin = 0, xmax = 1, linestyle = '--', alpha = 0.4, color='tab:red')
    plt.show()
    

    
def stress_parameter_and_show_nu(environment = utils_aux.environment, parameter = 'tempLP', values = [1,2,4,8,16], params = [], 
                                 to_print = 'a', showZero = True, verbose= False, savefig = False, 
                                 filename = 'check', nsim = 0, colors = colors, strag_vn = '1', strag_LP = '1'):
    Nt = params['Nt']
    stressed_values = np.array(values) * params[parameter]
    nu_all = np.zeros((Nt+1, len(stressed_values)))
    for it, tc in enumerate(stressed_values):
        np.random.seed(123)
        params_aux = params.copy()
        params_aux[parameter] = tc
        env_aux = environment(**params_aux)
        S, Y, Z, X, nu, wealth, perf, rew, venue_wealth, A_a_hist, A_b_hist, A_W_hist, A_B_hist, rew_drift_hist = env_aux.simulate_venue(nsims=nsims, strag_LP=strag_LP, strag_vn=strag_vn, verbose = True)
        nu_all[:,it] = nu[:,nsim]
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6,4))
    for i in range(len(stressed_values)):
        plt.plot(env_aux.timesteps, nu_all[:,i], label= to_print +  f'$* {(values[i])}$', color = colors[i])
    plt.xlabel(r'$t$', fontsize = 18)
    plt.ylabel(r'$\nu^*_t$', fontsize = 22)
    if showZero:
        plt.hlines(y = 0, xmin = 0, xmax = 1, linestyle = '--', alpha = 0.4, color='tab:red')
    plt.legend(fontsize = fs)
    if savefig:
        plt.tight_layout()
        plt.savefig('figures/'+filename+'.pdf', format="pdf", dpi=dpi)
    plt.show()
    
    if verbose:
        nu_to_show = np.zeros_like(env_aux.timesteps)
        nu_to_show2 = np.zeros_like(env_aux.timesteps)
        for it, t in enumerate(env_aux.timesteps):
            _,_,_,AB = env_aux.A(t=t, Yt=Y[it,nsim], Zt= Z[it,nsim], St= S[it,nsim], strag_vn = strag_vn)   
            nu_to_show[it] = env_aux.nu(t,   Yt=Y[it,nsim], Zt= Z[it,nsim], St= S[it,nsim], A_B = AB, strag_LP = strag_LP)
            nu_to_show2[it] = ( (env_aux.T - t) * (2*env_aux.a2*env_aux.feeV) - 2 * (S[it,nsim] + Z[it,nsim]) * env_aux.gamma * env_aux.tempLP * env_aux.eta**2 ) / (4* env_aux.eta**2 * env_aux.tempLP**2 * env_aux.gamma + 2*env_aux.tempLP)
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6,4))
        plt.plot(env_aux.timesteps, nu_to_show, '-', alpha = 0.4, color='tab:red', label = 'env')
        plt.plot(env_aux.timesteps, nu_to_show2, '--', alpha = 0.3, color = 'tab:blue', label='repl')
        plt.xlabel(r'$t$', fontsize = 18)
        plt.ylabel(r'$\nu^*_t~\mathrm{(replicated)}$', fontsize = 22)
        if showZero:
            plt.hlines(y = 0, xmin = 0, xmax = 1, linestyle = '--', alpha = 0.4, color='tab:red')
        plt.legend(fontsize = fs)
        plt.show()