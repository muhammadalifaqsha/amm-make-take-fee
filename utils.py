from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp


class environment:
    """
    Model parameters for the environment.
    """
    def __init__(self, sigma = 1., S0 = 100., Z0 = 100., Y0 = 1000., eta = 0.1, xi = 0.1, T = 1., a1 = 1, a2 = 1, a3 = 1, Nt =1_000,
                fee = 0.5, feeV = 0.5, C = 0.1, gamma = 1, zeta = 1, tempLP = 1e-2, P0 = 1, corr_BMs = 0.):
        self.sigma = sigma
        self.T = T
        self.S0 = S0
        self.Z0 = Z0
        self.Y0 = Y0
        self.eta = eta
        self.Nt = Nt
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.xi = xi
        self.fee = fee
        self.feeV = feeV
        self.C = C
        self.gamma = gamma
        self.zeta = zeta
        self.tempLP = tempLP
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.dt = self.T/self.Nt
        self.P0 = P0
        self.corr_BMs = corr_BMs

    def simulate_price_market(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.S0
        errs = np.random.randn(self.Nt, nsims)
        sigma = self.sigma
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + np.sqrt(self.dt) * sigma * errs[t,:]
        return x
    
    def lambda_a(self, Yt, Zt, St):
        Yt = Yt.reshape((-1,))
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Yt)
        zeros_n = np.zeros((nsims,))
        lambdaa = np.maximum(0, self.a1 + self.a2*Yt + self.a3*(St-Zt))*self.dt
        return lambdaa
    
    def lambda_b(self, Yt, Zt, St):
        Yt = Yt.reshape((-1,))
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Yt)
        zeros_n = np.zeros((nsims,))
        lambdab = np.maximum(0, self.a1 + self.a2*Yt + self.a3*(Zt-St))*self.dt
        return lambdab
    
    def simulate_jumps(self, Yt, Zt, St):
        Yt = Yt.reshape((-1,))
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Yt)
        zeros_n = np.zeros((nsims,))
        lambdaa = self.lambda_a(Yt, Zt, St)
        lambdab = self.lambda_b(Yt, Zt, St)
        jump_a = (np.random.uniform(size=(nsims,)) < lambdaa).astype(int)
        jump_b = (np.random.uniform(size=(nsims,)) < lambdab).astype(int)
        return jump_a, jump_b, lambdaa, lambdab

    def simulate_venue(self, nsims=1, strag_LP='1', strag_vn='1', verbose = False):
        S = np.zeros((self.Nt+1, nsims))
        S[0,:] = self.S0
        Y = np.zeros((self.Nt+1, nsims))
        Y[0,:] = self.Y0
        Z = np.zeros((self.Nt+1, nsims))
        Z[0,:] = self.Z0
        X = np.zeros((self.Nt+1, nsims))
        X[0,:] = self.Y0*self.Z0
        nu = np.zeros((self.Nt+1, nsims))
        #nu[0,:] = self.nu(0, Y[0,:], Z[0,:], S[0,:], strag)
        
        # Reward components
        A_a_hist = np.zeros((self.Nt, nsims))
        A_b_hist = np.zeros((self.Nt, nsims))
        A_W_hist = np.zeros((self.Nt, nsims))
        A_B_hist = np.zeros((self.Nt, nsims))
        rew_drift_hist = np.zeros((self.Nt, nsims))
        
        errs_W = np.random.randn(self.Nt, nsims)
        errs_B = np.random.randn(self.Nt, nsims)
        wealth = np.zeros((1,nsims))
        rew = np.zeros((self.Nt+1, nsims))
        rew[0,:] = self.P0
        
        venue_wealth = np.zeros((1,nsims))
        for t in range(self.Nt):
            jump_a, jump_b, lambdaa, lambdab = self.simulate_jumps(Y[t,:], Z[t,:], S[t,:])
#             A_a = self.xi*(S[t,:] - Z[t,:]*Y[t,:]/(Y[t,:] - self.eta)) - self.fee
#             A_b = -self.xi*(S[t,:] - Z[t,:]*Y[t,:]/(Y[t,:] + self.eta)) - self.fee
#             A_W = - Y[t,:]*self.sigma
#             nu_dummy = self.nu(self.timesteps[t], Y[t,:], Z[t,:], S[t,:], strag='1')
#             A_B = nu_dummy*2*self.tempLP*self.eta
            
            A_a, A_b, A_W, A_B = self.A(self.timesteps[t], Y[t,:], Z[t,:], S[t,:], strag_vn)
            nu[t,:] = self.nu(self.timesteps[t], Y[t,:], Z[t,:], S[t,:], A_B, strag_LP)
            
            # store reward components
            A_a_hist[t,:] = A_a
            A_b_hist[t,:] = A_b
            A_W_hist[t,:] = A_W
            A_B_hist[t,:] = A_B
            
            S[t+1,:] = S[t,:] + np.sqrt(self.dt) * self.sigma * errs_W[t,:]
            Y[t+1,:] = Y[t,:] + nu[t,:] * self.dt + np.sqrt(self.dt) * self.eta * errs_B[t,:] + self.xi*(jump_b - jump_a)
            Z[t+1,:] = Z[t,:] + Z[t,:]*((Y[t,:]/(Y[t,:]+self.xi))**2 - 1)*jump_b + Z[t,:]*((Y[t,:]/(Y[t,:]-self.xi))**2 - 1)*jump_a
            X[t+1,:] = X[t,:] + Z[t,:]*nu[t,:] * self.dt + Z[t,:] * self.eta * errs_B[t,:] * np.sqrt(self.dt) + X[t,:]*(Y[t,:]/(Y[t,:]+self.xi) - 1)*jump_b + X[t,:]*(Y[t,:]/(Y[t,:]-self.xi) - 1)*jump_a
            #nu[t+1, :] = self.nu(self.timesteps[t+1], Y[t+1,:], Z[t+1,:], S[t+1,:], strag_LP)
            
            rew[t+1, :] = rew[t, :] + (A_a*jump_a + A_b*jump_b + A_W * np.sqrt(self.dt) * errs_W[t,:] + A_B * np.sqrt(self.dt) * errs_B[t,:])
            
            # reward drift
#             if strag_vn == '0':
#                 rew_drift = 0
#             elif strag_vn == '1':
            rew_drift = self.C*Y[t,:]*np.abs(S[t,:] - Z[t,:])
            rew_drift -= A_B**2/(4*self.eta**2 * self.tempLP)
            rew_drift += 0.5*self.gamma*(A_W + self.sigma*Y[t,:])**2
            rew_drift += 0.5*self.gamma*(A_B + self.eta*(S[t,:]+Z[t,:]))**2
            rew_drift += A_B * nu[t,:]/self.eta
            rew_drift -= (lambdaa*(1 - np.exp(-self.gamma*(A_a-self.xi*(S[t,:] - Z[t,:]*Y[t,:]/(Y[t,:] - self.eta)) + self.fee)))/self.gamma)
            rew_drift -= (lambdab*(1 - np.exp(-self.gamma*(A_b+self.xi*(S[t,:] - Z[t,:]*Y[t,:]/(Y[t,:] + self.eta)) + self.fee)))/self.gamma)
            
            rew_drift_hist[t, :] = rew_drift
            rew[t+1, :] = rew[t+1, :] + self.dt*rew_drift
            
            wealth[0,:] -= Z[t,:]*nu[t, :]*self.dt
            wealth[0,:] -= (S[t,:] + self.tempLP*nu[t, :])*nu[t, :]*self.dt
            wealth[0,:] -= self.C*Y[t,:]*np.abs(S[t,:] - Z[t,:])*self.dt
            wealth[0,:] += self.fee*(jump_a + jump_b)
            venue_wealth[0,:] += self.feeV*(jump_a + jump_b)
        nu[-1, :] = nu[-2,:]
        wealth[0,:] += Y[-1,:]*(Z[-1,:] + S[-1,:])
        wealth[0,:] -= Y[0,:]*(Z[0,:] + S[0,:])
        wealth[0,:] += rew[-1,:]
        venue_wealth[0,:] -= rew[-1,:]
        perf = -np.exp(-self.gamma*wealth)
        if verbose:
            return S, Y, Z, X, nu, wealth, perf, rew, venue_wealth, A_a_hist, A_b_hist, A_W_hist, A_B_hist, rew_drift_hist, errs_B
        else:
            return S, Y, Z, X, nu, wealth, perf, rew, venue_wealth
    
    def partial_Y_h_1(self, t, Yt, Zt, St):
        lambdaa = self.lambda_a(Yt, Zt, St)
        lambdab = self.lambda_b(Yt, Zt, St)
        cond1 = True #((lambdaa > 0) & (lambdab > 0))
        strag1 = (self.T - t) * (2*self.a2*(self.fee + self.feeV) - self.C*np.abs(St-Zt))
        cond2 = False #((lambdaa > 0) & (lambdab == 0))
        if self.a2!= 0:
            strag2 = self.a2*(self.fee + self.feeV)*(1 - np.exp(-self.a2*self.xi*(self.T-t)))/(self.a2*self.xi)
            strag2 += Zt*(self.C + self.a2*self.xi)*(np.exp(self.a2*self.xi*(self.T-t))-1)/(self.a2*self.xi)
            strag2 += St*(self.C + self.a2*self.xi)*(np.exp(-self.a2*self.xi*(self.T-t))-1)/(self.a2*self.xi)
            
            strag3 = self.a2*(self.fee + self.feeV)*(np.exp(self.a2*self.xi*(self.T-t))-1)/(self.a2*self.xi)
            strag3 += Zt*(self.C + self.a2*self.xi)*(np.exp(-self.a2*self.xi*(self.T-t))-1)/(self.a2*self.xi)
            strag3 += St*(self.C + self.a2*self.xi)*(np.exp(self.a2*self.xi*(self.T-t))-1)/(self.a2*self.xi)
        else:
            strag2 = Zt*self.C*(self.T - t)
            strag2 += -St*self.C*(self.T-t)
            
            strag3 = -Zt*self.C*(self.T - t)
            strag3 += St*self.C*(self.T-t)
        cond3 = False #((lambdaa == 0) & (lambdab > 0))
        strag = cond1*strag1 + cond2*strag2 + cond3*strag3
        return strag
    
    def A(self, t, Yt, Zt, St, strag_vn):
        if strag_vn == '0':
            A_B = 0
            A_W = 0
            A_a = 0
            A_b = 0
        elif strag_vn == '1':
            A_a = self.xi*(St - Zt*Yt/(Yt - self.xi)) - self.fee
            A_b = -self.xi*(St - Zt*Yt/(Yt + self.xi)) - self.fee
            A_W = - Yt*self.sigma
            A_B = (self.partial_Y_h_1(t, Yt, Zt, St)/(self.tempLP*self.eta) - 2*(St+Zt)*self.gamma*self.eta)/(2*self.gamma + 1/(self.tempLP * self.eta**2))
        return A_a, A_b, A_W, A_B
    
    def nu(self, t, Yt, Zt, St, A_B, strag_LP):
        if strag_LP == '0':
            strag = 0
        elif strag_LP == '1':
            strag = A_B/(2*self.tempLP*self.eta)
        return strag
        