import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import odeint

class Zimmer:
    def __init__(self, LNA, measurements, ndim, timestep=10000, estimate_N=True, N=None):
        self.ndim = ndim
        self.timestep = timestep
        self.LNA = LNA
        self.measurements = measurements
        self.estimate_N = estimate_N
        self.N = N

    def likelihood_next(self, data_now, data_next, time_now, time_next, *params):
        '''Returns the probability of obtaining the data point at time t_{i+1}
        via the LNA starting from the data point at time t_i given a set of
        parameter values'''

        t = np.linspace(time_now, time_next, self.timestep)
        init_cond = np.array([data_now, 0.]) if self.ndim == 1 else np.concatenate((data_now, np.zeros(self.ndim**2)))
        lna = odeint(self.LNA, init_cond, t, args=params)[-1]
        mean = lna[0:self.ndim]
        cov = lna[self.ndim:]
        if self.estimate_N:
            n = params[-1]
        else:
            n = self.N
        if self.ndim == 1:
            std = np.sqrt(cov[0]/n)
            dist = norm(loc=mean[0], scale=std)
        else:
            cov = np.reshape(cov, (self.ndim, self.ndim))
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            if min_eig < 0.:
                cov -= 2*min_eig*np.eye(*cov.shape)
            dist = multivariate_normal(mean, cov/n, allow_singular=True)
        return dist.pdf(data_next)

    def costfn(self, params):
        '''Returns the value of the cost function for a set of parameter values'''

        data, times = self.measurements
        total = 0.
        for i in range(len(data)-1):
            total -= np.log1p(self.likelihood_next(data[i], data[i+1], times[i], times[i+1], *params))
        return total

    def draw_next(self, data_now, time_now, time_next, *params):
        '''Returns a candidate data point at time t_{i+1} given the real data
        point at time t_i'''

        t = np.linspace(time_now, time_next, self.timestep)
        init_cond = np.array([data_now, 0.]) if self.ndim == 1 else np.concatenate((data_now, np.zeros(self.ndim**2)))
        lna = odeint(self.LNA, init_cond, t, args=params)[-1]
        mean = lna[0:self.ndim]
        cov = lna[self.ndim:]
        if self.estimate_N:
            n = params[-1]
        else:
            n = self.N
        if self.ndim == 1:
            std = np.sqrt(cov[0]/n)
            return np.random.normal(loc=mean[0], scale=std)
        cov = np.reshape(cov, (self.ndim, self.ndim))
        return np.random.multivariate_normal(mean, cov/n)

    def reconstruct(self, *params):
        '''Returns the reconstructed process for a set of parameter values'''

        data, times = self.measurements
        x_t = [data[0]]
        for i in range(len(data)-1):
            x_t.append(self.draw_next(data[i], times[i], times[i+1], *params[0]))
        return np.array(x_t)
