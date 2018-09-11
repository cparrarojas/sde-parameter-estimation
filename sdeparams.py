import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import odeint

class Zimmer:
    def __init__(self, A, B, Jac, measurements, ndim, n_obs=None, timestep=10000, estimate_N=True, N=None):
        self.ndim = ndim
        self.timestep = timestep
        self.A = A
        self.B = B
        self.Jac = Jac
        self.measurements = measurements
        self.estimate_N = estimate_N
        self.N = N
        self.n_obs = ndim if n_obs is None else n_obs

    def LNA(self, x, t, *params):
        y = np.maximum(x[:self.ndim], np.zeros(self.ndim))
        x_det = self.A(y, t, *params)
        Xi = np.reshape(x[self.ndim:], (self.ndim,self.ndim))
        J = self.Jac(y, t, *params)
        dXidt = np.dot(J, Xi) + np.dot(Xi, J.transpose()) + self.B(y, t, *params)
        
        return np.concatenate((x_det, dXidt.flatten()))

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
            return dist.pdf(data_next), None

        elif self.ndim > self.n_obs:
            cov = np.reshape(cov, (self.ndim, self.ndim))
            cov_obs = cov[:self.n_obs, :self.n_obs]
            cov_hid = cov[self.n_obs:, self.n_obs:]
            min_eig_obs = np.min(np.real(np.linalg.eigvals(cov_obs)))
            min_eig_hid = np.min(np.real(np.linalg.eigvals(cov_hid)))

            if min_eig_obs < 0.:
                cov_obs -= 2*min_eig_obs*np.eye(*cov_obs.shape)

            if min_eig_hid < 0.:
                cov_hid -= 2*min_eig_hid*np.eye(*cov_hid.shape)

            if np.linalg.det(cov_obs) == 0.:
                cov_obs+=1e-6*np.eye(*cov_obs.shape)

            dist = multivariate_normal(mean[:self.n_obs], cov_obs/n)

            prefactor = np.dot(cov[self.n_obs:, :self.n_obs], np.linalg.inv(cov_obs))
            hidden_data = mean[self.n_obs:] + np.dot(prefactor, data_next - mean[:self.n_obs])

            return dist.pdf(data_next), hidden_data

        else:
            cov = np.reshape(cov, (self.ndim, self.ndim))
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            if min_eig < 0.:
                cov -= 2*min_eig*np.eye(*cov.shape)
            dist = multivariate_normal(mean, cov/n, allow_singular=True)
            return dist.pdf(data_next), None

    def costfn(self, params):
        '''Returns the value of the cost function for a set of parameter values'''

        data, times = self.measurements

        if self.ndim > self.n_obs:
            now = np.append(data[0], params[:self.ndim-self.n_obs])
        else:
            now = data[0]

        total = 0.
        for i in range(len(data)-1):
            estimation = self.likelihood_next(now, data[i+1], times[i], times[i+1], *params)
            total -= np.log(estimation[0])
            if self.ndim > self.n_obs:
                now = np.append(data[i+1], estimation[1])
            else:
                now = data[i+1]
        return total

    def draw_next(self, data_now, data_next, time_now, time_next, *params):
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
        elif self.ndim > self.n_obs:
            cov = np.reshape(cov, (self.ndim, self.ndim))
            cov_obs = cov[:self.n_obs, :self.n_obs]
            cov_hid = cov[self.n_obs:, self.n_obs:]
            min_eig_obs = np.min(np.real(np.linalg.eigvals(cov_obs)))
            min_eig_hid = np.min(np.real(np.linalg.eigvals(cov_hid)))

            if min_eig_obs < 0.:
                cov_obs -= 2*min_eig_obs*np.eye(*cov_obs.shape)

            if min_eig_hid < 0.:
                cov_hid -= 2*min_eig_hid*np.eye(*cov_hid.shape)

            if np.linalg.det(cov_obs) == 0.:
                return mean

            prefactor = np.dot(cov[self.n_obs:, :self.n_obs], np.linalg.inv(cov_obs))
            hidden_data = mean[self.n_obs:] + np.dot(prefactor, data_next - mean[:self.n_obs])

            return np.concatenate((np.random.multivariate_normal(mean[:self.n_obs], cov_obs/n), hidden_data))
        else:
            cov = np.reshape(cov, (self.ndim, self.ndim))
            return np.random.multivariate_normal(mean, cov/n)

    def reconstruct(self, *params):
        '''Returns the reconstructed process for a set of parameter values'''

        data, times = self.measurements

        if self.ndim > self.n_obs:
            now = np.append(data[0], params[0][:self.ndim-self.n_obs])
        else:
            now = data[0]
        x_t = [now]
        for i in range(len(data)-1):
            next = self.draw_next(now, data[i+1], times[i], times[i+1], *params[0])
            x_t.append(next)
            if self.ndim > self.n_obs:
                now = np.append(data[i+1], next[self.n_obs:])
            else:
                now = data[i+1]
        return np.array(x_t)

def likelihood_next(params, LNA, data_now, data_next, time_now, time_next, n_obs=None, estimate_N=True,
                    N=None, timestep=10000):
    '''Returns the probability of obtaining the data point at time t_{i+1}
    via the LNA starting from the data point at time t_i given a set of
    parameter values'''

    ndim = data_now.shape[0]
    if n_obs is None:
        n_obs = ndim

    t = np.linspace(time_now, time_next, timestep)

    init_cond = np.array([data_now, 0.]) if ndim == 1 else np.concatenate((data_now, np.zeros(ndim**2)))
    lna = odeint(LNA, init_cond, t, args=tuple(params))[-1]
    mean = lna[0:ndim]
    cov = lna[ndim:]

    n = params[-1] if estimate_N else N

    if ndim == 1:
        std = np.sqrt(cov[0]/n)
        dist = norm(loc=mean[0], scale=std)
        return dist.pdf(data_next), None

    elif ndim > n_obs:
        cov = np.reshape(cov, (ndim, ndim))
        cov_obs = cov[:n_obs, :n_obs]
        cov_hid = cov[n_obs:, n_obs:]
        min_eig_obs = np.min(np.real(np.linalg.eigvals(cov_obs)))
        min_eig_hid = np.min(np.real(np.linalg.eigvals(cov_hid)))

        if min_eig_obs < 0.:
            cov_obs -= 2*min_eig_obs*np.eye(*cov_obs.shape)

        if min_eig_hid < 0.:
            cov_hid -= 2*min_eig_hid*np.eye(*cov_hid.shape)

        if np.linalg.det(cov_obs) == 0.:
            cov_obs+=1e-6*np.eye(*cov_obs.shape)

        dist = multivariate_normal(mean[:n_obs], cov_obs/n)

        prefactor = np.dot(cov[n_obs:, :n_obs], np.linalg.inv(cov_obs))
        hidden_data = np.maximum(mean[n_obs:] + np.dot(prefactor, data_next - mean[:n_obs]), np.zeros(ndim-n_obs))

        return dist.pdf(data_next), hidden_data

    else:
        cov = np.reshape(cov, (ndim, ndim))
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        if min_eig < 0.:
            cov -= 2*min_eig*np.eye(*cov.shape)
        dist = multivariate_normal(mean, cov/n, allow_singular=True)
        return dist.pdf(data_next), None



def draw_next(params, LNA, data_now, data_next, time_now, time_next, n_obs=None, estimate_N=True,
              N=None, timestep=10000):
    '''Returns a candidate data point at time t_{i+1} given the real data
    point at time t_i'''

    ndim = data_now.shape[0]
    if n_obs is None:
        n_obs = ndim

    t = np.linspace(time_now, time_next, timestep)

    init_cond = np.array([data_now, 0.]) if ndim == 1 else np.concatenate((data_now, np.zeros(ndim**2)))
    lna = odeint(LNA, init_cond, t, args=tuple(params))[-1]
    mean = lna[0:ndim]
    cov = lna[ndim:]

    n = params[-1] if estimate_N else N

    if ndim == 1:
        std = np.sqrt(cov[0]/n)
        return np.random.normal(loc=mean[0], scale=std)
    elif ndim > n_obs:
        cov = np.reshape(cov, (ndim, ndim))
        cov_obs = cov[:n_obs, :n_obs]
        cov_hid = cov[n_obs:, n_obs:]
        min_eig_obs = np.min(np.real(np.linalg.eigvals(cov_obs)))
        min_eig_hid = np.min(np.real(np.linalg.eigvals(cov_hid)))

        if min_eig_obs < 0.:
            cov_obs -= 2*min_eig_obs*np.eye(*cov_obs.shape)
        if min_eig_hid < 0.:
            cov_hid -= 2*min_eig_hid*np.eye(*cov_hid.shape)

        if np.linalg.det(cov_obs) == 0.:
            return mean

        prefactor = np.zeros((ndim-n_obs, n_obs)) if np.linalg.det(cov_obs) == 0. else np.dot(cov[n_obs:, :n_obs], np.linalg.inv(cov_obs))
        hidden_data = mean[n_obs:] + np.dot(prefactor, data_next - mean[:n_obs])

        return np.concatenate((np.random.multivariate_normal(mean[:n_obs], cov_obs/n), hidden_data))
    else:
        cov = np.reshape(cov, (ndim, ndim))
        return np.random.multivariate_normal(mean, cov/n)
