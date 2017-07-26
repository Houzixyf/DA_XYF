import numpy as np
import inspect
from scipy.integrate import ode

from ipHelp import IPS
import pickle

class Simulator(object):
    """
    This class simulates the initial value problem that results from solving
    the boundary value problem of the control system.


    Parameters
    ----------

    ff : callable
        Vectorfield of the control system.

    T : float
        Simulation time.

    u : callable
        Function of the input variables.

    dt : float
        Time step.
    """

    def __init__(self, ff, T, start, u, z_par, dt=0.01):# dt=0.01
        """

        :param ff:      vectorfield function
        :param T:       end Time
        :param start:   initial state
        :param u:       input function u(t)
        :param dt:
        """
        self.ff = ff
        self.T = T
        self.u = u  ##:: self.eqs.trajectories.u
        self.dt = dt

        # this is where the solutions go
        self.xt = []
        self.ut = []
        self.nu = len(np.atleast_1d(self.u(0)))

        self.pt = z_par
        # time steps
        self.t = []

        # get the values at t=0
        self.xt.append(start)
        self.ut.append(self.u(0.0))  ##:: array([ 0.])
        self.t.append(0.0)

        # initialise our ode solver
        self.solver = ode(self.rhs)
        self.solver.set_initial_value(start)
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)
        # self.solver.set_integrator('lsoda', rtol=1e-6)
        # self.solver.set_integrator('dop853', rtol=1e-6)

    def rhs(self, t, x):
        """
        Retruns the right hand side (vector field) of the ode system.
        """
        u = self.u(t)
        p = self.pt
        dx = self.ff(x, u, p)
        
        return dx

    def calcstep(self):
        """
        Calculates one step of the simulation.
        """
        x = list(self.solver.integrate(self.solver.t+self.dt))
        t = round(self.solver.t, 5)  ##:: round(2.123456,5)=2.12346
        save_res = False
        if 0 <= t <= self.T:
            self.xt.append(x)
            ##:: when t=0.0: [[0.0, 0.0, 1.2566370614359172, 0.0], 
            ##:: when t=0.0+dt=0.01: [2.5944857092488461e-05, 0.0077834571264927509, 1.2566039006001195, -0.0099483487759786798]]
            self.ut.append(self.u(t))
            ##:: [array([ 0.]), array([ 1.55669143])] for t=0.0 and t=0.01
            self.t.append(t) # [0.0, 0.01]
            if t == 0.99 and save_res:
                res_x = self.xt
                res_u = self.ut
                res_t = t
                save_res_x = open('d:\\x_refsol.plk', 'wb')
                pickle.dump(res_x, save_res_x)
                save_res_x.close()
                save_res_u = open('d:\\u_refsol.plk', 'wb')
                pickle.dump(res_u, save_res_u)
                save_res_u.close()
                save_res_t = open('d:\\t_refsol.plk', 'wb')
                pickle.dump(res_t, save_res_t)
                save_res_t.close()
        return t, x

    def simulate(self):
        """
        Starts the simulation


        Returns
        -------

        List of numpy arrays with time steps and simulation data of system and input variables.
        """
        path = 'd:\temp_data' # 'E:\Yifan_Xue\DA\Data\Data_for_Brockett_pe(k,0.1,15)_t_0.99'
        t = 0
        while t <= self.T:
            t, y = self.calcstep()

            save_refsol = False
            if t == 0.99 and save_refsol:
                res_x = self.xt[-1]
                res_u = self.ut[-1]
                res_t = self.t[-1]
                save_res_x = open(path+'\\x.plk', 'wb')
                pickle.dump(res_x, save_res_x)
                save_res_x.close()
                save_res_u = open(path+'\\u.plk', 'wb')
                pickle.dump(res_u, save_res_u)
                save_res_u.close()
                save_res_t = open(path+'\\t.plk', 'wb')
                pickle.dump(res_t, save_res_t)
                save_res_t.close()

        self.ut = np.array(self.ut).reshape(-1, self.nu)



        return [np.array(self.t), np.array(self.xt), np.array(self.ut)]
