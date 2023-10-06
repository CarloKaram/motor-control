import numpy as np

class MotorCoulomb:
    ''' A DC motor with Coulomb friction.
        To simplify things we neglect the electrical dynamics
            V = R*i + K_b*dq
            tau = K_b*i
            tau = I_m*ddq + b_m*dq + tau_f
        where:
            V = voltage
            i = current
            dq = velocity
            ddq = acceleration
            tau = torque
            R = resistance
            I_m = motor inertia
            b_m = motor viscous friction coefficient
            tau_f = Coulomb friction torque

        Defining the system state as motor angle and velocity:
            x = (q, dq)
        
        Integrating the system with explicit Euler:
            tau = I (dq_next - dq)/dt + b*dq + tau_f        
            I*dq_next = dt*(tau - b*dq) + I*dq - dt*tau_f
            I*dq_next = s - dt*tau_f
        where s = dt*(tau - b*dq) + I*dq.
        We know that tau_f is bounded:
            -tau_0 <= tau_f <= tau_0        
        To ensure maximum dissipation we minimize | dq_next |^2
        This is equivalent to minimizing |s - dt*tau_f|^2
        The solution is simple.
        
        If |s/dt|<= tau_0 => tau_f=s/dt     => dq_next = 0
        Otherwise tau_f = tau_0*sign(s)	 => dq_next = (s-dt*tau_0*sign(s))/I        
    '''

    def __init__(self, dt, params):
        # store motor parameters in member variables
        self.dt  = dt
        self.R   = params.R
        self.K_b = params.K_b
        self.tau_0_m = params.tau_coulomb
        self.tau_0_j = 1e-2
        self.tau_0 = self.tau_0_m + self.tau_0_j
        self.I = params.I_m
        self.b_j = 1e-4
        self.b = params.b_m + self.b_j
        self.tau_f = 0.0

        # set initial motor state (pos, vel) to zero
        self.x = np.zeros(2)
        self.torque = 0.0

    def set_state(self, x):
        self.x = np.copy(x)
        
    def simulate(self, i, method='time-stepping'):
        ''' Simulate assuming current as control input '''
        dq = self.x[1]
        
        self.voltage = self.R * i + self.K_b * dq
        torque = self.K_b * i
        self.simulate_torque(torque, method)

    def simulate_voltage(self, V, method='time-stepping'):
        ''' Simulate assuming voltage as control input '''
        dq = self.x[1]
        
        i = (V - self.K_b * dq) / self.R
        torque = self.K_b * i
        self.simulate_torque(torque, method)

    def simulate_torque(self, torque, method='time-stepping'):
        ''' Simulate assuming torque as control input '''
        self.torque = torque
        dq = self.x[1]
        
        S = self.I * dq + self.dt * (self.torque - self.b * dq)
        
        if(method=='time-stepping'):
            if (np.abs(S/self.dt) <= self.tau_0):
                self.tau_f = S/self.dt
            else:
                self.tau_f = np.sign(S) * self.tau_0
        
        elif(method=='standard'):
            if (dq == 0.0):
                if np.abs(S/self.dt) < self.tau_0:
                    self.tau_f = S / self.dt
                else:
                    self.tau_f = np.sign(S) * self.tau_0
            else:
                self.tau_f = np.sign(dq) * self.tau_0
        
        else:
            print("ERROR: unknown integration method:", method)
            return self.x 
        
        # compute next state
        self.x[0] += dq * self.dt
        self.x[1] = (S - self.dt * self.tau_f) / self.I

        return self.x
     
    def q(self):
        return self.x[0]
        
    def dq(self):
        return self.x[1]

    def i(self):
        return self.torque / self.K_b
    
    def tau(self):
        return self.torque
    
    def tau_coulomb(self):
        return self.tau_f

    def V(self):
        return self.R*self.i() + self.K_b*self.dq()


def run_open_loop(ax, dt, method='time-stepping'):
    ''' Carry out an open-loop simulation of a DC motor with Coulomb friction
        given an input signal obtained by the sum of a constant term, a linear
        term (in time), and a sinusoidal term.
    '''
    from dc_motor import get_motor_parameters
    T = 2           # simulation time
    V_b = 0.0       # initial motor input
    V_a = 0.0       # linear increase in motor input per second
    V_w = .5        # frequency of the sinusoidal motor input
    V_A = 1.1       # amplitude of the sinusoidal motor input
    params = get_motor_parameters('Focchi2013')
    params.tau_coulomb = 1
    
    # simulate motor with linear+sinusoidal input torque
    N = int(T/dt)   # number of time steps
    motor = MotorCoulomb(dt, params)
#    motor.set_state(np.array([0.0, 1e-5]))
    q = np.zeros(N+1)
    dq = np.zeros(N+1)
    tau_f = np.zeros(N+1)
    tau = np.zeros(N)
    for i in range(N):
        tau[i] = V_a*i*dt + V_b + V_A*np.sin(2*np.pi*V_w*i*dt)
        motor.simulate_torque(tau[i], method)
        #motor.simulate(tau[i], method)
        
        q[i+1] = motor.q()
        dq[i+1] = motor.dq()
        tau_f[i] = motor.tau_f/dt
        tau[i] = motor.tau()

    # plot motor angle, velocity and current
    time = np.arange(0.0, T+dt, dt)
    time = time[:N+1]
    alpha = 0.8
    ax[0].plot(time, q, label ='q '+method, alpha=alpha)
    ax[1].plot(time, dq, label ='dq '+method, alpha=alpha)
    ax[2].plot(time[:-1], tau, label ='tau '+method, alpha=alpha)
    ax[-1].plot(time, tau_f, '--', label ='tau_f '+method, alpha=alpha)
    for i in range(len(ax)): ax[i].legend()
    plt.xlabel('Time [s]')


if __name__=='__main__':
    import arc.utils.plot_utils as plut
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    f, ax = plt.subplots(4,1,sharex=True)
    run_open_loop(ax, dt=1e-3, method='time-stepping')
    run_open_loop(ax, dt=1e-3, method='standard')
    plt.show()
    
