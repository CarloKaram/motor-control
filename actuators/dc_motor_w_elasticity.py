import numpy as np
from dc_motor import get_motor_parameters as gmp

def get_motor_parameters(name):
    params = gmp(name)
    if(name=='Focchi2013'):
        params.I_j = 0.2                # joint-side inertia
        params.K = 250                  # transmission torsional stiffness
        params.B = 0.1                  # transmission damping coefficient
        params.b_j = 1e-4               # joint viscous friction
        params.tau_0_j = 1e-2           # max joint Coulomb friction
    elif(name=="Maxon148877"):          # Maxon 148877 (150W,48V)
        params.I_j = 1.0                # joint-side inertia
        params.K = 101.6                # transmission torsional stiffness
        params.B = 0.5                  # transmission damping coefficient
        params.b_j = 1e-4               # joint viscous friction
        params.tau_0_j = 1e-2           # max joint Coulomb friction
    return params


class MotorWElasticity:
    ''' A DC motor with elastic transmission, with the following dynamics (neglecting electric pole)
            V = R*i + K_b*dq_m
            tau_m = K_b*i
            tau_m = I_m*ddq_m + B*(dq_m - dq_j) + K*(q_m - q_j) + b_m*dq_m + tau_f_m
            I_j*ddq_j + B*(dq_j - dq_m) + K*(q_j - q_m) + b_j*dq_j + tau_f_j = 0
        where:
            V = voltage
            i = current
            R = resistance
            K_b = motor speed/torque constant
            q_m = motor angle
            dq_m = motor velocity
            ddq_m = motorr acceleration
            q_j = joint angle
            dq_j = joint velocity
            ddq_j = joint acceleration
            tau_m = motor torque
            tau_f_m = motor Coulomb friction
            tau_f_j = joint Coulomb friction
            I_m = motor inertia
            b_m = motor viscous friction coefficient
            I_j = joint inertia
            b_j = joint viscous friction coefficient
            K = transmission torsional stiffness
            B = transmission damping coefficient

        Define the system state as motor angle q_m, motor velocity dq_m, joint angle q_j and velocity dq_j:
            x = (q_m, dq_m, q_j, dq_j)
        and the control input is the motor current i.
    '''
    
    def __init__(self, dt, params):
        # store motor parameters in member variables
        self.dt  = dt                           # simulation time step
        self.R   = params.R                     # motor resistance
        self.K_b = params.K_b                   # motor speed constant
        self.I_m = params.I_m                   # motor rotor inertia
        self.b_m = params.b_m                   # motor viscous friction
        self.I_j = params.I_j                   # joint-side inertia
        self.b_j = params.b_j                   # joint viscous friction
        self.K = params.K                       # transmission torsional stiffness
        self.B = params.B                       # transmission damping coefficient
        self.tau_0_m = params.tau_coulomb       # max motor Coulomb friction
        self.tau_0_j = params.tau_0_j           # max joint Coulomb friction

        # set initial state to zero
        self.x = np.zeros(4)

    def set_state(self, x):
        self.x = np.copy(x)

    def set_mot_state(self, x):
        self.x[0:2] = np.copy(x)    
        
    def simulate_voltage(self, V, method='time-stepping'):
        ''' Simulate assuming voltage as control input '''
        dq_m = self.x[1]
        i = (V - self.K_b*dq_m)/self.R
        self.simulate(i)
        
    def simulate(self, i, method='time-stepping'):
        q_m = self.x[0]
        dq_m = self.x[1]
        q_j = self.x[2]
        dq_j = self.x[3]
        self.current = i
        self.voltage = self.R * self.current + self.K_b * dq_m 
        self.tau_m = self.K_b * self.current

        s_m = self.I_m * dq_m + self.dt * (self.tau_m - self.b_m * dq_m + self.K * (q_j - q_m) + self.B * (dq_j - dq_m))
        s_j = self.I_j * dq_j + self.dt * (self.K * (q_m - q_j) + self.B * (dq_m - dq_j) - self.b_j * dq_j)    
        
        # compute friction torque
        if(method=='time-stepping'):
            if np.abs(s_m/self.dt) <= self.tau_0_m:
                self.tau_f_m = s_m/self.dt
            else:
                self.tau_f_m = self.tau_0_m*np.sign(s_m)
            if np.abs(s_j/self.dt) <= self.tau_0_j:
                self.tau_f_j = s_j/self.dt
            else:
                self.tau_f_j = self.tau_0_j*np.sign(s_j)      
        elif(method=='standard'):
            if dq_m==0.0:
                if np.abs(s_m/self.dt) < self.tau_0_m:
                    self.tau_f_m = s_m/self.dt
                else:
                    self.tau_f_m = self.tau_0_m*np.sign(s_m)
            else:
                self.tau_f_m = self.tau_0_m*np.sign(dq_m)
            if dq_j==0.0:
                if np.abs(s_j/self.dt) < self.tau_0_j:
                    self.tau_f_j = s_j/self.dt
                else:
                    self.tau_f_j = self.tau_0_j*np.sign(s_j)
            else:
                self.tau_f_j = self.tau_0_j*np.sign(dq_j)   
        else:
            print("ERROR: unknown integration method:", method)
            return self.x
        
        # compute acceleration
        ddq_m = (self.tau_m - self.tau_f_m - self.b_m * dq_m + self.K * (q_j - q_m) + self.B * (dq_j - dq_m)) #/ self.I_m
        ddq_j = (self.K * (q_m - q_j) + self.B * (dq_m - dq_j) - self.b_j * dq_j - self.tau_f_j) #/ self.I_j
        
        # compute next state
        self.x[0] += dq_m * self.dt + 0.5 * (self.dt ** 2) * ddq_m
        self.x[1] = (s_m - self.dt * self.tau_f_m) / self.I_m
        self.x[2] += dq_j * self.dt + 0.5 * (self.dt ** 2) * ddq_j
        self.x[3] = (s_j - self.dt * self.tau_f_j) / self.I_j

        return self.x
     
    def q_m(self):
        return self.x[0]
        
    def dq_m(self):
        return self.x[1]

    def q_j(self):
        return self.x[2]
        
    def dq_j(self):
        return self.x[3]

    def i(self):
        return self.current
           
    def tau(self):
        return self.tau_m                 
        
    def V(self):
        return self.voltage


if __name__=='__main__':
    #import arc.utils.plot_utils as plut
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    dt = 1e-5       # controller time step
    T = 3.0         # simulation time
    u_w = 1.0       # frequency of sinusoidal control input
    u_A = 1.0       # amplitude of sinusoidal control input
    params = get_motor_parameters('Maxon148877')
    
    # simulate motor with linear+sinusoidal input current
    motor = MotorWElasticity(dt, params)
    N = int(T/dt)   # number of time steps
    q_m = np.zeros(N)
    dq_m = np.zeros(N)
    q_j = np.zeros(N)
    dq_j = np.zeros(N)    
    current = np.zeros(N)
    tau = np.zeros(N)
    V = np.zeros(N)
    for i in range(N):
        t = i*dt
        q_m[i] = motor.q_m()
        dq_m[i] = motor.dq_m()
        q_j[i] = motor.q_j()
        dq_j[i] = motor.dq_j()
        
        current[i] = u_A*np.sin(2*np.pi*u_w*t)
        motor.simulate(current[i], 'time-stepping')
        
        V[i] = motor.V()
        tau[i] = motor.tau()

    # plot joint angle, velocity and torque
    f, ax = plt.subplots(4,1,sharex=True)
    time = np.arange(0.0, T, dt)
    ax[0].plot(time, q_j, label ='joint angle [rad]')
    ax[0].plot(time, q_m, label ='motor angle [rad/s]')
    ax[1].plot(time, dq_j, label ='joint velocity [rad]')
    ax[1].plot(time, dq_m, label ='motor velocity [rad/s]')
    ax[2].plot(time, tau, label ='motor torque [N]')
    ax[3].plot(time, V, label ='voltage [V]')
    for i in range(4): ax[i].legend()
    plt.xlabel('Time [s]')
    plt.show()
