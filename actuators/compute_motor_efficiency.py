from __future__ import print_function
from dc_motor_w_coulomb_friction import MotorCoulomb
from dc_motor_w_elasticity import MotorWElasticity
from dc_motor_w_elasticity import get_motor_parameters
import arc.utils.plot_utils
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty:
    def __init__(self):
        pass
    
def compute_efficiency(V, motor, dq_max):
    res = Empty()
    res.dq = np.arange(0.0, dq_max, 0.01)
    N = res.dq.shape[0]
    res.tau_f = np.zeros(N)         # friction torque (Coulomb+viscous)
    res.tau = np.zeros(N)           # output torque (motor+friction)
    res.P_m = np.zeros(N)           # mechanical output power
    res.P_e = np.zeros(N)           # electrical input power
    res.efficiency = np.zeros(N)    # motor efficiency (out/in power)
    
    for i in range(N):  
        if motor.__class__.__name__ == 'MotorCoulomb':
            res.model = 'MotorCoulomb'
            state = np.zeros(2)
            state[1] = res.dq[i]                                 # set motor velocity
            motor.set_state(state)
            motor.simulate_voltage(V, method='time-stepping')    # apply constant voltage
            res.tau_f[i] = motor.tau_coulomb() + res.dq[i] * motor.b
            res.tau[i] = motor.tau() - res.tau_f[i]
            res.P_m[i] = res.tau[i] * res.dq[i]
            res.P_e[i] = motor.i() * V 
            res.efficiency[i] = res.P_m[i] / res.P_e[i]
        elif motor.__class__.__name__ == 'MotorWElasticity':
            res.model = 'MotorWElasticity'
            state = np.zeros(4)
            state[1] = res.dq[i]                                 # set motor velocity
            state[3] = res.dq[i]                                 # set joint velocity (steady state)                            
            motor.set_state(state)
            motor.simulate_voltage(V, method='time-stepping')    # apply constant voltage
            res.tau_f[i] = motor.tau_f_m + motor.tau_f_j + res.dq[i] * (motor.b_m + motor.b_j) 
            res.tau[i] = motor.tau() - res.tau_f[i]
            res.P_m[i] = res.tau[i] * res.dq[i]
            res.P_e[i] = motor.i() * V 
            res.efficiency[i] = res.P_m[i] / res.P_e[i]
    
    i = np.argmax(res.efficiency)
    print("Max efficiency", res.efficiency[i])
    print("reached at velocity", res.dq[i], "and torque", res.tau[i])
    return res

V = 48                      # input voltage
dt = 1e-5                   # time step
params = get_motor_parameters('Maxon148877')

motor = MotorCoulomb(dt, params)
motor_w_elasticity = MotorWElasticity(dt, params)

dq_max = V / motor.K_b        # maximum motor vel for given voltage

res = compute_efficiency(V, motor, dq_max)
res_elast = compute_efficiency(V, motor_w_elasticity, dq_max)

def plot_stuff(res):
    f, ax = plt.subplots(1,1,sharex=True)
    alpha = 0.8
    ax.plot(res.tau, res.dq, label ='motor velocity', alpha=alpha)
    ax.plot(res.tau, res.P_m, label ='P_m', alpha=alpha)
    ax.plot(res.tau, res.P_e, label ='P_e', alpha=alpha)
    dq_max = np.max(res.dq)
    ax.plot(res.tau, res.efficiency * dq_max, label ='efficiency (scaled)', alpha=alpha)
    ax.legend()
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Velocity [rad/s]')
    plt.ylim([0, dq_max])
    if res.model == "MotorCoulomb":
        plt.title('Rigid transmission')
    elif res.model == 'MotorWElasticity':
        plt.title('Elastic transmission (steady state)')
    else:
        print("ERROR: unknown motor model") 


if __name__=='__main__':
    plot_stuff(res)
    plot_stuff(res_elast)

    f, ax = plt.subplots(1,1,sharex=True)
    alpha = 0.8
    ax.plot(res.tau, res.efficiency, label ='efficiency (rigid transm.)', alpha=alpha)
    ax.plot(res_elast.tau, res_elast.efficiency, label ='efficiency (elastic transm.)', alpha=alpha)
    ax.legend()
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Efficiency')
    plt.ylim([0, 1])
    plt.show()
