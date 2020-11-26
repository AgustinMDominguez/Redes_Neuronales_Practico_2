# Agustin Marcelo Dominguez - Nov 2020

def line(ch = '-', msg=''):
    for _ in range(80):
        print(ch, end='')
    print('\n\t' + msg)

line(msg="loading libraries...")
import random
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')
line(msg="Starting...")


# MODEL PARAMETERS
global_E_L = -65.0    # Rest
global_TAU_m = 10
global_R_m = 10
global_V_threshold = -50
global_V_0 = global_E_L
global_start = 0
global_end = 200

# CONSTANTS
global_samples = 500


def V_m(I_e, t, E_L=global_E_L, TAU_m=global_TAU_m, R_m=global_R_m, V_0=global_V_0):
    aux1 = E_L + R_m*I_e
    aux2 = V_0 - E_L - R_m*I_e
    aux3 = np.exp(-t/TAU_m)
    return aux1 + aux2*aux3

def period(I_e, E_L=global_E_L, TAU_m=global_TAU_m, R_m=global_R_m, V_threshold=global_V_threshold, V_0=global_V_0):
    """Returns -1 if infinite or the period"""
    rmIe = I_e * R_m
    if(E_L + rmIe <= V_threshold):
        return -1
    else:
        aux1= rmIe + E_L - V_threshold
        return -TAU_m * (np.log(aux1 / rmIe))

def getAnalyticSolution(I_e, start=global_start, end=global_end):
    print(f"\tObtaining Analytic Solution for I={I_e}")
    sol_x = []
    sol_y = []
    for i in range(global_samples):
        t = i * end / global_samples
        sol_x.append(t)
        sol_y.append(V_m(I_e, t))
    return (np.array(sol_x), np.array(sol_y))


def plotAnalyticSolutions(energy_inputs, start=global_start, end=global_end, save=False):
    _, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(start - 0.4, end)
    ax.set_ylim(-67, -40)
    ax.set_ylabel('Vm')
    ax.set_xlabel('time [ms]')
    ax.hlines(global_V_threshold, start, end, linestyles="dashed")

    for I_e in energy_inputs:
        sol = getAnalyticSolution(I_e, start, end)
        ax.plot(sol[0], sol[1], label="Input="+str(I_e)+"nA")
        ax.plot(period(I_e), global_V_threshold, '.', color="#000000")
    ax.legend(loc="lower right", ncol=2)
    if save:
        filename = 'analyticSolution.png'
        plt.savefig(filename, dpi=200)
        print("\tSaved "+filename)
    else:
        plt.show()


def plotFrequency(start, end, save=False, show=True):
    line(msg="Ploting Frequency graph...")
    sol_x = []
    sol_y = []
    for i in range(global_samples):
        I_e = i * end / global_samples
        sol_x.append(I_e)
        T = period(I_e)
        frequency = 0.0 if T==-1 else 1/T
        sol_y.append(frequency)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(start, end)
    ax.set_ylabel('frequency 1/T[ms]')
    ax.set_xlabel('I')
    ax.plot(np.array(sol_x), np.array(sol_y))
    x = 1.6
    step = (end-start)/5 # We will plot 5 different numeric values
    for _ in range(5):
        y = getNumericalPeriod(x, 80, 0.01)
        ax.plot(x, 1/y, '.', color="#000000")
        x += step
    if save:
        filename = 'frequencyPlot.png'
        plt.savefig(filename, dpi=200)
        print("\tSaved "+filename)
    else:
        plt.show()


def dvdt(t, v, I_e, E_L=global_E_L, TAU_m=global_TAU_m, R_m=global_R_m):
    return (E_L - v + R_m*I_e)/TAU_m

def rungeKuttaStep(t0, v0, step, I_e, E_L=global_E_L, TAU_m=global_TAU_m, R_m=global_R_m):
    """
    Finds value of v for a given t using step size h, and initial value v0 at y0. Code adapted from Prateek Bhindwar 
    """
    k1 = step * dvdt(t0, v0, I_e) 
    k2 = step * dvdt(t0 + 0.5 * step, v0 + 0.5 * k1, I_e)
    k3 = step * dvdt(t0 + 0.5 * step, v0 + 0.5 * k2, I_e) 
    k4 = step * dvdt(t0 + step, v0 + k3, I_e) 
    res = v0 + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # print(f"Got v0={v0} and by moving {step} i got {res}")
    return res

def getRK4Solution(I_e, end, step):
    print(f"Obtaining RK4 Solution for I={I_e}")
    sol_x = []
    sol_y = []
    t = 0
    v = -65
    while(t < end):
        if (v >= -50):
            v = global_E_L
            continue
        sol_x.append(t)
        sol_y.append(v)
        v = rungeKuttaStep(t,v,step,I_e)
        t += step
    return (np.array(sol_x), np.array(sol_y))

def getNumericalPeriod(I_e, samples, step):
    print(f"\tObtaining Period with RK4 for I={I_e}")
    sol = []
    t = 0
    v = -65
    prev = 0
    amount_samples = 0
    while(amount_samples < samples):
        if (v >= -50):
            amount_samples += 1
            sol.append(t-prev)
            prev = t
            v = global_E_L
            continue
        v = rungeKuttaStep(t,v,step,I_e)
        t += step
    return sum(sol)/samples

def plotComparisonAnalyticVsRK4(I_e, end, save=False):
    line(msg="Ploting Comparison between Analytic solution and RK4 solution...")
    _, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-3, end)
    ax.set_ylabel('Vm')
    ax.set_xlabel('Ie')
    anSolution = getAnalyticSolution(I_e, 0, end)
    rk4Solution = getRK4Solution(I_e, end, 0.01)
    ax.plot(anSolution[0], anSolution[1], label=f"Analytic I={I_e}nA")
    ax.plot(rk4Solution[0], rk4Solution[1], label=f"Runge Kutta I={I_e}nA")
    ax.legend(loc="lower center", ncol=2)
    if save:
        filename = 'numericalAnalyticComparison.png'
        plt.savefig(filename, dpi=200)
        print("\tSaved "+filename)
    else:
        plt.show()

def printEnergyinputsInfo(energy_inputs):
    for I_e in energy_inputs:
        per = period(I_e)
        T = "inf" if per==-1 else str(per)
        f = "0"   if per==-1 else str(1.0/per)
        if(I_e <= 1.5):
            numerical_period = "inf"
        else:
            numerical_period = getNumericalPeriod(I_e, 50, 0.01)
        print(f"\tI_e = {I_e}\n\t Analytic_Period = {T} - Freq = {f} \n\t Numerical_Period = {numerical_period}")

def plotRK4Comparison(energy_inputs, end, save=False):
    _, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-3, end)
    ax.set_ylabel('Vm')
    ax.set_xlabel('time')
    for I_e in energy_inputs:
        sol = getRK4Solution(I_e, end, 0.02)
        ax.plot(sol[0], sol[1], label=f"I={I_e}")
    ax.legend(loc="center", ncol=2)
    if save:
        filename = 'differentI_eComparison.png'
        plt.savefig(filename, dpi=200)
        print("\tSaved "+filename)
    else:
        plt.show()

def getRandomEnergyInputs(samples, mode):
    I_e_Arr = []
    # CONSTANTS
    MODE_1_BRIDGE_STEP = 0.03
    SIGMA_SPREAD = 2.0
    CHANGE_TWITCH = 0.03 # a rough percentage of how often should the value jump
    EASE_RATE = 2
    NOISE_SIGMA = 0.1
    SMALL_NOISE_SIGMA = 0.04
    """
    MODE 0:
        Floating numbers where every sample is pulled from an Uniform Distribution between 0.0 and 5.0
    MODE 1:
        Floating numbers where some samples are pulled from a Uniform Distribution between 0.0 and 4.0
        except that it will not jump immediately but will sample linearly towards that direction
        with each step moving MODE_1_BRIDGE_STEP
    MODE 2:
        It will generate 2 random numbers. The first is a certain amount of samples, and the other a value between 0.0 and 5.0
        Then it will put that amount of samples in a row with the value obtained. The result is a stair-like graph
        The Value is still pulled from a Uniform Distribution
    MODE 3:
        Same as MODE2 but the distribution used to obtain the value is a Gaussian one more centered on 1.5 values
    MODE 4:
        Same as MODE3 but it will ease linearly its way in the jump
    MODE 5:
        Same as MODE4 but it adds some noise to the flat parts
    MODE 6:
        Starts on 1.5 and then randomizes a small step in eash RK4 step
    ELSE:
        Chooses a random constant value

    """
    if(mode==0):
        for _ in range(samples):
            I_e_Arr.append(random.uniform(0,5))

    elif(mode==1):
        prev = 0
        samples_produced = 0
        while(samples_produced < samples):
            value = random.uniform(0,4)
            if (abs(prev-value) < MODE_1_BRIDGE_STEP):
                I_e_Arr.append(value)
                samples_produced += 1
            else:
                direction = -1 if prev > value else 1
                while(prev*direction < value*direction):
                    prev += direction*MODE_1_BRIDGE_STEP
                    I_e_Arr.append(prev)
                    samples_produced += 1
                prev = value

    elif(mode==2):
        mu = int(samples*CHANGE_TWITCH)
        sigma = int((mu-1)/SIGMA_SPREAD)
        samples_produced = 0
        while(samples_produced < samples):
            ran = max(1, int(random.gauss(mu, sigma))) 
            value = random.uniform(0,4)
            for _ in range(ran):
                I_e_Arr.append(value)
            samples_produced += ran

    elif(mode==3):
        mu = int(samples*CHANGE_TWITCH)
        sigma = int((mu-1)/SIGMA_SPREAD)
        samples_produced = 0
        while(samples_produced < samples):
            ran = max(1, int(random.gauss(mu, sigma))) 
            value = max(0,random.gauss(1.5,0.5))
            for _ in range(ran):
                I_e_Arr.append(value)
            samples_produced += ran

    elif(mode==4):
        mu = int(samples*CHANGE_TWITCH)
        sigma = int((mu-1)/SIGMA_SPREAD)
        samples_produced = 0
        prev = 0
        while(samples_produced < samples):
            ran = max(1, int(random.gauss(mu, sigma))) 
            value = max(0,random.gauss(1.6,1.5))
            substep = (value-prev)/(ran/EASE_RATE)
            for _ in range(int(ran/EASE_RATE)):                
                prev += substep
                I_e_Arr.append(prev)
            prev = value
            for _ in range(ran - int(ran/EASE_RATE)):                
                I_e_Arr.append(value)
            samples_produced += ran

    elif(mode==5):
        mu = int(samples*CHANGE_TWITCH)
        sigma = int((mu-1)/SIGMA_SPREAD)
        samples_produced = 0
        prev = 0
        while(samples_produced < samples):
            ran = max(1, int(random.gauss(mu, sigma))) 
            value = max(0,random.gauss(1.6,1.5))
            substep = (value-prev)/(ran/EASE_RATE)
            for _ in range(int(ran/EASE_RATE)):                
                prev += substep
                I_e_Arr.append(random.gauss(prev, SMALL_NOISE_SIGMA))
            prev = value
            for _ in range(ran - int(ran/EASE_RATE)):                
                I_e_Arr.append(max(0, random.gauss(value, NOISE_SIGMA)))
            samples_produced += ran
    
    elif(mode==6):
        value = 1.5
        samples_produced = 0
        min_value = 0
        max_value = 5.0
        while(samples_produced < samples):
            samples_produced += 1
            I_e_Arr.append(value)
            value = min(max_value, max(min_value, value + random.gauss(0, 0.02)))

    else:
        value = max(0,random.gauss(1.6,1.5))
        for _ in range(samples):
            I_e_Arr.append(value)

    return I_e_Arr[:samples]

def plotRandomCharge(end, step, mode=0, save=False):
    line(msg=f"Creating random input with mode {mode}")
    samples = int(end/step) + 10
    I_e_Arr = getRandomEnergyInputs(samples, mode)
    sol_x = []
    sol_y = []
    t = 0
    v = -65
    input_index = 0
    while(t < end):
        if (v >= -50):
            v = global_E_L
            continue
        sol_x.append(t)
        sol_y.append(v)
        v = rungeKuttaStep(t,v,step,I_e_Arr[input_index])
        input_index += 1
        t += step
    solution = (np.array(sol_x), np.array(sol_y))
    _, axes = plt.subplots(figsize=(10, 7), ncols=1, nrows=2)
    ax = axes[1]
    ax.set_xlim(-3, end)
    ax.set_ylabel('Vm')
    ax.set_xlabel('time')
    random_ax = axes[0]
    random_ax.set_xlim(-3, end)
    random_ax.set_ylabel('Extern Input')
    random_ax.set_xlabel('time')
    random_ax.hlines(1.5, 0, end, linestyles="dashed")
    ax.plot(solution[0], solution[1])
    random_ax.plot(solution[0],np.array(I_e_Arr[:len(solution[0])]))
    plt.tight_layout()
    if save:
        filename = f'randomInput_mode{mode}.png'
        plt.savefig(filename, dpi=200)
        print("\tSaved "+filename)
    else:
        plt.show()

def plotAllRandomModes(save=True):
    for mode in range(7):
        plotRandomCharge(300, 0.02, mode=mode, save=save)


many_energy_inputs = [
    0.5,
    1.1,
    1,6,
    1,8,
    2.0,
    2,5,
    3.0,
    4.0,
    5.0,
    6.0
]

limited_energy_inputs = [
    0.5,
    1.1,
    2.0,
    3.0,
    4.0,
]

alternative_energy_inputs = {
    0.5,
    1.1,
    1.501,
    1.51,
    2.0
}


printEnergyinputsInfo(many_energy_inputs)
plotAnalyticSolutions(limited_energy_inputs, 0, 40, save=True)
plotFrequency(1.3, 3.4, save=True)
plotComparisonAnalyticVsRK4(2, global_end, save=True)
plotRK4Comparison(alternative_energy_inputs, global_end, save=True)
plotAllRandomModes(save=True)