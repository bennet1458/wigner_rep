import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)


def p_marginal_t1(x, q_1, q_2, t_0, t_1, t_2):
    alpha = (q_1*t_0+q_2*(t_0+t_1))/m
    beta = (t_0+t_1+t_2)/m
    gamma = q_1+q_2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+b**2/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t4(x, q_1, q_2, t_0, t_1, t_2):
    alpha = q_2*(t_0+t_1)/m
    beta = (t_0+t_1+t_2)/m
    gamma = q_2

    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t5(x, q_1, q_2, t_0, t_1, t_2):
    alpha = q_1*t_0/m
    beta = (t_0+t_1+t_2)/m
    gamma = q_1

    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t6(x, q_1, q_2, t_0, t_1, t_2):
    alpha = q_1*t_0/m+q_2*(t_0+t_1)/(2*m)
    beta = (t_0+t_1+t_2)/m
    gamma = q_1/2+q_2/2

    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)


    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    b_2 = -q_1*(t_1+t_2)/(hbar*m)
    c_2 = q_1/hbar*(x+q_2*t_1/(2*m))
    ffactor = 1/2*np.sqrt(np.pi/a)
    wfactor = 1/(2*np.pi*sigma_x*sigma_p)

    inner_exp_1 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) - b_1*b_2/(2*a))
    inner_exp_2 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) + b_1*b_2/(2*a))
    cos1 = np.cos(b*(b_1+b_2)/(2*a)+c_1+c_2)
    cos2 = np.cos(b*(b_1-b_2)/(2*a)+c_1-c_2)

    func = wfactor*ffactor*(0
        # +inner_exp_1*cos1
        +inner_exp_2*cos2
        )

    return func

def p_marginal_t9(x, q_1, q_2, t_0, t_1, t_2):
    alpha = 0
    beta = (t_0+t_1+t_2)/m
    gamma = 0
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+b**2/(4*a))
    func = wfactor*ffactor*exp
    return func



def entire_marginal(x, q_1, q_2, t_0, t_1, t_2):
    sum = (0
           +p_marginal_t1(x, q_1, q_2, t_0, t_1, t_2)
        #    +p_marginal_t2(q_1, q_2, t_0, t_1, t_2)
        #    +p_marginal_t3(q_1, q_2, t_0, t_1, t_2)
           +p_marginal_t4(x, q_1, q_2, t_0, t_1, t_2)
           +p_marginal_t5(x, q_1, q_2, t_0, t_1, t_2)
           +p_marginal_t6(x, q_1, q_2, t_0, t_1, t_2)
        #    +p_marginal_t7(q_1, q_2, t_0, t_1, t_2)
        #    +p_marginal_t8(q_1, q_2, t_0, t_1, t_2)
           +p_marginal_t9(x, q_1, q_2, t_0, t_1, t_2)
           )

    return sum

def gauss_dist(b_1, b_2, mu_b, sigma_b):
    factor = 1/(2*np.pi*sigma_b**2)
    return np.exp(-((b_1-mu_b)**2+(b_2-mu_b)**2)/(2*sigma_b**2))

