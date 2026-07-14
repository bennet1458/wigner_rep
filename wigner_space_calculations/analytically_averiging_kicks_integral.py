import numpy as np
import matplotlib.pyplot as plt
import momentum_distribution_funcs as md
import wigner_functions as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)

t_2 = 150e-6 #150.0e-6   # time after second kick
q_av = 10e-23#10e-23   # larger q → clearer interference
q = q_av

##########################################
epsilon_0 = 8.854187817e-12  # vacuum permittivity
e = 1.602176634e-19  # elementary charge
c = 2.99792458e8  # speed of light

Q = 500*e  # particle charge
k_e = 1/(4*np.pi*epsilon_0)  # Coulomb's constant
v = 2e4
A = 2*k_e*e*Q/v

q_av = 10e-23
b_min = 50e-9/2
q_max = A/b_min

# grid
res_x = 1500
res_b = 100
plotwidth = 60
x = np.linspace(-4*sigma_x+(q-plotwidth*sigma_p)*(t_2)/m, 4*sigma_x+(q+plotwidth*sigma_p)*(t_1+t_2)/m, res_x)
w = 2

def p_marginal_t1(q_1, q_2):
    alpha = q_2*t_1/m
    beta = (t_1+t_2)/m
    gamma = q_1+q_2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+b**2/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t2(q_1, q_2):
    alpha = q_2*t_1/(2*m)
    beta = (t_1+t_2)/m
    gamma = q_1/2+q_2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    b_1 = -q_1*(t_1+t_2)/(hbar*m)
    c_1 = q_1/hbar*(x+q_2*t_1/m)
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/2*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+(b**2-b_1**2)/(4*a))
    cos = np.cos(b*b_1/(2*a)+c_1)
    func = wfactor*ffactor*exp*cos
    return func

def p_marginal_t3(q_1, q_2):
    alpha = q_2*t_1/(2*m)
    beta = (t_1+t_2)/m
    gamma = q_1+q_2/2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)
    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/2*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+(b**2-b_1**2)/(4*a))
    cos = np.cos(b*b_1/(2*a)+c_1)
    func = wfactor*ffactor*exp*cos
    return func

def p_marginal_t4(q_1, q_2):
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*x + q_1/(sigma_p**2)
    c = -1/(2*sigma_x**2)*x**2 - q_1**2/(2*sigma_p**2)
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t5(q_1, q_2):
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/m) + q_2/(sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/m)**2 - q_2**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func

def p_marginal_t6(q_1, q_2):
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/(2*m)) + (q_1+q_2)/(2*sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/(2*m))**2 - (q_1+q_2)**2/(8*sigma_p**2)
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

def p_marginal_t7(q_1, q_2):
    alpha = 0
    beta = (t_1+t_2)/m
    gamma = q_1/2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)
    b_1 = -q_1*(t_1+t_2)/(hbar*m)
    c_1 = q_1*x/hbar
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/2*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+(b**2-b_1**2)/(4*a))
    cos = np.cos(b*b_1/(2*a)+c_1)
    func = wfactor*ffactor*exp*cos
    return func

def p_marginal_t8(q_1, q_2):
    alpha = q_2*t_1/(2*m)
    beta = (t_1+t_2)/m
    gamma = q_2/2
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)
    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/2*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+(b**2-b_1**2)/(4*a))

    cos = np.cos(b*b_1/(2*a)+c_1)
    func = wfactor*ffactor*exp*cos
    return func

def p_marginal_t9(q_1, q_2):
    alpha = 0
    beta = (t_1+t_2)/m
    gamma = 0
    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)
    exp = np.exp(c+b**2/(4*a))
    func = wfactor*ffactor*exp
    return func



def entire_marginal(q_1, q_2):
    sum = (0
           +p_marginal_t1(q_1, q_2)
        #    +p_marginal_t2(q_1, q_2)
        #    +p_marginal_t3(q_1, q_2)
           +p_marginal_t4(q_1, q_2)
           +p_marginal_t5(q_1, q_2)
           +p_marginal_t6(q_1, q_2)
        #    +p_marginal_t7(q_1, q_2)
        #    +p_marginal_t8(q_1, q_2)
           +p_marginal_t9(q_1, q_2)
           )

    return sum


########################################

def gauss_dist(b_1, b_2, mu_b, sigma_b):
    factor = 1/(2*np.pi*sigma_b**2)
    return np.exp(-((b_1-mu_b)**2+(b_2-mu_b)**2)/(2*sigma_b**2))


def avg_marginal(sigma_b_dash):
    print('q_av')
    print(q_av)
    mu_b = md.find_mu(sigma_b_dash, b_min=b_min, q_av=q_av, A=A)
    print(A/mu_b)
    
    sigma_b = sigma_b_dash*mu_b
    b = np.linspace(mu_b-sigma_b, mu_b+sigma_b, res_b)

    b_1 = b[:, np.newaxis, np.newaxis]
    b_2 = b[np.newaxis, :, np.newaxis]
    q_1 = A/b_1
    q_2 = A/b_2
    marginal = entire_marginal(q_1, q_2)
    gauss = gauss_dist(b_1, b_2, mu_b, sigma_b)
    integrant = marginal*gauss
    result = np.sum(integrant, axis=(0, 1))/res_b

    return result



fig, axes = plt.subplots(2, 1, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0].plot(x, entire_marginal(q_av, q_av), linewidth=2, linestyle='-', label='exact kick')
axes[0].set_xlabel("x")
axes[0].set_ylabel("Marginal")
axes[0].legend()


sigma_b_dash_values = [1/3, 1/4, 1/6, 1/8, 1/20]
visibility_values = np.zeros(len(sigma_b_dash_values))
for i, sigma_b_dash in enumerate(sigma_b_dash_values):
    marginal = avg_marginal(sigma_b_dash)
    visibility_values[i] = wf.modulation_depth(marginal, w=w)
    axes[1].plot(x, marginal, linewidth=2, linestyle='-', label='kick distribution with $\sigma_b$ = 1/{:.0f}, visibility = {:.2f}'.format(1/sigma_b_dash, visibility_values[i]))

axes[1].set_xlabel("x")
axes[1].set_ylabel("Marginal")
axes[1].legend()

plt.tight_layout()
plt.show()

