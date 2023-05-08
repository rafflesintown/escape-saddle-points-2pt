
import numpy as np

np.random.seed(10)

L = np.e
gamma = 1.
tau = np.e

def get_grad_f(x):
    d = len(x)
    x_abs = np.abs(x)
    i_first = d #first i such that |x(i)| <= tau
    f_x = 0
    nu = -g1(2 * tau) + 4*L * tau**2
    for i in np.arange(d):
        if x_abs[i] <= 2 * tau:
            i_first = i
            break

    #print i_first

    grad_f = np.zeros((d,))
    for j in np.arange(i_first): #this is done in all cases
        if x[j] >= 0:
            grad_f[j] = 2. * L * (x[j] - 4. * tau)
        else: 
            grad_f[j] = 2. * L * (x[j] + 4. * tau)

    if i_first <= (d-1) -1:
        if x_abs[i_first] <= tau: #case 1
            grad_f[i_first] = -2. * gamma * x[i_first]
            for j in np.arange(i_first+1, d):
                grad_f[j] = 2. * L * x[j]
                #print j, grad_f[j]
        elif tau <= x_abs[i_first] <= 2*tau: #case 2
            #print i_first, x[i_first], "i_first < d-1, i am splining"
            if x[i_first] >= 0:
                grad_f[i_first] = g1prime(x[i_first]) + x[i_first+1]**2 * g2prime(x[i_first])
                grad_f[i_first + 1] = 2. * x[i_first+1] * g2(x[i_first]) 
            else:
                grad_f[i_first] = -g1prime(-x[i_first]) - x[i_first+1]**2 * g2prime(-x[i_first])
                grad_f[i_first + 1] = 2. * x[i_first+1] * g2(-x[i_first])
            for j in np.arange(i_first+2, d):
                grad_f[j] = 2. * L * x[j]
    elif i_first == d-1:
        if x_abs[i_first] <= tau: #case 1
            grad_f[i_first] = -2. * gamma * x[i_first]
        elif tau <= x_abs[i_first] <= 2*tau: #case 2
            if x[i_first] >= 0:
                grad_f[i_first] = g1prime(x[i_first])
            else:
                grad_f[i_first] = -g1prime(-x[i_first])
    else: #i_first == d-1
        pass # do nothing     
    return grad_f



def eval_f(x):
    # we split up the domain into chunks like in
    # the paper
    d = len(x)
    x_abs = np.abs(x)
    i_first = d #first i such that |x(i)| <= tau
    f_x = 0
    nu = -g1(2 * tau) + 4*L * tau**2
    for i in np.arange(d):
        if x_abs[i] <= 2 * tau:
            i_first = i
            break

    for j in np.arange(i_first): #this is done in all cases
        if x[j] >= 0:
            f_x += L * (x[j] - 4.0 * tau)**2
        else: 
            f_x += L * (x[j] + 4.0 * tau)**2 

    if i_first <= (d-1) -1:
        if x_abs[i_first] <= tau: #case 1
            f_x -= gamma * x[i_first]**2 
            for j in np.arange(i_first+1, d):
                f_x += L * x[j]**2 
        elif tau <= x_abs[i_first] <= 2*tau: #case 2
            if x[i_first] >= 0:
                f_x += g(x[i_first], x[i_first+1])
            else:
                f_x += g(-x[i_first], x[i_first + 1])
            for j in np.arange(i_first+2, d):
                f_x += L * x[j]**2 
    elif i_first == d-1:
        if x_abs[i_first] <= tau: #case 1
            f_x -= gamma * x[i_first]**2
        elif tau <= x_abs[i_first] <= 2*tau: #case 2
            if x[i_first] >= 0:
                f_x += g1(x[i_first])
            else:
                f_x += g1(-x[i_first])
    else: #i_first == d-1
        pass # do nothing           

    f_x -= i_first * nu
    return f_x





def anti_der_p(xi):
    c0 = -2. * gamma * tau
    c1 = -2. * gamma
    S = (-4. * L * tau - (-2.*gamma * tau))/ (tau)
    c2 = (3 * S - 2. * L - 2. *(-2. * gamma))/(tau)
    c3 = - (2. * S - 2. * L - (-2. * gamma))/(tau)**2
    delta_x = (xi - tau)
    return c0*delta_x + c1 * delta_x**2/2. + c2 * delta_x**3/3. + c3 * delta_x**4/4.


def g1(xi):
    return anti_der_p(xi) - anti_der_p(tau)- gamma * tau**2

def g2(xi):
    ratio1 = (10. * (L + gamma) *(xi - 2. * tau)**3)/tau**3
    ratio2 = (15. * (L + gamma) * (xi - 2*tau)**4)/tau**4
    ratio3 = (6. * (L + gamma) * (xi - 2*tau)**5)/tau**5
    return -gamma - ratio1 - ratio2 - ratio3

def g(xi1,xi2):
    return g1(xi1) + g2(xi1) * xi2**2

def g1prime(xi):
    c0 = -2. * gamma * tau
    c1 = -2. * gamma
    S = (-4. * L * tau - (-2.*gamma * tau))/ (tau)
    c2 = (3 * S - 2. * L - 2. *(-2. * gamma))/(tau)
    c3 = - (2. * S - 2. * L - (-2. * gamma))/(tau)**2
    delta_x = (xi -  tau)
    return c0 + c1*delta_x + c2*delta_x**2 + c3 * delta_x**3

def g2prime(xi):
    ratio1_der = 3 * (10. * (L + gamma) *(xi - 2. * tau)**2)/tau**3
    ratio2_der = 4 * (15. * (L + gamma) * (xi - 2*tau)**3)/tau**4
    ratio3_der = 5 * (6 * (L + gamma) * (xi - 2*tau)**4)/tau**5
    return -ratio1_der - ratio2_der - ratio3_der


