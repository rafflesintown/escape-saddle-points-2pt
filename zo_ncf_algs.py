import numpy as np

from zo_ncf_utils import construct_g1
from zo_ncf_utils import construct_g2
from zo_ncf_utils import construct_g
from zo_ncf_utils import construct_f

np.random.seed(10)





# construct f
import random
tau = np.exp(1)
L = np.exp(1)
# rho = 0.2 * np.exp(1) / 1e4
rho = 2e-4
gamma = 1

g2 = construct_g2(L, gamma, tau)
g1 = construct_g1(L, gamma, tau)
g = construct_g(g1, g2)
nu = - g1(2 * tau) + 4 * L * tau ** 2

f = construct_f(L, gamma, tau, nu, g, g1)

global function_query
function_query = 0


def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    return x / np.linalg.norm(x)


epsilon = 1e-4


# coordinate-wise gradient estimator
def nabla_ff(x, mu):
    der_x = []
    global function_query
    for i in range(len(x)):
        der_x_i = f(x[:i] + [x[i] + mu] + x[i + 1:]) - f(x[:i] + [x[i] - mu] + x[i + 1:])
        der_x.append(der_x_i / (2 * mu))
    function_query = function_query + 2 * len(x)
    return der_x


# zeroth-other negative curvature finding
def ncf(x):
    # initialization
    d = len(x)
    p = 0.01
    # rho = L / 1e2
    delta = np.sqrt(rho*epsilon)
    C_1 = 0.5
    iterss = np.power(C_1, 2)*np.log(d/p)*np.sqrt(L)/np.sqrt(delta)
    iterss = iterss.astype(int)
    sigma = np.power(d / p, -2*C_1) * delta/np.power(iterss, 2)/rho
    rr = np.power(len(x)/p, C_1)*sigma

    x_0 = np.array(x)
    xi = sigma * get_rand_vec(len(x))
    x = np.array(x) + xi
    y_0 = np.array(np.zeros(len(x)))
    y = xi
    v = np.array(np.ones(len(x)))
    for t in range(iterss):
        # mu = np.linalg.norm(y)
        mu = 1e-3
        M_y = - 1/L*(np.array(nabla_ff(list(x_0+y), mu)) - np.array(nabla_ff(list(x_0), mu))) + (1 - 3*delta/(4*L))*y
        yy = 2 * M_y - y_0
        x = x_0 + yy - M_y

        y_0 = y
        y = yy
        norm_x_x_0 = np.linalg.norm(x - x_0)
        if norm_x_x_0 >= rr:
            v_negative_curvature = (x - x_0)/np.linalg.norm(x - x_0)
            return v_negative_curvature

    v = np.array(np.ones(len(x)))
    return v


def positive_or_negative():
    if random.random() < 0.5:
        return 1
    else:
        return -1


def zo_gd_ncf(x_0, iters):
    x = x_0
    d = len(x)
    values = []
    function_query_complexity = []
    count = 0
    eta = 1 / (3 * L)
    global function_query
    for i in range(iters):
        muu = epsilon/L/np.sqrt(d)/10
        der_x = nabla_ff(x, muu)
        der_x = np.array(der_x)

        x_new = np.array(x) - eta * der_x
        # print(np.linalg.norm(der_x) - (3 / 4) * epsilon)

        if np.linalg.norm(der_x) <= (3 / 4) * epsilon:
            v = ncf(x_new)
            if (v == np.array(np.ones(d))).all():
                # x_new = x_new
                break
            else:
                count = count + 1
                print('times of escaping saddle points: ', count)
                for j in range(d):
                    flag = random.choice((-1, 1))
                    v[j] = flag * v[j]
                x_new = np.array(x_new) + v

        x = list(x_new)
        values.append(f(x))
        function_query_complexity.append(function_query)

    function_query = 0
    return function_query_complexity, values


########################################################################################################################
# gaussian gradient estimator another version
def nabla_ff_gaussian_1(f, x, sigma):
    global function_query
    u = np.random.normal(0, sigma**2, len(x))
    der_x = (f(x + u) - f(x)) / sigma**2 * u
    function_query = function_query + 2

    return der_x


# uniform distribution over a ball with radius
def uniform_distribution_over_unit_ball(x_0, r):
    ratio = np.random.uniform(0, 1, 1)
    xi = np.random.uniform(-1, 1, len(x_0))
    # xi = np.random.uniform(-1, 1, len(x_0))
    xi = xi / np.linalg.norm(xi) * r * ratio[0:1]

    return xi


# zeroth-order perturbed sgd
def zpsgd(x_0, iters, batch_size):
    d = len(x_0)
    x = x_0
    sigma = np.sqrt(epsilon / (rho * d))
    r = epsilon
    eta = 1 / (4 * L)
    x_list = []
    values = []
    function_query_complexity = []
    global function_query
    for t in range(iters):
        if t % 100 ==0:
            print(t)
        x_list.append(x)
        values.append(f(x))
        function_query_complexity.append(function_query)

        g = np.zeros(d)
        for i in range(batch_size):
            g = g + np.array(nabla_ff_gaussian_1(f, list(x), sigma) / batch_size)

        xi = uniform_distribution_over_unit_ball(x, r)
        x = list(np.array(x) - eta * (g + xi))

    print(x)
    function_query = 0

    return function_query_complexity, values


########################################################################################################################

def nabla_f(x):
    h = 0.01
    der_x = []
    global function_query
    for i in range(len(x)):
        der_x_i = f(x[:i] + [x[i] + h] + x[i + 1:]) - f(x[:i] + [x[i] - h] + x[i + 1:])
        der_x.append(der_x_i / (2 * h))
    function_query = function_query + 2 * len(x)
    return der_x





def experiment_pagd(x_0, iters):
    t_thresh = -1
    t_noise = - t_thresh - 1
    g_thresh = np.exp(1) * gamma / 100
    r = np.exp(1) / 100
    x = x_0
    d = len(x)
    eta = 1 / (4 * L)
    t_noise = - t_thresh - 1
    values = []
    function_query_complexity = []
    global function_query

    for i in range(iters):
        if i % 100 == 0:
            print(i)
        der_x = nabla_f(x)
        der_x = np.array(der_x)

        if (3 / 4) * g_thresh >= np.linalg.norm(der_x) and (i - t_noise > t_thresh):
            noise = r * get_rand_vec(len(x))
            x = np.array(x) + noise
            x = list(x)
            der_x = nabla_f(x)
            der_x = np.array(der_x)
            t_noise = i

        x_new = np.array(x) - eta * der_x
        x = list(x_new)
        values.append(f(x))
        function_query_complexity.append(function_query)

    print(x)
    function_query = 0
    return function_query_complexity, values


########################################################################################################################
def dfpi(x):
    d = len(x)
    s = np.random.uniform(-1, 1, d)
    s = s / np.linalg.norm(s)
    # T_dfpi = int(1 / epsilon**(2/3) * L * np.log(d))
    T_dfpi = 20
    r = 0.001
    c = 0.001
    for t in range(T_dfpi):
        g_pos = np.array(nabla_ff(list(x + r * s), c))
        g_neg = np.array(nabla_ff(list(x - r * s), c))
        eta = 1 / L
        s = s - eta * (g_pos - g_neg) / (2 * r)
        s = s / np.linalg.norm(s)

    return s


def rspi(x_0, iters, sigma_1, sigma_2, T_sigma_1, ratio):
    x = x_0
    d = len(x_0)
    values = []
    function_query_complexity = []
    global function_query
    t_ncf = 0
    for k in range(iters):
        print(k)

        s_1 = np.random.uniform(-1, 1, d)
        s_1 = s_1 / np.linalg.norm(s_1)
        x_current = x
        x_forward = x + sigma_1 * s_1
        x_backward = x - sigma_1 * s_1
        list_value_1 = [f(list(x_current)), f(list(x_forward)), f(list(x_backward))]
        index = np.argmin(list_value_1)
        function_query = function_query + 3
        if index == 0:
            # x = x_current
            t_ncf += 1
            print('times of ncf:', t_ncf)
            s_2 = dfpi(x)
            x_current = x
            x_forward = x + sigma_2 * s_2
            x_backward = x - sigma_2 * s_2
            list_value_2 = [f(list(x_current)), f(list(x_forward)), f(list(x_backward))]
            index = np.argmin(list_value_2)
            if index == 0:
                x = x_current
            elif index == 1:
                x = x_forward
            else:
                x = x_backward
        elif index == 1:
            x = x_forward
        else:
            x = x_backward

        if k % T_sigma_1 == 0 and sigma_1 > 0.5:
            sigma_1 = sigma_1 * ratio
        x = list(x)
        values.append(f(x))
        function_query_complexity.append(function_query)
    print(x)
    function_query = 0
    return function_query_complexity, values