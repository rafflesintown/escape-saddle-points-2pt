import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd

from my_algs import *
from my_f_utils import *
from zo_ncf_algs import *


#PAGD vs ZOPGD 2-pt
plt.close()
np.random.seed(10)

# L = np.e
eps= 1e-3
# gamma = 1.
tau = np.e
d = 10
if d == 10:
  T = 600
elif d == 30:
  T = 1800
n_trials = 30
T_pagd = T
N = T_pagd * 2*d
T_zo = int(N/2.)
mu = 1e-2 # perturbation radius
noise_scale_ZO = 0.05
g_thres = gamma * np.e/100
t_thres = 1
r = np.e/100
x0 = np.random.normal(size = d)
x0 *= eps #try make x0 scale with eps
# x0[:] = x0[0]
x0 = x0.reshape(d,1)
(x_out, f_out) = GD(x0, T)
f_out = np.repeat(f_out,2*d)


fig = plt.figure()
plt.plot(np.arange(N), f_out, label = "GD",c = "blue")



#plot ncf from earlier
x0_ncf = x0.reshape(d,).tolist()
zo_gd_ncf_complexity, zo_gd_ncf_vals = zo_gd_ncf(x0_ncf, T)
n_ncf_vals = len(zo_gd_ncf_vals)
zo_ncf_vals_all = np.empty((n_trials, n_ncf_vals))
zo_ncf_vals_all[0,:] = zo_gd_ncf_vals

zo_gd_ncf_complexity_final = zo_gd_ncf_complexity
for i in np.arange(1,n_trials):
  zo_gd_ncf_complexity, zo_gd_ncf_vals = zo_gd_ncf(x0_ncf, T)
  if len(zo_gd_ncf_vals) < n_ncf_vals:
    extra_vals = np.ones(n_ncf_vals - len(zo_gd_ncf_vals) ) * zo_gd_ncf_vals[-1]
    zo_gd_ncf_vals = np.append(zo_gd_ncf_vals, extra_vals)
  elif len(zo_gd_ncf_vals) > n_ncf_vals:
    new_zo_ncf_vals_all = np.empty((n_trials, len(zo_gd_ncf_vals)))
    for j in np.arange(i):
      zo_ncf_vals_j = zo_ncf_vals_all[j,:]
      extra_vals = np.ones(len(zo_gd_ncf_vals) - n_ncf_vals) * zo_ncf_vals_j[-1]
      new_zo_ncf_vals_all[j,:] = np.append(zo_ncf_vals_j,extra_vals)
    zo_ncf_vals_all = new_zo_ncf_vals_all
    zo_gd_ncf_complexity_final = zo_gd_ncf_complexity
    n_ncf_vals = len(zo_gd_ncf_vals)
  zo_ncf_vals_all[i,:] = zo_gd_ncf_vals

#calculate minimum value (np.ones(d) * 4tau is local minimum, see https://arxiv.org/pdf/1705.10412.pdf)
min_xval = np.ones(d) * (4*tau)
min_fval =  eval_f(min_xval)
print("min f val", min_fval)


plt.plot(zo_gd_ncf_complexity_final, np.mean(zo_ncf_vals_all, axis = 0), label = "ZO-GD-NCF", color = "red")
plt.fill_between(zo_gd_ncf_complexity_final, np.maximum(np.mean(zo_ncf_vals_all,axis = 0) - 1.5 * np.std(zo_ncf_vals_all,axis = 0),min_fval),\
  np.mean(zo_ncf_vals_all,axis = 0) + 1.5 * np.std(zo_ncf_vals_all,axis = 0), color = "red",alpha = .1)


#plot pagd
f_out_PAGD_all  = np.empty((n_trials,N))
for i in np.arange(n_trials):
  (x_out_PAGD,f_out_PAGD) = PAGD(x0,T_pagd,mu,t_thres,g_thres,r)
  f_out_PAGD = np.repeat(f_out_PAGD,2*d)
  f_out_PAGD_all[i,:] = f_out_PAGD
plt.plot(np.arange(N), np.mean(f_out_PAGD_all,axis = 0), label = "PAGD", color = "orange")
plt.fill_between(np.arange(N), np.maximum(np.mean(f_out_PAGD_all,axis = 0) - 1.5 * np.std(f_out_PAGD_all,axis = 0),min_fval), \
  np.mean(f_out_PAGD_all,axis = 0) + 1.5 * np.std(f_out_PAGD_all,axis = 0),color = "orange", alpha = .1)

#plot ZOPGD 2-pt
f_out_ZOPGD_all = np.empty((n_trials,N))
for i in np.arange(n_trials):
  (x_out_ZO, f_out_ZO, ZO_grad_hist, ZO_grad_coef_hist, ZO_noise_hist) = zero_order_noisy(x0,T_zo,mu,noise_scale_ZO)
  f_out_ZO = np.repeat(f_out_ZO,2)
  f_out_ZOPGD_all[i,:] = f_out_ZO

plt.plot(np.arange(N), np.mean(f_out_ZOPGD_all,axis = 0), label = "ZOPGD 2-pt", color = "green")
plt.fill_between(np.arange(N), np.maximum(np.mean(f_out_ZOPGD_all,axis = 0) - 1.5 * np.std(f_out_ZOPGD_all,axis = 0),min_fval), \
  np.maximum(np.mean(f_out_ZOPGD_all,axis = 0) + 1.5 * np.std(f_out_ZOPGD_all,axis = 0),min_fval),color = "green", alpha = .1)

plt.plot(np.arange(N), np.ones(N) * min_fval, label = "minimum function value", color = "brown", linestyle = "dashed")

plt.xlabel('Number of function queries')
plt.ylabel('function value')
plt.legend(loc = "best")
if d == 10:
  plt.xticks(np.arange(0,N,1000))
elif d == 30:
  plt.xticks(np.arange(0,N,5000))
plt.xticks(rotation = 90)
plt.grid()
fig.savefig("pagd_vs_zopgd_vs_ncf_d_%d_ntrials_%d.pdf"%(d,n_trials),bbox_inches="tight")
plt.close()


#save data
df_ncf = pd.DataFrame({"zo_ncf_mean": np.mean(zo_ncf_vals_all,axis = 0), "zo_ncf_std": np.std(zo_ncf_vals_all, axis = 0), \
     "zo_gd_ncf_complexity": zo_gd_ncf_complexity_final})
df_ncf.to_csv("ncf_d_%d_ntrials_%d.csv" % (d,n_trials))

df_rest = pd.DataFrame({"pagd_mean": np.mean(f_out_PAGD_all, axis = 0), "pagd_std": np.std(f_out_PAGD_all,axis = 0), \
     "zopgd_mean": np.mean(f_out_ZOPGD_all,axis = 0), "zopgd_std": np.std(f_out_ZOPGD_all,axis = 0), \
     "min_fval": min_fval * np.ones(N)})
df_rest.to_csv("pagd_zopgd_d_%d_ntrials_%d.csv" % (d,n_trials))

