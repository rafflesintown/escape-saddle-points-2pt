import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

from my_f_utils import *

np.random.seed(10)

#this implements the PAGD algorithm in the Flokas paper

def PAGD(x0,T,mu,t_thres,g_thres,r):
  d = len(x0)
  x = np.copy(x0).reshape(d)
  x_out = np.zeros((T,d))
  x_out[0,:] = np.copy(x)
  f_out = np.zeros((T,1))
  f_out[0] = eval_f(x)
  t_noise = -t_thres - 1
  for t in np.arange(1,T):
    ZO_grad = np.zeros(d)
    fx = eval_f(x)
    for i in np.arange(d):
      e_i = np.zeros(d)
      e_i[i] = 1.
      x_plus = x + mu * e_i
      ZO_grad += (eval_f(x_plus) - fx)/mu * e_i
    ZO_grad_norm = np.linalg.norm(ZO_grad)
    if ZO_grad_norm <= g_thres and t - t_noise > t_thres:
      Y = np.random.normal(scale = 1., size = (d,))
      Y = Y/np.linalg.norm(Y) * r
      x += Y
      t_noise = t
    x = x - 1.0/(4 * L) * (ZO_grad)
    x_out[t,:] = x
    f_out[t] = eval_f(x)
  return (x_out,f_out)

#version of PGD
def PGD(x0,T,t_thres,g_thres,r):
  d = len(x0)
  x = np.copy(x0).reshape(d)
  x_out - np.zeros((T,d))
  x_out[0,:] = np.copy(x)
  f_out = np.zeros((T,1))
  f_out[0] = eval_f(x)
  t_noise = -t_thres - 1
  for t in np.arange(1,T):
    grad_f = get_grad_f(x)
    fx = eval_f(x)
    grad_f_norm = np.linalg.norm(grad_f)
    if grad_f_norm <= g_thres and t - t_noise > t_thres:
      Y = np.random.normal(scale = 1., size = (d,))
      Y = Y/np.linalg.norm(Y) * r
      x += Y
      t_noise = t
    x = x - 1.0/(4 * L) * (grad_f)
    x_out[t,:] = x
    f_out[t] = eval_f(x)
  return (x_out,f_out)

def zero_order_noisy(x0,T,mu,noise_scale):
    d = len(x0)
    x = np.copy(x0).reshape((d,))
    x_out = np.zeros((T, d))
    x_out[0,:] = np.copy(x)
    f_out = np.zeros((T,1))
    f_out[0] = eval_f(x)
    ZO_grad_hist = np.zeros((T,d))
    ZO_grad_coef_hist = np.zeros((T,1))
    ZO_noise_hist = np.zeros((T,d))
    for t in np.arange(1,T):
        Z = np.random.normal(size = (d,))
        x_plus = x + mu * Z
        x_minus = x - mu * Z
        ZO_grad_coef_hist[t] = (eval_f(x_plus) - eval_f(x_minus))/(2. *mu)
        ZO_grad = (eval_f(x_plus) - eval_f(x_minus))/(2. * mu) * Z
        ZO_grad_hist[t,:] = ZO_grad
        ZO_noise_hist[t,:] = ZO_grad - get_grad_f(x)
        noise = np.random.normal(scale = noise_scale, size = (d,))
        x = x -1.0/(4. * (d) * L) * (ZO_grad + noise)
        x_out[t,:] = x
        f_out[t] = eval_f(x)
    return (x_out, f_out, ZO_grad_hist, ZO_grad_coef_hist, ZO_noise_hist)

def GD(x0,T):
    d = len(x0)
    x = np.copy(x0).reshape((d,))
    x_out = np.zeros((T, d))
    x_out[0,:] = np.copy(x)
    f_out = np.zeros((T,1))
    f_out[0] = eval_f(x)
    for t in np.arange(1, T):
        x = x - 1.0/(4.0 * L) * get_grad_f(x)
        x_out[t,:] = x
        #print t, x
        f_out[t] = eval_f(x)
    return (x_out, f_out)
