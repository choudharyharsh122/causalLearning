from ot.utils import list_to_array
from ot.backend import get_backend
import warnings
import argparse
import numpy as np
import time
from scipy.optimize import linprog
from qpsolvers import solve_qp
from ot.utils import unif, dist, list_to_array
import autoray as ar
# from .backend import get_backend


def makeparms(maxiter=1, beta=10, rho=0.8, lamb=0.5, hess='diag', tau=1., mbsz=1, numcon=1, geomp=0.7, stepdecay='dimin', gammazero=0.1, zeta=0.1):
    params = {
        'maxiter': maxiter,  # number of iterations performed
        'beta': beta,  # trust region size
        'rho': rho,  # trust region for feasibility subproblem
        'lamb': lamb,  # weight on the subfeasibility relaxation
        'hess': hess,  # method of computing the Hessian of the QP, options include 'diag' 'lbfgs' 'fisher' 'adamdiag' 'adagraddiag'
        'tau': tau,  # parameter for the hessian
        'mbsz': mbsz,  # the standard minibatch size, used for evaluating the progress of the objective and constraint
        'numcon': numcon,  # number of constraint functions
        'geomp': geomp,  # parameter for the geometric random variable defining the number of subproblem samples
        'stepdecay': stepdecay, # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
        'gammazero': gammazero,  # initial stepsize
        'zeta': zeta,  # parameter associated with the stepsize iteration
    }
    return params


# def computekappa( cval, cgrad, rho, lamb, mc, n):
#     obj = np.concatenate(([1.], np.zeros((n,))))
#     Aubt = np.concatenate((([-1.]), cgrad))
#     # if there are multiple constraints? Aubt.reshape(mc,n+1) ??
#     Aubt = Aubt.reshape(mc, n+1)
#     res = linprog(c=obj, A_ub=Aubt, b_ub=[-cval], bounds=(-rho, rho))
#     return ((1-lamb)*max(0, cval)+lamb*max(0, res.fun))

def computekappa(cval, cgrad, lamb, rho, mc, n, scalef):  
    obj = np.concatenate(([1.], np.zeros((n,))))
    Aubt = np.column_stack((-np.ones(mc), np.array(cgrad)))
    try:
       res = linprog(c=obj, A_ub=Aubt, b_ub=-np.array(cval), bounds=[(-rho, rho)])
       #print("IMPORTANT!!!!!",res.fun)
       return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
    except:
       return (1-lamb)*max(0, sum(cval)) + lamb*max(0, rho)
    #res = linprog(c=obj, A_ub=Aubt, b_ub=-np.array(cval), bounds=[(-rho, rho)])
    #return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
    #return lamb*max(0, res.fun)


# def solvesubp(fgrad, cval, cgrad, kap, beta, tau, hesstype, mc, n):
#     if hesstype == 'diag':
#        #P = tau*nx.eye(n)
#        P = tau*np.identity(n)
#     return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((mc, n)), list_to_array([(kap-cval)]), np.zeros((0, n)), np.zeros((0,)), -beta*np.ones((n,)), beta*np.ones((n,)), solver='osqp')


def solvesubp(fgrad, cval, cgrad, kap_val, beta, tau, hesstype, mc, n):
    if hesstype == 'diag':
       # P = tau*nx.eye(n)
       P = tau*np.identity(n)
       #print("P shape:", P.shape)
       kap = kap_val * np.ones(mc)
       cval = np.array(cval)
       #print("cval shape:", cval.shape)
       #print("cgrad shape: ", cgrad.shape)
       #print("fgrad shape: ", fgrad.shape)
    return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((mc, n)), kap-cval, np.zeros((0, n)), np.zeros((0,)), -beta*np.ones((n,)), beta*np.ones((n,)), solver='osqp')

# initw : Initial parameters of the Network (Weights and Biases)


def StochasticGhost(obj_fun, obj_grad, con_funs, con_grads, initw, params):
    
   #print(con_funs)
   N = params["N"]
   n = params["n"]  
   maxiter = params["maxiter"]
   beta = params["beta"]
   rho = params["rho"]
   lamb = params["lamb"]
   tau = params["tau"]
   hess = params["hess"]
   mbsz = params["mbsz"]
   mc = params["numcon"]
   geomp = params["geomp"]
   stepdec = params["stepdecay"]
   gamma0 = params["gammazero"]
   zeta = params["zeta"]
   gamma = gamma0
   lossbound = params["lossbound"]
   scalef = params["scalef"]

   epsilon = 0.01
   steps = []
   
   w = initw
   for i in range(len(w)):
      w[i] = ar.to_numpy(w[i])

   feval = obj_fun(w, mbsz) 
   ceval = np.zeros((mc,))
   #ceval_white = np.zeros((mc,))
   #ceval_black = np.zeros((mc,))
   Jeval = np.zeros((mc, n))

   # Getting all the constraints
   #iterfs = np.zeros((maxiter,))
   #iterfs[0] = feval
   iterfs = []
   iterfs.append(feval)
   dir_obj = np.zeros((maxiter,))
   dir_cons = np.zeros((maxiter, mc))
   obj_grad_norm = np.zeros((maxiter,))
   cons_grad_norm = np.zeros((maxiter, mc))
   for i in range(mc):
      conf = con_funs[i]
      #ceval[i] = np.max(conf(w, mbsz), 0)
      #print("mc is:", mc)
      #print("i-1 is:", i-1)
      #print("The i is:", i)
      ceval[i] = np.max(conf(w, mbsz, i-1), 0)
   #itercs = np.zeros((maxiter,))
   #itercs = np.zeros((maxiter, mc))
   #itercs_black = np.zeros((maxiter, mc))
   #itercs_white = np.zeros((maxiter, mc))
   #itercs[0,:] = np.max(ceval)

   itercs = []
   #itercs.append(np.max(ceval))

   min_w = None
   min_inf_norm_dsol = 9999
   consecutive_small_steps = 0
   for iteration in range(5999):
      print("Iteration: ", iteration + 1)

      if stepdec == 'dimin':
         gamma = gamma0 / (iteration + 1) ** zeta
      elif stepdec == 'constant':
         gamma = gamma0
      elif stepdec == 'slowdimin':
         gamma = gamma * (1 - zeta * gamma)
      elif stepdec == 'stepwise':
         gamma = gamma0 / (10 ** (int(iteration * zeta)))

      Nsamp = np.random.geometric(p=geomp)
      while (2 ** (Nsamp + 1)) > N:
         Nsamp = np.random.geometric(p=geomp)

      mbatches = [1, 2 ** Nsamp, 2 ** Nsamp, 2 ** (Nsamp + 1)]
      dsols = np.zeros((4, n))

      for j in range(4):
         feval = obj_fun(w, mbatches[j])
         fgrad = ar.to_numpy(obj_grad(w, mbatches[j]))
         ceval = np.zeros(mc)
         Jeval = np.zeros((mc, n))
         
         for i in range(mc):
               conf = con_funs[i]
               conJ = con_grads[i]
               ceval[i] = np.max(conf(w, mbatches[j], i - 1) - lossbound[i], 0)
               Jeval[i, :] = ar.to_numpy(conJ(w, mbatches[j], i - 1))

         kap = computekappa(ceval, Jeval, rho, lamb, mc, n, scalef)
         dsol = solvesubp(fgrad, ceval, Jeval, kap, beta, tau, hess, mc, n)
         dsols[j, :] = dsol

      dsol = dsols[0, :] + (dsols[3, :] - 0.5 * dsols[1, :] - 0.5 * dsols[2, :]) / (geomp * ((1 - geomp) ** Nsamp))
      
      inf_norm_dsol = np.max(np.abs(dsol))
      steps.append(inf_norm_dsol)
      print("Inf norm of step-size: ", inf_norm_dsol)
      
      if inf_norm_dsol < epsilon:

         if inf_norm_dsol < min_inf_norm_dsol:
            min_inf_norm_dsol = inf_norm_dsol
            min_w = [np.copy(param) for param in w]

         consecutive_small_steps += 1
         print(f"Small step detected. Count: {consecutive_small_steps}")
         if consecutive_small_steps >= 8:
               print("Terminating early due to 5 consecutive small steps.")
               break
      else:
         consecutive_small_steps = 0
         # Update w with the step
         start = 0
         for i in range(len(w)):
            #print(w[i].size)
            end = start + np.size(w[i])
            #print("Parameter ", i+1, np.reshape(dsol[start:end], np.shape(w[i])))
            w[i] = w[i] + gamma*np.reshape(dsol[start:end], np.shape(w[i]))
            start = end
         #print("Updated w:", w)
         #print("old w: ", w)
      
      
      #print("new w", w)
      
      feval = obj_fun(w, mbsz)
      # dir der purposes
      fgrad = obj_grad(w, mbsz)
      # dir der purposes
      #iterfs[iteration] = feval

      iterfs.append(feval)

      for i in range(mc):
         conf = con_funs[i]
         ceval[i] = np.max(conf(w, mbsz, i-1), 0)
         #Jeval[i, :] = ar.to_numpy(conJ(w, mbsz, i-1))
      #itercs[iteration, :] = ceval

      itercs.append(ceval)

   iterfs = np.array(iterfs)
   itercs = np.array(itercs)
   steps = np.array(steps)

   return w, iterfs, itercs, min_w, steps
