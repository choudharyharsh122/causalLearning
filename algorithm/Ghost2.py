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

def computekappa(cval, cgrad, lamb, rho, mc, n):  
    obj = np.concatenate(([1.], np.zeros((n,))))
    Aubt = np.column_stack((-np.ones(mc), np.array(cgrad)))
    res = linprog(c=obj, A_ub=Aubt, b_ub=-np.array(cval), bounds=[(-rho, rho)])
    #print("IMPORTANT!!!!!",res.fun)
    return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
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


def backtracking_line_search(w, dsol, gradient, obj_fun, step_size, mbsz, alpha=0.3, beta=0.8, max_ls_iter=100):
        """ Perform Backtracking Line Search to find an appropriate step size """
        ls_iter = 0
        while ls_iter < max_ls_iter and obj_fun([wi + step_size * dsi for wi, dsi in zip(w, dsol)], mbsz) > obj_fun(w, mbsz) - alpha * step_size * np.dot(gradient, gradient):
            step_size *= beta
            ls_iter += 1
        if ls_iter == max_ls_iter -1:
            print("Couldn't find the lest obj reduction")
        else:
            print("reduced objective at iter:", ls_iter)
        return step_size


def StochasticGhost(obj_fun, obj_grad, con_funs, con_grads, initw, params):
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

    
    w = initw
    for i in range(len(w)):
        w[i] = ar.to_numpy(w[i])

    feval = obj_fun(w, mbsz)  
    ceval = np.zeros((mc,))
    ceval_white = np.zeros((mc,))
    ceval_black = np.zeros((mc,))
    Jeval = np.zeros((mc, n))

    # Getting all the constraints
    # iterfs = np.zeros((maxiter,))
    # iterfs[0] = feval

    iterfs = []
    #iterfs.append(feval)
    for i in range(mc):
       conf = con_funs[i]
       #ceval[i] = np.max(conf(w, mbsz), 0)
       ceval[i] = np.max(conf(w, mbsz, i-1), 0)
    #itercs = np.zeros((maxiter,))
    #itercs = np.zeros((maxiter, mc))
    # itercs_black = np.zeros((maxiter, mc))
    # itercs_white = np.zeros((maxiter, mc))
    #itercs[0,:] = np.max(ceval)

    itercs = []
    #itercs.append(np.max(ceval))

    min_w = None
    min_feval = 9999
    epsilon = 0.002

    for iteration in range(0, 10000):

        print("Iteration : ", iteration+1)

        if stepdec == 'dimin':
           gamma = gamma0/(iteration+1)**zeta
        if stepdec == 'constant':
           gamma = gamma0
        if stepdec == 'slowdimin':
           gamma = gamma*(1-zeta*gamma)
        if stepdec == 'stepwise':
           gamma = gamma0 / (10**(int(iteration*zeta)))

        Nsamp = np.random.geometric(p=geomp)
        #while (2**(Nsamp+1)) > N:
        #  Nsamp = np.random.geometric(p=geomp)

        # Only specify the number of minibatches here
        # Lets the user decide on the samples for each minibatch number
        mbatches = [1, 2**Nsamp, 2**Nsamp, 2**(Nsamp+1)]
        dsols = np.zeros((4, n))

        #for j in range(4):
        feval = obj_fun(w, N)
        fgrad = ar.to_numpy(obj_grad(w, N))
        #print(type(fgrad))
        # cval = []
        # cgrad = []
        for i in range(mc):
            # con_funs[i](conf) and con_grads[i](conJ) ith constraint and constraint grad
            conf = con_funs[i]
            conJ = con_grads[i]
            # ceval and Jeval are evaluations of ith constraint and constraint grads for the parameter values
            # nx.max(conf(w,mbatches[j]),0) to ensure the problem is always in the feasible region
            #ceval[i] = np.max(conf(w, mbatches[j]) - lossbound[i], 0)
            ceval[i] = np.max(conf(w, N, i-1) - lossbound[i], 0)
            Jeval[i, :] = ar.to_numpy(conJ(w, N, i-1))
            
            #print(type(Jeval[i, :]))
            # cval = nx.concatenate((cval, ceval[i]))
            # cgrad = nx.concatenate((cgrad, Jeval[i, :]))

            # Compute Kappa for the Subproblem bound 
        kap = computekappa(ceval, Jeval, rho, lamb, mc, n)
            # Solving the subproblem
        dsol = solvesubp(fgrad, ceval, Jeval, kap, beta, tau, hess, mc, n)
         #dsols[j, :] = dsol
        inf_norm_dsol = np.max(np.abs(dsol))
        print("Inf norm of step-size: ", inf_norm_dsol)
        
        if inf_norm_dsol < epsilon:

            # if inf_norm_dsol < min_inf_norm_dsol:
            #     min_inf_norm_dsol = inf_norm_dsol
            #     min_w = [np.copy(param) for param in w]

            consecutive_small_steps += 1
            print(f"Small step detected. Count: {consecutive_small_steps}")
            if consecutive_small_steps >= 10:
                print("Terminating early due to 10 consecutive small steps.")
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
        
        feval = obj_fun(w, mbsz)
        # dir der purposes
        fgrad = obj_grad(w, mbsz)
        # dir der purposes
        #iterfs[iteration] = feval

        iterfs.append(feval)

        if feval < min_feval:
         min_feval = feval
         min_w = [np.copy(param) for param in w]
        for i in range(mc):
          conf = con_funs[i]
          ceval[i] = np.max(conf(w, mbsz, i-1), 0)
          #Jeval[i, :] = ar.to_numpy(conJ(w, mbsz, i-1))
          #ceval[i] = np.max(conf(w, mbsz)[0], 0)
          #ceval_black[i] = np.max(conf(w, mbsz)[1], 0)
          #ceval_white[i] = np.max(conf(w, mbsz)[2], 0)
        itercs.append(ceval)

    iterfs = np.array(iterfs)
    itercs = np.array(itercs)
        #itercs_black[iteration, :] = ceval_black
        #itercs_white[iteration, :] = ceval_white

    return w, iterfs, itercs, min_w
