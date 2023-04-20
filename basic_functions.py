# This file contains basic functions for SCAD penalized data fitting.

#packages
import numpy as np
from numpy import linalg as LA 
import math
import time 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pandas as pd
import plotly.graph_objects as go
#from scipy import optimize
from scipy.optimize import minimize
from sklearn.model_selection import KFold

#Content:
## 1. generate covariance matrix: cov_generator()
## 1. Data Genration: data_generator()
## 2. Estimation of coefficient matrix B based on the SCAD-penalized 
#   least squars function given the tuning pearmeter: estB_LS_SCAD()
## 3. Optimizing the tuning parameter by K-fold cross-validation: cv_tuning()

#
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# # generate coefficient matrix B
# def B_generator(size, distribution, sparsity_level, range=(-10,10)):
#     """Generate the sparse coefficient matrix 

#     Args:
#         size (tuple): (row number, col number)
#         distribution (str): either "normal" or "uniform" from which the random numbers are genrated
#         sparsity_level (float): a proportion of zero among the coefficent matrix
#         range (tuple, optional): the range of the value sampled from when uniform distributions is used. Defaults to (-10,10).

#     Returns:
#         nparray: the coefficient matrix of row number by col number
#     """
    
#     B_0=np.zeros(size=size)
#     np.random.choice
    

# generate covariance matrix  decreasing_rate
def cov_generator(dim, MODEL="exp", decreasing_rate=0.35, variance=1, power_base=0.5):
    """
    Generate Covaraince matrix, either a random full positive definite matrix, or a diagonal matrix
    
    dim (int): an integer indicating the dimension of the covariance matrix.
    MODEL (str): a string indicating the model to generate the covariance: "identity", "exp", "power"
    decreasing rate (float): the number multiplied at the power, negative value for "exp" mode; positive value for "power" model
    variance (float or list): the var on the diagonal of the final matrix, could be a number or a list of values.   
    power_base (float): the base number for the "power" model, a positive number in (0,1)
    """
    
    if MODEL=="identity":
        if variance!=None:
            diag_values=variance
        if len(diag_values)==dim:
            Sigma=np.eye(dim)@np.diag(diag_values)
            return Sigma
        elif len(diag_values)==1:
            Sigma=np.eye(dim)*(diag_values)
            return Sigma 
    elif MODEL=="exp":
        col_max=np.tile(np.arange(dim),(dim,1))
        row_max=col_max.T
        if len(variance)==1:
            Sigma=variance*np.eye(dim)@np.exp(-decreasing_rate*abs(col_max-row_max))
        elif len(variance)==dim:
            Sigma=np.diag(variance)*np.exp(-decreasing_rate*abs(col_max-row_max))
        return Sigma
    elif MODEL=="power":
        col_max=np.tile(np.arange(dim),(dim,1))
        row_max=col_max.T
        if len(variance)==1:
            Sigma=variance*power_base**(decreasing_rate*abs(col_max-row_max))
        elif len(variance)==dim:
            Sigma=np.diag(variance)@power_base**(decreasing_rate*abs(col_max-row_max))
        return Sigma


def data_generator(dim_Y, dim_X, N_Obs, COV_X, COV_U, COV_Ex, B, X_distribution="Normal", Ex_distribution="Normal",
                    U_distribution="Normal"):
    """_summary_
    Args:
        dim_Y: dimension of response vector (Y_i)
        dim_X: dimension of covariates vector (X_i)
        N_Obs: the number of observations
        COV_X: the covariance matrix of the covariates (X_i)
        COV_U: the covariance matrix of the error (U_i)
        COV_EX: the covariance matrix of the measurement error (E_xi)
        X_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy","GMM"). indicating if the data X(covariates) and Ex(ME) generated from normal distributions
        Ex_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy","GMM"). indicating the distribution from that ME Ex is generated when Sig_Ex is diagonal
        U_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy"). indicating the distribution that U is generated from 
    """

    #1. generate U from COV_U
    if U_distribution=="Normal":
        C_u=LA.cholesky(COV_U)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_Y,N_Obs))
        U=C_u@Z
    elif U_distribution=="Uniform":
        err_level=COV_U[0,0]
        U=np.random.uniform(low=-np.sqrt(3*err_level), high=np.sqrt(3*err_level), size=(dim_Y,N_Obs))
    elif U_distribution=="Cauchy":
        U=np.random.standard_cauchy(size=(dim_Y,N_Obs))
        err_level=COV_U[0,0]
        U=(U/U.std())*np.sqrt(err_level=COV_U[0,0])
    elif U_distribution=="Laplace":
        err_level=COV_U[0,0]
        U=np.random.laplace(loc=0, scale=np.sqrt(err_level/2), size=(dim_Y,N_Obs))
    elif U_distribution=="Gamma":
        Z=np.random.gamma(shape=4, scale=0.5, size=(dim_Y,N_Obs))
        Z=Z-2
        C_u=LA.cholesky(COV_U)
        U=C_u@Z
    elif U_distribution=="chi-square":
        Z=np.random.chisquare(2, size=(dim_Y,N_Obs))
        Z=(Z-2)/2
        C_u=LA.cholesky(COV_U)
        U=C_u@Z
     
    #2. generate X_{q\times n} from COV_X
    if X_distribution=="Normal":
        C_x=LA.cholesky(COV_X)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X, N_Obs))
        X=C_x@Z
    elif X_distribution=="Uniform":
        Z=np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(dim_X, N_Obs))  
        C_x=LA.cholesky(COV_X)
        X=C_x@Z  
    elif X_distribution=="Cauchy":
        Z=np.random.standard_cauchy(size=(dim_X,N_Obs))
        Z=Z/Z.std()
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="Laplace":
        Z=np.random.laplace(loc=0, scale=np.sqrt(0.5), size=(dim_X,N_Obs))
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="chi-square":
        Z=np.random.chisquare(2, size=(dim_X,N_Obs))
        Z=(Z-2)/2
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="Gamma":
        Z=np.random.gamma(shape=4, scale=0.5, size=(dim_X,N_Obs))
        Z=Z-2
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
        
    #3. Construct Y
    Y=B@X+U
       
    #4. generate measurement error 
    if Ex_distribution=="Normal":
        C_Ex=LA.cholesky(COV_Ex)
        Z=np.random.normal(loc=0.0, scale=1.0,size=(dim_X,N_Obs))
        Ex=C_Ex@Z        
    elif Ex_distribution=="Uniform":
        Z=np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(dim_X, N_Obs))  
        C_x=LA.cholesky(COV_Ex)
        Ex=C_x@Z
    elif Ex_distribution=="Cauchy":
        Z=np.random.standard_cauchy(size=(dim_X,N_Obs))
        Z=Z/Z.std()
        C_x=LA.cholesky(COV_Ex)
        Ex=C_x@Z
    elif Ex_distribution=="Laplace":
        Z=np.random.laplace(loc=0, scale=np.sqrt(0.5), size=(dim_X,N_Obs))
        C_x=LA.cholesky(COV_Ex)
        Ex=C_x@Z
    elif Ex_distribution=="chi-square":
        Z=np.random.chisquare(2, size=(dim_X,N_Obs))
        Z=(Z-2)/2
        C_x=LA.cholesky(COV_Ex)
        Ex=C_x@Z
    elif Ex_distribution=="Gamma":
        Z=np.random.gamma(shape=4, scale=0.5, size=(dim_X,N_Obs))
        Z=Z-2
        C_x=LA.cholesky(COV_Ex)
        Ex=C_x@Z
        
    #construct X*   
    X_star=X+Ex
    
    #return results
    return Y, X_star, X, U 

# the objective function in order to minimize PBCSCAD
def tight_upper_bound(vec_B, Y, X_star, tun_para, cav_deri):
     p, n=Y.shape
     q=X_star.shape[0]
     return 0.5*LA.norm(Y-vec_B.reshape(p,q)@X_star)**2+n*tun_para*np.sum(abs(vec_B))+np.sum(cav_deri*vec_B)


# optimization for penalized bias-corrected least squares function
def minimize_BCLSSCAD(tun_para, Y, X_star, Sigma_Ex, lower_bound, 
                      upper_bound, opt_method, case, 
                      stop_criterion="threshold", threshold=10**(-6)):
    """minimize the bais-corrected SCAD penalized objective

    Args:
        tun_para (float): The tuning parameter lambda
        Y (nparray): response data (Y) with dimension p by n
        X_star (nparray): error-contaminated data (X) with dimension q by n
        Sigma_Ex (nparray): the covariance matrix of the measurement error
        lower_bound (float): the lower bound of the beta values
        upper_bound (float): the upper bound of the beta values
        case (str): "corrected" or "naive" which influences the initial value
        stop_criterion (string): indicating the stoping criterion for the interation. 
                                "objective below": stop iteration when the new objective function is greater than the previous one
                                "threshold": stop interation when diff < threshold 10**(-6)
    """
    a=3.7
    p, n_obs=Y.shape 
    q=X_star.shape[0] 
    if case=="corrected":      
        B_start=Y@X_star.T@LA.inv(X_star@X_star.T-n_obs*Sigma_Ex)
    elif case=="naive":
        B_start=Y@X_star.T@LA.inv(X_star@X_star.T)
    
    CONTINUE=True
    CONVERGE=True
    B_c=B_start.reshape(-1)
    obj_fun=10**(100)
    round=1
    while CONTINUE:
        
        print(f"optimize round {round}")
        cav_part1=np.array(-n_obs*B_c.reshape(p,q)@Sigma_Ex).reshape(-1) 
        
        B_c_vec=B_c.reshape(-1,1)
        col1=abs(B_c_vec)<=tun_para
        col2=np.greater(abs(B_c_vec),tun_para) & np.less(abs(B_c_vec),a*tun_para)
        col3=abs(B_c_vec)>=a*tun_para
        col_logi=np.concatenate((col1, col2, col3), axis=1)
        value1=np.zeros((p*q,1))
        value2=(tun_para-abs(B_c_vec))/(a-1)*np.sign(B_c_vec)
        value3=-tun_para*np.sign(B_c_vec)
        col_value=np.concatenate((value1, value2, value3), axis=1)
        temp=col_value*col_logi
        cav_part2=temp.sum(axis=1)
        
        cav_deri=cav_part1+n_obs*cav_part2
        
        if lower_bound==None and upper_bound==None:           
            res=minimize(fun=tight_upper_bound, x0=B_c.reshape(-1), 
                         args=(Y, X_star, tun_para, cav_deri,),method=opt_method) 
        else:
            bounds=[(lower_bound,upper_bound)]*(p*q)
            res=minimize(fun=tight_upper_bound, x0=B_c.reshape(-1), 
                         args=(Y, X_star, tun_para, cav_deri,),bounds=bounds,method=opt_method)

        if stop_criterion=="objective below":
            if (res.fun > obj_fun or res.success==False or round > 20) and round > 1:
                print(f"objective function:{res.fun}")
                CONTINUE=False
                B_est=B_c
            else:
                print(f"objective function:{res.fun}")
                obj_fun=res.fun
                B_c=res.x
                
        elif stop_criterion=="threshold":  
            #print(f"objective function:{res.fun}")
            print(f"F norm diff est: {LA.norm(res.x-B_c)}") 
             
            if LA.norm(res.x-B_c)<threshold or round>200:
                CONTINUE=False 
                if round>200 and LA.norm(res.x-B_c)>=threshold:
                    CONVERGE=False
                B_est=res.x
            else:
                obj_fun=res.fun
                B_c=res.x 
            
        round+=1 
        
            
    return B_est.reshape(p,q), CONVERGE

def minimize_LQA_naive(tun_para, Y, X_star):
    """using LQA to do minimization for the naive method

    Args:
        tun_para (float): the parameter value
        Y (nparray): response data
        X_star (nparray): covariates data

    Returns:
        B_new: estimate of B
    """

    a=3.7
    #obtain the dimension of the data
    p,n=Y.shape
    q=X_star.shape[0]
    #setting the initial values of B to be the naive estimate
    B_start=Y@X_star.T@LA.inv(X_star@X_star.T)
    
    #start looping until B hat converges
    CONTINUE=True
    B_c=B_start
    round=1
    #diff_F_norm=100000
    while CONTINUE:
        #print(f"minimization round {round}")
        B_new=np.zeros((p,q))
        #estimating B by each row
        for k in range(p):
            derivative_list=[]
            for h in range(q):
                if abs(B_c[k,h])<=tun_para:
                    derivative_list.append(tun_para/abs(B_c[k,h]))
                elif abs(B_c[k,h])>tun_para and abs(B_c[k,h])<=a*tun_para:
                    derivative_list.append((a*tun_para-B_c[k,h])/((a-1)*abs(B_c[k,h])))                  
                elif abs(B_c[k,h])>a*tun_para:
                    derivative_list.append(0)
            B_new[k,]=Y[k,].reshape(1,-1)@X_star.T@LA.inv(X_star@X_star.T+n*np.diag(derivative_list))
        #finish looping, then calculate the difference between two iterations
        diff_F_norm_new=LA.norm(B_c-B_new)
        print(f"diff_F_norm_new={diff_F_norm_new}")
        if round>1000 or diff_F_norm_new<10**(-6):
            CONTINUE=False
        B_c=B_new
            #diff_F_norm=diff_F_norm_new            
        round+=1
        
    return B_c
                
# paramater tuning plot 
# check the trend that how the objective is change with differene paramater values
def cv_tuning_plot(Y, X_star, Sigma_Ex, param_lower_bound,param_upper_bound, plot_points, 
                   lower_bound, upper_bound, opt_method,figure_name,case,K=5, generate_plot=True):
    """Cross-validation trend ploting

    Args:
        Y (nparray): response data
        X_star (nparray): error-contaminated covariates
        Sigma_Ex (nparray): covariance matrix of the measurement error
        param_lower_bound (float): lb of tuning parameter lambda
        param_upper_bound (float): ub of runing parameter lambda
        plot_points (int): number of points in the plots
        lower_bound (float): lb for beta values
        upper_bound (float): ub for beta values
        opt_method (float): the algorithm used to minimize the subobjective
        figure_name (str): the name of figure
        case (str): "corrected" or "niave"
        K (int, optional): Fold number. Defaults to 5.
    """
    print(f"K={K}")
    kf=KFold(n_splits=K, shuffle=True, random_state=1)
    
    loss=[]
    for param in np.linspace(param_lower_bound, param_upper_bound, plot_points):
        print(f"parameter: {param}")
        loss_per_param=[]
        i=1
        for train_ind, test_ind in kf.split(Y.T):
            #print(f"Fold: {i}")
            
            #define the training and test set for each fold fitting
            X_train, Y_train=X_star[:,train_ind], Y[:,train_ind]
            X_test, Y_test=X_star[:,test_ind], Y[:,test_ind]
            
            if case=="corrected":
            
                B_est, _=minimize_BCLSSCAD(tun_para=param, Y=Y_train, X_star=X_train,
                                        Sigma_Ex=Sigma_Ex, lower_bound=lower_bound,
                                        upper_bound=upper_bound, opt_method=opt_method, 
                                        case=case)
            elif case=="naive":
                B_est=minimize_LQA_naive(tun_para=param, Y=Y_train, X_star=X_train)
            
            n=len(test_ind)
            #loss_per_param.append(-np.trace(X_test@Y_test.T@B_est)+0.5*np.trace((X_test@X_test.T-n*Sigma_Ex)@B_est.T@B_est))
            loss_per_param.append(LA.norm(Y_test-B_est@X_test)**2-n*np.trace(Sigma_Ex@B_est.T@B_est))
            i+=1
            
        loss.append(np.mean(loss_per_param))
        
    if generate_plot==True:       
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(np.linspace(param_lower_bound,param_upper_bound, plot_points), loss)
        #plt.show()
        plt.savefig(figure_name+".png")
        
    return loss

#parameter tuning
def cv_tuning_pointwise(Y, X_star, Sigma_Ex,param_lower_bound,param_upper_bound, 
                        plot_points, case, generate_figure, K=5):
    """tuning the parameter lambda for the naive case

    Args:
        Y (nparray): response data
        X_star (nparray): covariates data
        Sigma_Ex (nparray): for the naive case, it should be the zero array
        param_lower_bound (float): the lower bound of the parameter lambda
        param_upper_bound (float): the upper bound of the parameter lambda
        plot_points (int): number of points in the potential lambda list
        case (str): indicating the case: should be "naive"

    Returns:
        opt_lambda: the optimized lambda value
    """
    
    loss_list=cv_tuning_plot(Y, X_star, Sigma_Ex, param_lower_bound,param_upper_bound, plot_points, 
                   lower_bound=-100, upper_bound=100, opt_method="trust-constr",
                   figure_name="figure",case=case,K=K, generate_plot=generate_figure)
    
    low_index=np.where(loss_list==min(loss_list))
    opt_lambda=np.linspace(param_lower_bound, param_upper_bound, plot_points)[low_index][0]
    
    return opt_lambda

def cv_tuning_adaptive(Y, X_star, Sigma_Ex, param_lower_bound, param_upper_bound, 
              points_per_round, lower_bound, upper_bound, opt_method, Round,
              case,K=5,threshold=0.01):
    """tuning the parameter lambda

    Args:
        Y (nparray): the response data 
        X_star (nparray): the error-contaminated data
        Sigma_Ex (nparray): the covariance matrix of measurement error
        parameter_lower_bound (int): the lower value bound of the tuning parameter lambda
        param_upper_bound (int): the upper value bound of the tuning parameter lambda
        points_per_round (int): iterative tuning the points per round
        lower_bound (float): the lower bound of the beta range
        upper_bound (float): the upper bound of the beta range
        opt_method (str): the optimization algorithm
        Round (int): round limit
        case (str): either "corrected" or "naive"
        K (int, optional): the number of fold in cv. Defaults to 5.
        threshold (float, optional): the threshold to determine when to stop. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    
    kf=KFold(n_splits=K, shuffle=True, random_state=1)
    current_opt_param=0
    params_values=np.linspace(param_lower_bound, param_upper_bound, points_per_round)
    print(f"initial parameter value list:{params_values}")
    for r in range(Round):
        
        print(f"Round {r+1}")
        loss=[]
        
        for param in params_values:
            #print(f"parameter: {param}")
            loss_per_param=[]
            k=1
            
            for train_ind, test_ind in kf.split(Y.T):
                #print(f"Fold: {k}")
                
                #define the training and test set for each fold fitting
                X_train, Y_train=X_star[:,train_ind], Y[:,train_ind]
                X_test, Y_test=X_star[:,test_ind], Y[:,test_ind]
                
                B_est,_=minimize_BCLSSCAD(tun_para=param, Y=Y_train, X_star=X_train,
                                        Sigma_Ex=Sigma_Ex, lower_bound=lower_bound,
                                        upper_bound=upper_bound, opt_method=opt_method, case=case)
                
                n=len(test_ind)
                #loss_per_param.append(-np.trace(X_test@Y_test.T@B_est)+0.5*np.trace((X_test@X_test.T-n*Sigma_Ex)@B_est.T@B_est))
                loss_per_param.append(LA.norm(Y_test-B_est@X_test)**2-n*np.trace(Sigma_Ex@B_est.T@B_est))
                k+=1
                
            loss.append(np.mean(loss_per_param)) 
            
        sorted_loss=sorted(loss)
        print(f"loss: {loss}")

        index = []
        for j in range(4):
            positions = [i for i, x in enumerate(loss) if x == sorted_loss[j]]
            index=index+positions
        index2=np.array(index)
        if sum(index2<index2[0])==0:
            index=index+[index[0]-1]
        elif sum(index2>index2[0])==0:
            index=index+[index[0]+1]
        print(f"index {index}")
        param_lower_bound=params_values[min(index)]
        param_upper_bound=params_values[max(index)]
        opt_param=params_values[index[0]]   
        print(f"current of parameter: {opt_param}")
        params_values=np.linspace(param_lower_bound, param_upper_bound, points_per_round) 
        print(f"updated parameter list: {params_values}")
        
        if abs(current_opt_param-opt_param)<threshold:         
            break
        else:
            current_opt_param=opt_param
            
    return opt_param
        
def single_simulation(p, q, N, B, Sigma_XX, Sigma_UU, Sigma_Ex,
                      opt_lambda_correct, 
                      X_distribution="Normal", Ex_distribution="Normal", 
                      U_distribution="Normal", selection_threshold=10**(-6),
                      naive_lambda_lb=0.001, naive_lambda_ub=0.16, n_lambda_points=160):
    """pipline of single simulations.

    Args:
        p (int): dimension of response Y
        q (int): dimension of covariates X
        N (int): number of observarions
        B (nparray): coefficient matrix B
        Sigma_XX (nparray): covariance matrix of X
        Sigma_UU (nparray): covariance matrix of U
        Sigma_Ex (nparray): covariance matrix of measurement error
        opt_lambda_correct (float): optimal lambda for corrected method
        opt_lambda_naive (float): optimal lambda for naive method
        X_distribution (str, optional): distribution from which the covariates is generated. Defaults to "Normal".
        Ex_distribution (str, optional): distribution from which the measurement error is generated. Defaults to "Normal".
        U_distribution (str, optional): distribution from which the model error is generated. Defaults to "Normal".
        selection_threshold (float): threshold to determine zero estimates
    """
    
    #step 1: generate data
    Y, X_star, X, U=data_generator(dim_Y=p, dim_X=q, N_Obs=N, COV_X=Sigma_XX, COV_U=Sigma_UU, 
                                    COV_Ex=Sigma_Ex, B=B, X_distribution=X_distribution, Ex_distribution=Ex_distribution,
                                    U_distribution=U_distribution)

    # Step 2: Estimating parameter B:
    
    ##(1) naive OLS method
    B_hat_ols=Y@X_star.T@LA.inv(X_star@X_star.T)
    
    ##(2) naive SCAD-penalized method
    opt_lambda_naive=cv_tuning_pointwise(Y, X_star, Sigma_Ex=np.zeros((q,q)),param_lower_bound=naive_lambda_lb,
                                         param_upper_bound=naive_lambda_ub, plot_points=n_lambda_points, case="naive", 
                                         generate_figure=False)
    #print(f"optimal lambda:{opt_lambda_naive}")
    

    #B_naive_scad, CONVERGE_naive=minimize_BCLSSCAD(tun_para=opt_lambda_naive, Y=Y, X_star=X_star, Sigma_Ex=np.zeros((q,q)),  lower_bound=-100, upper_bound=100,opt_method="trust-constr", case="naive", stop_criterion="threshold")
    B_naive_scad=minimize_LQA_naive(tun_para=opt_lambda_naive, Y=Y, X_star=X_star)
    
    B_naive_scad_original=B_naive_scad.copy()
    
    B_naive_scad[abs(B_naive_scad)<selection_threshold]=0
    
    ##(3) bias corrected SCAD-penalized method
    # opt_lambda=cv_tuning(Y=Y, X_star=X_star, Sigma_Ex=Sigma_Ex, param_lower_bound=0.1, param_upper_bound=1, points_per_round=10, lower_bound=-100, upper_bound=100, opt_method="trust-constr", Round=10, case="corrected", K=5, threshold=0.001)
    # print(f"optimal lambda:{opt_lambda}")
    
    B_correct_scad, CONVERGE_correct=minimize_BCLSSCAD(tun_para=opt_lambda_correct, Y=Y, X_star=X_star, Sigma_Ex=Sigma_Ex, lower_bound=-100, upper_bound=100,opt_method="trust-constr", case="corrected")
    
    B_correct_scad_original=B_correct_scad.copy()
    
    B_correct_scad[abs(B_correct_scad)<selection_threshold]=0
    
    # B_est_list=[B_hat_ols, B_naive_scad, B_correct_scad]
    # plot_title=["naive OLS", "naive SCAD", "bias-corrected SCAD"]
    # i=0
    # for B_est in B_est_list:
    #     fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
    #     #cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
    #     sns.heatmap(B_est, annot=True,#fmt='f',
    #                 # linewidth=.1, 
    #                 # linecolor="black",
    #                 ax=ax,cmap="PiYG", 
    #                 vmin=-10, vmax=10, annot_kws={"fontsize":12})
    #     plt.title(plot_title[i])
    #     i+=1
    
    # Step 3: Model evaluations
    
    ## (1) Estimating accuracy evaluation
    ### (a) MRME
    
    ME_ols=np.trace((B_hat_ols-B)@Sigma_XX@(B_hat_ols-B).T)
    ME_naive=np.trace((B_naive_scad-B)@Sigma_XX@(B_naive_scad-B).T)
    ME_correct=np.trace((B_correct_scad-B)@Sigma_XX@(B_correct_scad-B).T)
    
    ratio_naive_ols=ME_naive/ME_ols
    ratio_correct_ols=ME_correct/ME_ols
    
    ### (b) Frobenius norm of difference between estimate and true
    # for non-zero betas
    nonzero_indicator=(B!=0)
    
    diff_naive_true=B_naive_scad-B
    naive_Fnorm=LA.norm(diff_naive_true*nonzero_indicator)
    
    diff_correct_true=B_correct_scad-B
    correct_Fnorm=LA.norm(diff_correct_true*nonzero_indicator)
    
    ## (2) Evaluating selection accuracy
    ### (a) specificity:
    specificity_naive=np.sum(B_naive_scad[~nonzero_indicator]==0)/np.sum(B==0)
    specificity_correct=np.sum(B_correct_scad[~nonzero_indicator]==0)/np.sum(B==0)
    
    ### (b) sensitivity:
    sensitivity_naive=np.sum(B_naive_scad[nonzero_indicator]!=0)/np.sum(B!=0)
    sensitivity_correct=np.sum(B_correct_scad[nonzero_indicator]!=0)/np.sum(B!=0)
    
    return [naive_Fnorm, correct_Fnorm, 
            ratio_naive_ols, ratio_correct_ols,
            specificity_naive, specificity_correct,
            sensitivity_naive, sensitivity_correct,
            B_naive_scad.reshape(-1), B_correct_scad.reshape(-1),
            B_naive_scad_original.reshape(-1), B_correct_scad_original.reshape(-1),
            CONVERGE_correct
            ]
