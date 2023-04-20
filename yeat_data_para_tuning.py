#%%
import numpy as np
import pandas as pd
from numpy import diag, linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import basic_functions as bf
import time 
from sklearn.model_selection import KFold
import multiprocessing as mp


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def single_SCAD_estimation(i, Y_data, X_data, Sigma_Ex, param, KF, case="corrected"):
    #
    print(f"i={i}")
    print(f"lambda={param}")
    loss_per_param=[]
    for train_ind, test_ind in KF.split(Y_data.T):
            #print(f"Fold: {i}")
            
            #define the training and test set for each fold fitting
            X_train, Y_train=X_data[:,train_ind], Y_data[:,train_ind]
            X_test, Y_test=X_data[:,test_ind], Y_data[:,test_ind]
            
            if case=="corrected":
            
                B_est=bf.minimize_BCLSSCAD(tun_para=param, Y=Y_train, X_star=X_train,
                                        Sigma_Ex=Sigma_Ex, lower_bound=-100,
                                        upper_bound=100, opt_method="trust-constr", 
                                        case=case)
            elif case=="naive":
                B_est=bf.minimize_LQA_naive(tun_para=param, Y=Y_train, X_star=X_train)
            
            n=len(test_ind)
            #loss_per_param.append(-np.trace(X_test@Y_test.T@B_est)+0.5*np.trace((X_test@X_test.T-n*Sigma_Ex)@B_est.T@B_est))
            loss_per_param.append(LA.norm(Y_test-B_est@X_test)**2-n*np.trace(Sigma_Ex@B_est.T@B_est))
    
    return [i, np.mean(loss_per_param)]


def main():
    #read in the data
    X=pd.read_csv("X_yeast.csv", index_col=[0])
    Y=pd.read_csv("Y_yeast.csv", index_col=[0])

    N_Obs=X.shape[0]
    p=Y.shape[1]
    q=X.shape[1]

    #%%
    #centered X and Y
    X_center=X.apply(lambda x: x-x.mean())
    #X_center=np.array(X_center.T)
    # print(X_center.mean(axis=0))
    # print(X_center.shape)
    Y_center=Y.apply(lambda x: x-x.mean())
    #Y_center=np.array(Y_center.T)
    # print(Y_center.mean(axis=0))
    # print(Y_center.shape)

    #%%
    #standardize data
    X_normalized=X_center/X_center.std(axis=0)
    # print(X_normalized.std(axis=0))
    # print(X_normalized.shape)
    Y_normalized=Y_center/Y_center.std(axis=0)
    # print(Y_normalized.std(axis=0))
    # print(Y_normalized.shape)

    #%%
    # data matrix transpose 
    Y_data=np.array(Y_normalized.T)
    X_star_data=np.array(X_normalized.T)
    #print(Y_data.shape, X_star_data.shape)
    #%%
    # sample/ empirical covariance covariates
    X_samp_cov=X_star_data@X_star_data.T/N_Obs
    #sns.heatmap(X_samp_cov)
    #print(X_samp_cov.min(), X_samp_cov.max())

    #%%
    ##############################################
    # bias-corrected SCAD penalized 
    ##############################################

    error_level_1=0.4
    Sigma_Ex=error_level_1*X_samp_cov
    K=20
    param_lower_bound=0.01
    param_upper_bound=0.6
    plot_points=60
    parameter_list=np.linspace(param_lower_bound, param_upper_bound, plot_points)

    time1=time.time()

    kf=KFold(n_splits=K, shuffle=True, random_state=1)  

    pool = mp.Pool(mp.cpu_count())
    
    loss_list_result= [pool.apply_async(single_SCAD_estimation, args=(i, Y_data, X_star_data, Sigma_Ex, param, kf)) for i, param in enumerate(parameter_list)]
    
    #sort the results
    # loss_list_result.sort(key=lambda x: x[0])
    # print(f"Sorted results: {loss_list_result}")
    order_list=[r.get()[0] for r in loss_list_result]
    loss_list_result_final = [r.get()[1] for r in loss_list_result]

    pool.close() 
    pool.join()

    results_array=np.array([order_list, loss_list_result_final]).T
    print(results_array)
    print(results_array.shape)

    results_array=results_array[results_array[:,0].argsort()]
    print(results_array)

    ordered_loss_list=results_array[:,1]

    time2=time.time()
    print(f"time used: {(time2-time1)/60} mins.") 

    low_index=np.where(ordered_loss_list==min(ordered_loss_list))
    opt_lambda=parameter_list[low_index][0]
    print(f"optimal lambda is {opt_lambda}")

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(parameter_list, ordered_loss_list)
    plt.show()


if __name__ == "__main__":
    main()
    
