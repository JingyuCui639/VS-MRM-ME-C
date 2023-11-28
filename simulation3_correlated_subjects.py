#%%
import numpy as np
#from scipy import sparse
#import matplotlib.pyplot as plt
#from numpy import linalg as LA 
import basic_functions as bf
#import seaborn as sns
import time
import pandas as pd
import multiprocessing as mp
import scipy.stats as ss

# Full matrix B
# full_B=np.array([[3, 0, -0.8, 0, 1.5],
#                  [0, 0.8, 0, -3, 0],
#                  [-1.5, 0, 2, 0, -1]])

full_B=np.array([[ 7, 0, -3, 0, -6, 0, 10,   0, -7,  0, -2.5, 0, 7],
 [ 0,  1.2, 0,  -3,  0,1.5,  0,   5,  0,   -2,  0, 6,   0 ],
 [  -1.7, 0 ,4,   0,  6, 0,  -1.3,   0,  2,  0, 2,  0,  -7],
 [ 0,   -4,  0,  5,  0, -10, 0,   -2,   0,  6, 0,    -4, 0 ]])


# plot matrix of true B
# fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
# #cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
# sns.heatmap(B_true, annot=True,#fmt='f',
#             # linewidth=.1, 
#             # linecolor="black",
#             ax=ax,cmap="PiYG", 
#             vmin=-10, vmax=10, annot_kws={"fontsize":12})

def main():
    p=4
    q=13 # q_list=[5, 7, 8, 10, 13]
    N=5000 # N_list=[200, 500, 1000, 2000, 5000]
    B_true=full_B[:p, :q]
    simu_num=512
    simu_num_kept=500
    error_level=0.3
    np.random.seed(10)
    # merror_level_list=[0.3, 0.7]
    X_distribution, Ex_distribution, U_distribution=["Normal", "Normal", "Normal"] # ["Uniform", "Gamma", "chi-square"] ["Normal", "Normal", "Normal"]

    selection_threshold=10**(-6)
    dependency_case_list=["independent", "covariates", "ME" ,"covariates and ME"]
    
    # tuning parameter range: 
    lambda_lb=0.01
    lambda_ub=0.5
    n_lambda_points=50
    opt_method_subobj=None
    initial_case="zeros"

    start_time=time.time()
    
    results_table=np.zeros((8,4))

    i=0
    
    #opt_lambda_correct=opt_lambda_list_corrected[i]
    
    # step 1: generate covariances
    Sigma_XX=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[1], power_base=0.7)
    Sigma_UU=bf.cov_generator(dim=p, MODEL="power", decreasing_rate=1, variance=[0.1], power_base=0.3)
    Sigma_Ex=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[error_level], power_base=0.5)
    Sigma_Xn=bf.cov_generator(dim=N, MODEL="power", decreasing_rate=1, variance=[1], power_base=0.2)
    Sigma_En=bf.cov_generator(dim=N, MODEL="power", decreasing_rate=1, variance=[error_level], power_base=0.2)
    
    for dependency_case in dependency_case_list:
        
        print(f"dependency case: {dependency_case}")
        
        # Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())
        single_simu_result_list=[]
        single_simu_result_list=pool.starmap_async(bf.single_simulation3, [(dependency_case, p, q, N, B_true, Sigma_XX, Sigma_UU, 
                                                                        Sigma_Ex, Sigma_Xn, Sigma_En, initial_case, 
                                                                        opt_method_subobj, 
                                                                        X_distribution, 
                                                                        Ex_distribution, U_distribution, selection_threshold, 
                                                                        lambda_lb, lambda_ub, n_lambda_points) 
                                                                        for k in range(simu_num)]).get()

        
        pool.close()
    
        single_simu_result_array_all=np.array(single_simu_result_list, dtype=object)
        print(f"The total converged cases: {np.sum(single_simu_result_array_all[:,12]*single_simu_result_array_all[:,13])} out of {simu_num}")
        
        #save the entire 500 results
        column_names=["Frobratio_naive_ols", "Frobratio_correct_ols", 
            "MRME_naive_ols", "MRME_correct_ols",
            "specificity_naive", "specificity_correct",
            "sensitivity_naive", "sensitivity_correct",
            "B_naive_scad", "B_correct_scad",
            "B_naive_scad_original", "B_correct_scad_original", "CONVERGE_naive",
            "CONVERGE_correct", "opt_lambda_naive", "opt_lambda_correct", "B_hat_ols"]

        
        single_simu_result_array_df=pd.DataFrame(single_simu_result_array_all, columns=column_names)
        
        single_simu_result_array_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_entire_raw_results_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        single_simu_result_array=single_simu_result_array_all[:simu_num_kept,:]
        
        #print(single_simu_result_array[:,:8])
        median_naive_Fnorm=np.median(single_simu_result_array[:,0])
        median_correct_Fnorm=np.median(single_simu_result_array[:,1])
        median_ratio_naive_ols=np.median(single_simu_result_array[:,2])
        median_ratio_correct_ols=np.median(single_simu_result_array[:,3])
        mean_specificity_naive=single_simu_result_array[:,4].mean()
        mean_specificity_correct=single_simu_result_array[:,5].mean()
        mean_sensitivity_naive=single_simu_result_array[:,6].mean()
        mean_sensitivity_correct=single_simu_result_array[:,7].mean()
        
        results_table[i*2,]=[median_naive_Fnorm, median_ratio_naive_ols, mean_specificity_naive, mean_sensitivity_naive]
        results_table[i*2+1,]=[median_correct_Fnorm, median_ratio_correct_ols, mean_specificity_correct, mean_sensitivity_correct]
        
        B_ols_est=np.zeros((simu_num, p*q))
        
        B_naive_scad_est=np.zeros((simu_num, p*q))
        B_naive_scad_est_orig=np.zeros((simu_num, p*q))
        
        B_correct_scad_est=np.zeros((simu_num, p*q))
        B_correct_scad_est_orig=np.zeros((simu_num, p*q))
        
        for j in range(simu_num):
            B_naive_scad_est[j,]=single_simu_result_array_all[j,8]
            B_correct_scad_est[j,]=single_simu_result_array_all[j,9]
            B_naive_scad_est_orig[j,]=single_simu_result_array_all[j,10]
            B_correct_scad_est_orig[j,]=single_simu_result_array_all[j,11]
            B_ols_est[j,]=single_simu_result_array_all[j,16]
        
        B_ols_est_df=pd.DataFrame(B_ols_est)
        B_ols_est_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_B_ols_est_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        B_naive_scad_est_df=pd.DataFrame(B_naive_scad_est)
        B_naive_scad_est_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_B_naive_scad_est_thres_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        B_correct_scad_est_df=pd.DataFrame(B_correct_scad_est)
        B_correct_scad_est_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_B_correct_scad_est_thres_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        B_naive_scad_est_orig_df=pd.DataFrame(B_naive_scad_est_orig)
        B_naive_scad_est_orig_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_B_naive_scad_est_Orig_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        B_correct_scad_est_orig_df=pd.DataFrame(B_correct_scad_est_orig)
        B_correct_scad_est_orig_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_B_correct_scad_est_Orig_"+X_distribution+"_dependency_case_"+str(dependency_case)+".csv")
        
        i+=1
    
    col_names=["Median_Fnorm_ratio", "MRME", "avg_specificity", "avg_sensitivity"]
    index_labels=["independent, p-naive", "independent , proposed", "covariates, p-naive", "covariates, proposed", "ME, p-naive", "ME, proposed", "covariates+ME, p-naive", "covariates+ME, proposed"]
    results_table_df=pd.DataFrame(results_table, columns=col_names, index=index_labels) 
    results_table_df.to_csv("(revis simu 3-2ed) N_"+str(N)+"_results_table_"+X_distribution+".csv")
        
    end_time=time.time()
    print(f"time used: {(end_time-start_time)/60} mins.")


if __name__ == "__main__":
    main()
