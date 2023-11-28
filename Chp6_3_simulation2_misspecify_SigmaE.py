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
    q=10 
    N=2000 
    B_true=full_B[:p, :q]
    simu_num=508
    simu_num_kept=500
    error_level=0.5
    np.random.seed(10)
    # merror_level_list=[0.3, 0.7]
    X_distribution, Ex_distribution, U_distribution=["Normal", "Normal", "Normal"] # ["Uniform", "Gamma", "chi-square"] ["Normal", "Normal", "Normal"]
    # misspecification_list=["true", "zero", "diagonal", "under estimate", "over estimate"]
    selection_threshold=10**(-6)
    
    # tuning parameter range
    lambda_lb=0.01
    lambda_ub=0.5
    n_lambda_points=50
    opt_method_subobj=None
    initial_case="zeros"

    start_time=time.time()
    
    #opt_lambda_correct=opt_lambda_list_corrected[i]
    
    # step 1: generate covariances
    Sigma_XX=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[1], power_base=0.7)
    Sigma_UU=bf.cov_generator(dim=p, MODEL="power", decreasing_rate=1, variance=[0.1], power_base=0.3)
    Sigma_Ex=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[error_level], power_base=0.5)
    
    # Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    single_simu_result_list=[]
    single_simu_result_list=pool.starmap_async(bf.single_simulation2, [(p, q, N, B_true, Sigma_XX, Sigma_UU, 
                                                                    Sigma_Ex, initial_case, 
                                                                    opt_method_subobj, 
                                                                    X_distribution, 
                                                                    Ex_distribution, U_distribution, selection_threshold, 
                                                                    lambda_lb, lambda_ub, n_lambda_points) 
                                                                    for k in range(simu_num)]).get()
    
    pool.close()
    
    single_simu_result_array_all=np.array(single_simu_result_list, dtype="object")
    print(f"The total converged cases: {np.sum(single_simu_result_array_all[:,-1])} out of {simu_num} cases")
    
    # save the entire simulation results, unconverging case included
    column_names=["trueSig_FR", "zeroSig_FR",  "underSig_FR", "overSig_FR",
            "MEratio_trueSig_ols", "MEratio_zeroSig_ols",  "MEratio_underSig_ols", "MEratio_overSig_ols",
            "specificity_trueSig", "specificity_zeroSig",  "specificity_underSig", "specificity_overSig",
            "sensitivity_trueSig", "sensitivity_zeroSig",  "sensitivity_underSig", "sensitivity_overSig",
            "trueSig_B_estimates", "zeroSig_B_estimates",  "underSig_B_estimates", "overSig_B_estimates",
            "CONVERGE"]
    single_simu_result_array_df=pd.DataFrame(single_simu_result_array_all, columns=column_names)
    
    single_simu_result_array_df.to_csv("(revis simu 2-2ed) N_"+str(N)+"_entire_raw_results_"+X_distribution+"_error_level_"+str(error_level)+".csv")
    
    # extracting the first 500 converging cases
    index=np.array(single_simu_result_array_all[:,-1], dtype=bool)
    single_simu_result_array1=single_simu_result_array_all[index,:]
    single_simu_result_array=single_simu_result_array1[:simu_num_kept,:16]
    print(f"# rows in the collected data: {single_simu_result_array.shape[0]}")
    
    #save the 500 estimates for 5 cases
    B_est_trueSig=np.zeros((simu_num_kept, p*q))
    B_est_zeroSig=np.zeros((simu_num_kept, p*q))
    # B_est_diagSig=np.zeros((simu_num_kept, p*q))
    B_est_underSig=np.zeros((simu_num_kept, p*q))
    B_est_overSig=np.zeros((simu_num_kept, p*q))
    
    for j in range(simu_num_kept):
        B_est_trueSig[j,]=single_simu_result_array1[j,16]
        B_est_zeroSig[j,]=single_simu_result_array1[j,17]
        # B_est_diagSig[j,]=single_simu_result_array1[j,]
        B_est_underSig[j,]=single_simu_result_array1[j,18]
        B_est_overSig[j,]=single_simu_result_array1[j,19]
    
    B_est_trueSig_df=pd.DataFrame(B_est_trueSig)
    B_est_trueSig_df.to_csv("(revis simu 2-2ed) B_est_trueSig_"+X_distribution+".csv")
    
    B_est_zeroSig_df=pd.DataFrame(B_est_zeroSig)
    B_est_zeroSig_df.to_csv("(revis simu 2-2ed) B_est_zeroSig_"+X_distribution+".csv")
    
    # B_est_diagSig_df=pd.DataFrame(B_est_diagSig)
    # B_est_diagSig_df.to_csv("(revis simu 2-ed) B_est_diagSig_"+X_distribution+".csv")
    
    B_est_underSig_df=pd.DataFrame(B_est_underSig)
    B_est_underSig_df.to_csv("(revis simu 2-2ed) B_est_underSig_"+X_distribution+".csv")
    
    B_est_overSig_df=pd.DataFrame(B_est_overSig)
    B_est_overSig_df.to_csv("(revis simu 2-2ed) B_est_overSig_"+X_distribution+".csv")
    
    means=single_simu_result_array.mean(axis=0)
    medians=np.median(single_simu_result_array,axis=0)
    
    results_table=means.reshape(4,4).T
    print(f"mean results_table:{results_table}")
    results_table_median=medians.reshape(4,4).T
    print(f"median result table: {results_table_median}")
    
    results_table[:,0]=results_table_median[:,0]
    results_table[:,1]=results_table_median[:,1]
    print(f"updated results table:{results_table}")

    # B_naive_scad_est=np.zeros((simu_num, p*q))
    # B_naive_scad_est_orig=np.zeros((simu_num, p*q))
    
    # B_correct_scad_est=np.zeros((simu_num, p*q))
    # B_correct_scad_est_orig=np.zeros((simu_num, p*q))
    
    # for j in range(simu_num):
    #     B_naive_scad_est[j,]=single_simu_result_array[j,8]
    #     B_correct_scad_est[j,]=single_simu_result_array[j,9]
    #     B_naive_scad_est_orig[j,]=single_simu_result_array[j,10]
    #     B_correct_scad_est_orig[j,]=single_simu_result_array[j,11]
    
    # B_naive_scad_est_df=pd.DataFrame(B_naive_scad_est)
    # B_naive_scad_est_df.to_csv("(revis)N_"+str(N)+"_B_naive_scad_est_thres_"+X_distribution+"_error_level_"+str(error_level)+".csv")
    
    # B_correct_scad_est_df=pd.DataFrame(B_correct_scad_est)
    # B_correct_scad_est_df.to_csv("(revis)N_"+str(N)+"_B_correct_scad_est_thres_"+X_distribution+"_error_level_"+str(error_level)+".csv")
    
    # B_naive_scad_est_orig_df=pd.DataFrame(B_naive_scad_est_orig)
    # B_naive_scad_est_orig_df.to_csv("(revis)N_"+str(N)+"_B_naive_scad_est_Orig_"+X_distribution+"_error_level_"+str(error_level)+".csv")
    
    # B_correct_scad_est_orig_df=pd.DataFrame(B_correct_scad_est_orig)
    # B_correct_scad_est_orig_df.to_csv("(revis)N_"+str(N)+"_B_correct_scad_est_Orig_"+X_distribution+"_error_level_"+str(error_level)+".csv")
    

    
    col_names=["Median_Fnorm_ratio", "MRME", "avg_specificity", "avg_sensitivity"]
    index_labels=["true Sigma_Ex", "structure-zero", "magnitude-under", "magnitude-over"]
    results_table_df=pd.DataFrame(results_table, columns=col_names, index=index_labels) 
    results_table_df.to_csv("(revis simu 2-2ed) N_"+str(N)+"_results_table_"+X_distribution+".csv")
        
    end_time=time.time()
    print(f"time used: {(end_time-start_time)/60} mins.")


if __name__ == "__main__":
    main()
