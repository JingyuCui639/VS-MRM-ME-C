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

# generate true B
B_true=np.array([[6, 0, 8, 0,10, 0, 0, -2, 0, 0],	
                 [0,6, 0,5,  0, 7, 0,  0,-4, 0],	
                 [0, 0, 6, 0, 2, 0,	-1.5,  0, 0,-9],
                 [0, 0, 0,1.5, 0, -2, 0, -3, 0, 0],		
                 [7, 0, 0, 0, -4, 0, -6,  0,-3, 0],	
                 [10,2, 0, 0, 0,-7, 0,	-9, 0, -4]])

# plot matrix of true B
# fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
# #cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
# sns.heatmap(B_true, annot=True,#fmt='f',
#             # linewidth=.1, 
#             # linecolor="black",
#             ax=ax,cmap="PiYG", 
#             vmin=-10, vmax=10, annot_kws={"fontsize":12})

def main():
    p=6
    q=10
    N=3000
    simu_num=500
    np.random.seed(10)
    merror_level_list=[0.2, 0.5, 0.8]
    X_distribution, Ex_distribution, U_distribution=["Uniform", "Gamma", "chi-square"] # ["Uniform", "Gamma", "chi-square"] ["Normal", "Normal", "Normal"]
    # opt_lambda_list_corrected=[0.37, 0.59, 0.835] # for Normal case N=3000 #done!
    opt_lambda_list_corrected=[0.38, 0.555, 0.83] # for Non-Normal case N=3000 
    # opt_lambda_list_corrected=[0.5, 0.835, 1.335] # for Normal case N=1000 #done!
    # opt_lambda_list_corrected=[0.48, 0.86, 1.335] # for Non-Normal case N=1000 #done!
    # opt_lambda_list_corrected=[0.67, 1.265, 2.27] # for Normal case N=500 #done!
    # opt_lambda_list_corrected=[0.655, 1.26, 2.375] # for Non-Normal case N=500  # done!
    selection_threshold=10**(-6)
    
    # # Tuning parameter range for the naive method
    # # for n=500 and 1000
    # naive_lambda_lb=0.005
    # naive_lambda_ub=0.6
    # n_lambda_points=120
    
    # for the case of n=3000
    naive_lambda_lb=0.001
    naive_lambda_ub=0.2
    n_lambda_points=100

    start_time=time.time()
    
    results_table=np.zeros((6,4))

    i=0
    for error_level in merror_level_list:
        
        print(f"measurement error level: {error_level}")
        
        opt_lambda_correct=opt_lambda_list_corrected[i]
        
        # step 1: generate covariances
        Sigma_XX=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[1], power_base=0.7)
        Sigma_UU=bf.cov_generator(dim=p, MODEL="power", decreasing_rate=1, variance=[0.1], power_base=0.3)
        Sigma_Ex=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[error_level], power_base=0.5)
        
        # Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())
        single_simu_result_list=[]
        single_simu_result_list=pool.starmap_async(bf.single_simulation, [(p, q, N, B_true, Sigma_XX, Sigma_UU, 
                                                                        Sigma_Ex,opt_lambda_correct, X_distribution, 
                                                                        Ex_distribution, U_distribution, selection_threshold, 
                                                                        naive_lambda_lb, naive_lambda_ub, n_lambda_points) 
                                                                        for k in range(simu_num)]).get()
        pool.close()
        
        single_simu_result_array=np.array(single_simu_result_list)
        print(f"The total converged cases: {np.sum(single_simu_result_array[:,12])}")
        
        #save the entire 500 results
        single_simu_result_array_df=pd.DataFrame(single_simu_result_array)
        single_simu_result_array_df.to_csv("N_"+str(N)+"_entire_raw_results_"+X_distribution+"_error_level_"+str(error_level)+".csv")
        
        #print(single_simu_result_array[:,:8])
        mean_naive_Fnorm=single_simu_result_array[:,0].mean()
        mean_correct_Fnorm=single_simu_result_array[:,1].mean()
        median_ratio_naive_ols=np.median(single_simu_result_array[:,2])
        median_ratio_correct_ols=np.median(single_simu_result_array[:,3])
        mean_specificity_naive=single_simu_result_array[:,4].mean()
        mean_specificity_correct=single_simu_result_array[:,5].mean()
        mean_sensitivity_naive=single_simu_result_array[:,6].mean()
        mean_sensitivity_correct=single_simu_result_array[:,7].mean()
        
        results_table[i*2,]=[mean_naive_Fnorm, median_ratio_naive_ols, mean_specificity_naive, mean_sensitivity_naive]
        results_table[i*2+1,]=[mean_correct_Fnorm, median_ratio_correct_ols, mean_specificity_correct, mean_sensitivity_correct]
        
        B_naive_scad_est=np.zeros((simu_num, 60))
        B_naive_scad_est_orig=np.zeros((simu_num, 60))
        
        B_correct_scad_est=np.zeros((simu_num, 60))
        B_correct_scad_est_orig=np.zeros((simu_num, 60))
        
        for j in range(simu_num):
            B_naive_scad_est[j,]=single_simu_result_array[j,8]
            B_correct_scad_est[j,]=single_simu_result_array[j,9]
            B_naive_scad_est_orig[j,]=single_simu_result_array[j,10]
            B_correct_scad_est_orig[j,]=single_simu_result_array[j,11]
        
        B_naive_scad_est_df=pd.DataFrame(B_naive_scad_est)
        B_naive_scad_est_df.to_csv("N_"+str(N)+"_B_naive_scad_est_thres_"+X_distribution+"_error_level_"+str(error_level)+".csv")
        
        B_correct_scad_est_df=pd.DataFrame(B_correct_scad_est)
        B_correct_scad_est_df.to_csv("N_"+str(N)+"_B_correct_scad_est_thres_"+X_distribution+"_error_level_"+str(error_level)+".csv")
        
        B_naive_scad_est_orig_df=pd.DataFrame(B_naive_scad_est_orig)
        B_naive_scad_est_orig_df.to_csv("N_"+str(N)+"_B_naive_scad_est_Orig_"+X_distribution+"_error_level_"+str(error_level)+".csv")
        
        B_correct_scad_est_orig_df=pd.DataFrame(B_correct_scad_est_orig)
        B_correct_scad_est_orig_df.to_csv("N_"+str(N)+"_B_correct_scad_est_Orig_"+X_distribution+"_error_level_"+str(error_level)+".csv")
        
        i+=1
    
    results_table_df=pd.DataFrame(results_table) 
    results_table_df.to_csv("N_"+str(N)+"_results_table_"+X_distribution+"3.csv")
        
    end_time=time.time()
    print(f"time used: {(end_time-start_time)/60} mins.")


if __name__ == "__main__":
    main()