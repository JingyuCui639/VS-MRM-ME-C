#%%
import numpy as np
#from scipy import sparse
import matplotlib.pyplot as plt
from numpy import linalg as LA 
import basic_functions as bf
from sklearn.model_selection import KFold
import seaborn as sns
import time
import pandas as pd
import multiprocessing as mp
#%%
def single_tuning(loop_id, p, q, N, B, Sigma_XX, Sigma_UU, Sigma_Ex, 
                    X_distribution, Ex_distribution, U_distribution,
                    lambda_lb, lambda_ub, n_points_in_plot, case):

        print(f"random sample {loop_id+1}")
        Y, X_star, X, U=bf.data_generator(dim_Y=p, dim_X=q, N_Obs=N, COV_X=Sigma_XX, COV_U=Sigma_UU, 
                                            COV_Ex=Sigma_Ex, B=B, X_distribution=X_distribution, Ex_distribution=Ex_distribution,
                                            U_distribution=U_distribution)
        if case=="naive":
            tuning_Sigma_Ex=np.zeros((q,q))
        elif case=="corrected":
            tuning_Sigma_Ex=Sigma_Ex
        # start_time=time.time()
        #bf.cv_tuning_plot(Y, X_star=X_star,Sigma_Ex=Sigma_E2,plot_points=50,lower_bound=-1000, upper_bound=1000, opt_method="Powell",K=5)
        loss_list=bf.cv_tuning_plot(Y=Y, X_star=X_star, Sigma_Ex=tuning_Sigma_Ex, 
                        param_lower_bound=lambda_lb, param_upper_bound=lambda_ub, 
                        plot_points=n_points_in_plot, lower_bound=-100, 
                        upper_bound=100, opt_method="trust-constr", 
                        figure_name="cv_plot_sample_"+str(loop_id+1),case=case,K=5)
        # end_time=time.time()
        # time_used=(end_time-start_time)/3600
        # print(f"time used: {time_used} hours, finish one loop")


        #print(f"minimum loss :{min(loss_list)}")
        low_index=np.where(loss_list==min(loss_list))
        opt_lambda_i=np.linspace(lambda_lb, lambda_ub, n_points_in_plot)[low_index][0]
        print(f"optimal lambda for sample {loop_id+1} is {opt_lambda_i}")
        return opt_lambda_i


#%%
def main():

    B_true=np.array([[6, 0, 8, 0,10, 0, 0, -2, 0, 0],	
                    [0,6, 0,5,  0, 7, 0,  0,-4, 0],	
                    [0, 0, 6, 0, 2, 0,	-1.5,  0, 0,-9],
                    [0, 0, 0,1.5, 0, -2, 0, -3, 0, 0],		
                    [7, 0, 0, 0, -4, 0, -6,  0,-3, 0],	
                    [10,2, 0, 0, 0,-7, 0,	-9, 0, -4]])


    # fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
    # #cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
    # sns.heatmap(B_true, annot=True,#fmt='f',
    #             # linewidth=.1, 
    #             # linecolor="black",
    #             ax=ax,cmap="PiYG", 
    #             vmin=-10, vmax=10, annot_kws={"fontsize":12})


    #define dimensions
    p=6
    q=10
    N=1000
    loop_num=100
    merror_level_list=[0.2, 0.5, 0.8]
    error_level=0.8
    lambda_lb=0.5
    lambda_ub=2
    n_points_in_plot=151
    X_distribution, Ex_distribution, U_distribution=["Uniform", "Gamma", "chi-square"] #["Normal", "Normal", "Normal"] #["Uniform", "Gamma", "chi-square"]
    case="corrected"

    #step 1: generate covariances
    Sigma_XX=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[1], power_base=0.7)
    Sigma_UU=bf.cov_generator(dim=p, MODEL="power", decreasing_rate=1, variance=[0.1], power_base=0.3)
    Sigma_Ex=bf.cov_generator(dim=q, MODEL="power", decreasing_rate=1, variance=[error_level], power_base=0.5)

    np.random.seed(1)
    # Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    start_time_all=time.time()
    
    opt_lambda_list=[]

    opt_lambda_list = pool.starmap_async(single_tuning, [(loop_id, p, q, N, B_true, Sigma_XX, Sigma_UU, Sigma_Ex, 
                    X_distribution, Ex_distribution, U_distribution,
                    lambda_lb, lambda_ub, n_points_in_plot, case) for loop_id in range(loop_num)]).get()

    pool.close()


    # for i in range(loop_num):
        
    #     single_tuning_result=single_tuning(loop_num=i)
        
    #     print(single_tuning_result)

        # print(f"random sample {i+1}")

        # Y, X_star, X, U=bf.data_generator(dim_Y=p, dim_X=q, N_Obs=N, COV_X=Sigma_XX, COV_U=Sigma_UU, 
        #                                     COV_Ex=Sigma_Ex, B=B_true, X_distribution="Normal", Ex_distribution="Normal",
        #                                     U_distribution="Normal")
        
        # start_time=time.time()
        # #bf.cv_tuning_plot(Y, X_star=X_star,Sigma_Ex=Sigma_E2,plot_points=50,lower_bound=-1000, upper_bound=1000, opt_method="Powell",K=5)
        # loss_list=bf.cv_tuning_plot(Y=Y, X_star=X_star, Sigma_Ex=Sigma_Ex, 
        #                 param_lower_bound=lambda_lb, param_upper_bound=lambda_ub, 
        #                 plot_points=n_points_in_plot, lower_bound=-100, 
        #                 upper_bound=100, opt_method="trust-constr", 
        #                 figure_name="cv_plot_sample_"+str(i+1),case="corrected",K=5)
        # end_time=time.time()
        # time_used=(end_time-start_time)/3600
        # print(f"time used: {time_used} hours")


        # print(f"minimu loss :{min(loss_list)}")
        # low_index=np.where(loss_list==min(loss_list))
        # opt_lambda_i=np.linspace(lambda_lb, lambda_ub, n_points_in_plot)[low_index][0]
        # print(f"optimal lambda for sample {i+1} is {opt_lambda_i}")
        # opt_lambda_list.append(single_tuning_result)

    print(f"list of optimal lambda values are {opt_lambda_list}")
    print(f"The median of lambda is {np.median(opt_lambda_list)}")
    print(f"The mean of lambda is {np.mean(opt_lambda_list)}")
    print(f"The min lambda is {np.min(opt_lambda_list)}")
    print(f"The max lambda is {np.max(opt_lambda_list)}")
    
    end_time_all=time.time()
    all_times_used=(end_time_all-start_time_all)/60
    print(f"Total time used: {all_times_used} mins")
    
    plt.hist(opt_lambda_list)
    plt.show()

    #loss_df=pd.DataFrame(loss_list_store)
    #loss_df.to_csv("loss_"+str(loop_num)+"loops_normal_errlevel08_range_02_06_101.csv")
    #print(f"The loss list values are stored.")



if __name__ == "__main__":
    main()


#%%
# # list of optimal lambda for Normal + error level=0.2
# lambda_list=[0.41414141414141414, 0.4585858585858586, 0.46262626262626266, 0.5232323232323233, 0.4101010101010101,
# 0.4868686868686869, 0.4505050505050505, 0.37777777777777777, 0.37373737373737376, 0.38181818181818183, 
# 0.4222222222222222, 0.3292929292929293, 0.44242424242424244, 0.397979797979798, 0.45454545454545453, 
# 0.47070707070707074, 0.4222222222222222, 0.43434343434343436, 0.3292929292929293, 0.40202020202020206, 
# 0.3656565656565657, 0.4101010101010101, 0.345454529292929293, 0.44242424242424244, 0.397979797979798, 
# 0.45454545454545453, 0.47070707070707074, 0.4222222222222222, 0.43434343434343436, 0.3292929292929293, 
# 0.40202020202020206, 0.3656565656565657, 0.4101010101010101, 0.34545454545454546, 0.4787878787878788, 
# 0.4222222222222222]

# print(np.median(lambda_list))
# #0.418
#%%
# list=[1.1, 0.85, 0.85, 0.85, 0.85, 1.1, 0.85, 1.1]
# np.median(list)