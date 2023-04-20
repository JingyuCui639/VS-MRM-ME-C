
#%%
import numpy as np
#from scipy import sparse
import matplotlib.pyplot as plt
from numpy import linalg as LA 
import basic_functions as bf
import seaborn as sns
import time
import pandas as pd
from sklearn.model_selection import KFold

#read in the tobacco data
tobacco_data = np.loadtxt("tobacco_leaf.txt")

#standardization
data_centered=tobacco_data-tobacco_data.mean(axis=0)
data_normalized=data_centered/data_centered.std(axis=0)
np.set_printoptions(suppress=True)
print(data_normalized)
# double-check 
print(data_normalized.std(axis=0))
data_normalized.mean(axis=0)

#%%
data_normalized.shape
#%%
Y=data_normalized[:, :3].T
X_star=data_normalized[:, 3:].T
p,n=Y.shape
q=X_star.shape[0]
# sample/ empirical covariance for 6 predictors
X_samp_cov=X_star@X_star.T/n
sns.heatmap(X_samp_cov)

#%%
## naive SCAD-penalized method on (X*, Y)
# naive optimal_lambda=0.1
start=time.time()
opt_lambda_naive=bf.cv_tuning_pointwise(Y=Y, X_star=X_star, Sigma_Ex=np.zeros((q,q)),param_lower_bound=0.005,
                                        param_upper_bound=0.2, plot_points=196, case="naive", 
                                        generate_figure=True, K=10)
end=time.time()
print(f"optimal lambda for naive method is {opt_lambda_naive}")
print(f"time used: {(end-start)/60} mins.")
#%%
B_naive_scad=bf.minimize_LQA_naive(tun_para=0.1, Y=Y, X_star=X_star)

fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
#cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
sns.heatmap(B_naive_scad, annot=True,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="PiYG", 
            vmin=-0.65, vmax=0.65, 
            annot_kws={"fontsize":16})
#%%
B_naive_scad.min(), B_naive_scad.max()
#%%
selection_threshold=10**(-6)
B_naive_scad[abs(B_naive_scad)<selection_threshold]=0
#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
sns.heatmap(B_naive_scad, annot=True,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="PiYG", 
            vmin=-0.65, vmax=0.65, 
            annot_kws={"fontsize":16})
plt.show()

# %%
#Scenario 1

# error level=0.1: opt_lamnda= 0.252 (0.1-0.3, 101 points, K=10)
# error level=0.15: opt_lambda=0.275 (0.1-0.4, 61 points, K=10)
# error level=0.2: opt_lamnda=0.31 (0.1-0.5, 81 points, K=10)
# error level=0.25: opt_lambda=0.345 (0.1-0.5, 81 points, K=10)
# error level=0.3: opt_lamnda=0.47 (0.2-0.6, 81 points, K=10)
# error level=0.35: opt_lambda=0.575 (0.3-0.9, 121 points, K=10)
# error level=0.4: opt_lamnda=0.695 (0.5-1, 101 points, k=10)
error_level_1=0.4
Sigma_Ex=error_level_1*X_samp_cov
#%%
## bias-corrected SCAD-penalized method on (X*, Y)
time1=time.time()
opt_lambda_corrected=bf.cv_tuning_pointwise(Y=Y, X_star=X_star, Sigma_Ex=Sigma_Ex, param_lower_bound=0.1,
                                        param_upper_bound=0.3, plot_points=201, case="corrected", 
                                        generate_figure=True, K=5)
time2=time.time()
print(f"time used: {(time2-time1)/60} mins.")
#%%
opt_lambda_corrected

#%%
B_correct_scad_4, CON=bf.minimize_BCLSSCAD(tun_para=0.695, Y=Y, X_star=X_star, Sigma_Ex=Sigma_Ex, lower_bound=-100, upper_bound=100,opt_method="trust-constr", case="corrected")
print(CON)
#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
sns.heatmap(B_correct_scad_4, annot=True,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="PiYG", 
            vmin=-0.65, vmax=0.65, 
            annot_kws={"fontsize":16})

#%%
selection_threshold=10**(-6)
B_correct_scad_4[abs(B_correct_scad_4)<selection_threshold]=0
#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
sns.heatmap(B_correct_scad_4, annot=True,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="PiYG", 
            vmin=-0.65, vmax=0.65, 
            annot_kws={"fontsize":16})

#%%
corrected_est_list=[B_correct_scad_1, B_correct_scad_15,
                    B_correct_scad_2, B_correct_scad_25,
                    B_correct_scad_3, B_correct_scad_35]
#,                    B_correct_scad_4]

F_norm_diff_list=[]

for corrected_est in corrected_est_list:
    F_norm_diff_list.append(LA.norm(B_naive_scad-corrected_est))
    
print(F_norm_diff_list)

#%%
fig=plt.subplots(figsize=(8, 5), dpi=200)
x=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35]#, 0.4]    
#plt.plot(x, F_norm_diff_list, marker="o", markersize=12,linestyle="-.", linewidth=3,label="Scenario 1: proportional")
plt.plot(x, F_norm_diff_list ,marker=".",markersize=12, linestyle=":", linewidth=2)
#plt.legend(loc='best', fontsize=20)
#plt.xlabel("$\xi$")
#plt.ylabel("$\|\hat{B}_{x^*c}-\hat{B}_{x^*}\|$")
plt.show()