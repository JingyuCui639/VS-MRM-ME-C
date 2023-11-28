
#%%
import numpy as np
import pandas as pd
from numpy import diag, linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import basic_functions as bf
import time 

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#read in the data
X=pd.read_csv("X_yeast.csv", index_col=[0])
Y=pd.read_csv("Y_yeast.csv", index_col=[0])

N_Obs=X.shape[0]
p=Y.shape[1]
q=X.shape[1]

#%%
print(N_Obs, p, q)
print(X.shape, Y.shape)
#%%
#centered X and Y
X_center=X.apply(lambda x: x-x.mean())
#X_center=np.array(X_center.T)
print(X_center.mean(axis=0))
print(X_center.shape)
Y_center=Y.apply(lambda x: x-x.mean())
#Y_center=np.array(Y_center.T)
print(Y_center.mean(axis=0))
print(Y_center.shape)

#%%
#standardize data
X_normalized=X_center/X_center.std(axis=0)
print(X_normalized.std(axis=0))
print(X_normalized.shape)
Y_normalized=Y_center/Y_center.std(axis=0)
print(Y_normalized.std(axis=0))
print(Y_normalized.shape)

#%%
# data matrix transpose 
Y_data=np.array(Y_normalized.T)
X_star_data=np.array(X_normalized.T)
print(Y_data.shape, X_star_data.shape)

#%%
# sample/ empirical covariance covariates
X_samp_cov=X_star_data@X_star_data.T/N_Obs
sns.heatmap(X_samp_cov)
print(X_samp_cov.min(), X_samp_cov.max())


#%%
####################################################
## naive SCAD-penalized method on (X*, Y)
####################################################
## 1. not standardize, just centered
# naive optimal_lambda=0.052222222222222225
# beta range: -0.47395659600506224 0.9124559834904782
## 2. normalized (both standardize and centered)
# naive optimal lambda = 0.046
# beta range: -0.3718337633700859 0.47756318718273205

start=time.time()
opt_lambda_naive=bf.cv_tuning_pointwise(Y=Y_data, X_star=X_star_data, Sigma_Ex=np.zeros((q,q)),param_lower_bound=0.001,
                                        param_upper_bound=0.2, plot_points=200, case="naive", 
                                        generate_figure=True, K=10)

end=time.time()
#%%
print(f"time used: {(end-start)/3600} hours")
print(f"optimal lambda for naive method is {opt_lambda_naive}")
#%%
B_naive_scad=bf.minimize_LQA_naive(tun_para=0.046, Y=Y_data, X_star=X_star_data)
#%%

print(B_naive_scad.min(), B_naive_scad.max())
print(LA.norm(B_naive_scad, ord=1))
print(LA.norm(B_naive_scad, ord=np.inf))

#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
#cbar_ax = fig.add_axes([.91, .3, .03, .4]) #adjust the location of color bar
sns.heatmap(B_naive_scad, annot=False,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="PiYG", #"PiYG", "seismic"
            vmin=-0.85, vmax=0.85, 
            annot_kws={"fontsize":12})

#%%
selection_threshold=10**(-5)
B_naive_scad[abs(B_naive_scad)<selection_threshold]=0
#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
sns.heatmap(B_naive_scad, annot=False,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="seismic",#"PiYG", 
            vmin=-0.85, vmax=0.85, 
            annot_kws={"fontsize":12})
plt.show()

# %%
##############################################
# bias-corrected SCAD penalized 
##############################################

# error level=0.1: opt_lamnda= 0.09 (range: 0.01-1, 100 points, K=10) sure!!!
# error level=0.2: opt_lamnda= 0.12 (range: 0.01-5, 50 points, K=10) sure!!!
# error level=0.3: opt_lamnda= 0.16 (lambda range: 0.01-1, 100 points, K=10) sure!!!
# error level=0.4: opt_lamnda= 0.26 (lambda range: 0.01-0.6, 60 points, K=20) sure!!!

error_level_1=0.4
Sigma_Ex=error_level_1*X_samp_cov
#%%
## bias-corrected SCAD-penalized method on (X*, Y)
time1=time.time()
opt_lambda_corrected=bf.cv_tuning_pointwise(Y=Y_data, X_star=X_star_data, Sigma_Ex=Sigma_Ex, 
                                        param_lower_bound=0.01,
                                        param_upper_bound=1, plot_points=100, case="corrected", 
                                        generate_figure=True, K=5)
time2=time.time()
print(f"time used: {(time2-time1)/60} mins.")
#%%
opt_lambda_corrected

#%%
start_time=time.time()
B_correct_scad_4, CON=bf.minimize_BCLSSCAD(tun_para=0.26, Y=Y_data, 
                                           X_star=X_star_data, 
                                           Sigma_Ex=Sigma_Ex, lower_bound=-100, 
                                           upper_bound=100,opt_method="trust-constr", 
                                           case="corrected", threshold=10**(-5))
print(CON)
end_time=time.time()
print(f"time used: {(end_time-start_time)/60} mins")
#%%
B_est=B_correct_scad_4
print(B_est.min(), B_est.max())
print(LA.norm(B_est, ord=1))
print(LA.norm(B_est, ord=np.inf))
#print(max(np.sum(abs(B_est), axis=0)))
#%%
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
sns.heatmap(B_correct_scad_4, annot=False,#fmt='f',
            # linewidth=.1, 
            # linecolor="black",
            ax=ax,cmap="seismic", #"PiYG", "seismic"
            vmin=-0.85, vmax=0.85, 
            annot_kws={"fontsize":12})

#%%
selection_threshold=10**(-5)
B_correct_scad_2[abs(B_correct_scad_2)<selection_threshold]=0

#%%
# calculating the Frobenus norm and plot 
corrected_est_list=[B_correct_scad_1, B_correct_scad_2, B_correct_scad_3, B_correct_scad_4]

F_norm_diff_list=[]
N_zeros=[]
N_nonzeros=[]

for corrected_est in corrected_est_list:
    F_norm_diff_list.append(LA.norm(B_naive_scad-corrected_est))
    N_zeros.append(np.sum(corrected_est==0))
    N_nonzeros.append(np.sum(corrected_est!=0))
print(f"F_norm_diff_list:{F_norm_diff_list}")
print(f"N_zeros: {N_zeros}")
print(f"N_nonzeros:{N_nonzeros}")

#%%
print(np.sum(B_naive_scad==0))
print(np.sum(B_naive_scad!=0))
#%%
fig=plt.subplots(figsize=(8, 5), dpi=200)
x=[0.1, 0.2, 0.3, 0.4]    
#plt.plot(x, F_norm_diff_list, marker="o", markersize=12,linestyle="-.", linewidth=3,label="Scenario 1: proportional")

plt.vlines(x, ymin=[0.95, 0.95, 0.95, 0.95], ymax=F_norm_diff_list, colors='blue', ls='--', lw=2)
plt.axhline(y = 0.95, color = 'grey')
plt.plot(x, F_norm_diff_list ,marker="o",markersize=10, linestyle=" "#, linewidth=3
        )
#plt.legend(loc='best', fontsize=20)
#plt.xlabel("$\xi$")
#plt.ylabel("$\|\hat{B}_{x^*c}-\hat{B}_{x^*}\|$")
plt.show()

############################ STOP ############################
#%%
Y_samp_cov=Y_data@Y_data.T/N_Obs
sns.heatmap(Y_samp_cov)
print(np.min(Y_samp_cov.min()),np.max(Y_samp_cov.max()))
################################################
#Simulation 1 for the naive and corrected estimates

#%%
#calculate the empirical covariance of X*
empi_cov_X=X_center@X_center.T/N_Obs
print(empi_cov_X)
#%%
empi_cov_X.shape
#%%
#Naive estimate
B_lse=(Y_center@X_center.T)/N_Obs@LA.inv(empi_cov_X)
print(f"B naive :({B_lse.min(), B_lse.max()})")

fig, ax =plt.subplots(3,1, figsize=(20, 15))
sns.heatmap(B_lse, vmin = -0.9, vmax = 1.3, center=0,   
            ax=ax[0])

#case 1. the cov of error is related to the cov of true x
error_level_index1=0.05
#error_level_index2=0.2

Sigma_Ex=error_level_index1*empi_cov_X
Sigma_trueX=(1-error_level_index1)*empi_cov_X

K1=empi_cov_X@LA.inv(Sigma_trueX)

B_correct1=B_lse@K1
print(f"B_correct error 0.05:({B_correct1.min(),B_correct1.max()})")

sns.heatmap(B_correct1, vmin = -0.9, vmax = 1.3, center=0,  ax=ax[1])

# Sigma_Ex=error_level_index2*empi_cov_X
# Sigma_trueX=(1-error_level_index2)*empi_cov_X

# K2=empi_cov_X@LA.inv(Sigma_trueX)

# B_correct2=B_lse@K2
# print(f"B_correct error 0.2:({B_correct2.min(),B_correct2.max()})")

# sns.heatmap(B_correct2, vmin = -0.9, vmax = 1.3, center=0,   ax=ax[2])

#case 2. the cov of the measurement error is not related 
# to the cov of x

cov_error_vec=empi_cov_X.diagonal()*error_level_index1
Sigma_Ex=np.eye(q)*np.diag(cov_error_vec)
Sigma_trueX=empi_cov_X-Sigma_Ex

K3=empi_cov_X@LA.inv(Sigma_trueX)

B_correct3=B_lse@K3
print(f"B_correct identity error 0.05:({B_correct3.min(), B_correct3.max()})")

sns.heatmap(B_correct3, vmin = -0.9, vmax = 1.3,center=0,   ax=ax[2])
plt.show()

#%%
print(f"L2 norm of diff (full cov 0.05):{LA.norm(B_correct1-B_lse)}")
print(f"L2 norm of diff (full cov 0.2): {LA.norm(B_correct2-B_lse)}")
print(f"L2 norm of diff (diag 0.05):{LA.norm(B_correct3-B_lse)}")

#%%
estimate_data=pd.DataFrame(np.concatenate((B_lse.reshape(-1,1), 
                            B_correct1.reshape(-1,1),
                            B_correct2.reshape(-1,1),
                            B_correct3.reshape(-1,1)),axis=1), columns=["Naive", "Scenario 1 (0.05)",
                                                                "Scenario 1 (0.2)", 
                                                                "Scenario 2 (0.05)"])
import plotly.graph_objects as go

layout = go.Layout(
    autosize=False,
    width=1000,
    height=600,)

fig = go.Figure( layout=layout)

days = ["Naive", "Scenario 1 (0.05)","Scenario 1 (0.2)", "Scenario 2 (0.05)"]


for day in days:
    fig.add_trace(go.Violin(x=estimate_data[day],
                            name=day,
                            box_visible=True,
                            meanline_visible=True
                            #,points="all"
                            ))

fig.show()
#%%
#plot curves for scenarios 1 and 2 with xi between 0.001 and 0.05
empi_cov_X=X_center@X_center.T/N_Obs
#Naive estimate
B_lse=(Y_center@X_center.T)/N_Obs@LA.inv(empi_cov_X)
sc1_norm_list=[]
sc2_norm_list=[]
for xi in np.linspace(start=0.001, stop=0.05, num=20):
    #Scenario 1. the cov of error is proportional to the cov of true x
    Sigma_Ex=xi*empi_cov_X
    Sigma_trueX=(1-xi)*empi_cov_X

    K1=empi_cov_X@LA.inv(Sigma_trueX)

    B_correct1=B_lse@K1
    sc1_norm_list.append(LA.norm(B_correct1-B_lse))
    
    #Scenario 2. Diagonal measurement error cov structure
    cov_error_vec=empi_cov_X.diagonal()*xi
    Sigma_Ex=np.eye(q)*np.diag(cov_error_vec)
    Sigma_trueX=empi_cov_X-Sigma_Ex

    K3=empi_cov_X@LA.inv(Sigma_trueX)

    B_correct3=B_lse@K3
    sc2_norm_list.append(LA.norm(B_correct3-B_lse))

fig=plt.subplots(figsize=(10, 8))
x=np.linspace(start=0.001, stop=0.05, num=20)    
plt.plot(x, sc1_norm_list, marker="o", markersize=12,linestyle="-.", linewidth=3,label="Scenario 1: proportional")
plt.plot(x, sc2_norm_list,marker="*",markersize=12, linestyle=":", linewidth=3,label="Scenario 2: diagonal")
plt.legend(loc='best', fontsize=20)
#plt.xlabel("$\xi$")
#plt.ylabel("$\|\hat{B}_{x^*c}-\hat{B}_{x^*}\|$")
plt.show()

