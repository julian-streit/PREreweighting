# Load modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import MDAnalysis as md
import math
import random
import pandas as pd
from kneed import DataGenerator,KneeLocator
cmap = plt.cm.get_cmap('Spectral')
from scipy.stats import pearsonr

# ensemble used here
topology_path = '../'
u = md.Universe(topology_path+'nc_modelrc_EM_1.gro')

# NMR resonances
residues_fln5 = u.select_atoms('resid 10:114 and name H and not resname PRO').resids + 636 # FLN5 residues NMR
print('Maximum possible NMR-observable residues:',len(residues_fln5))
print('NMR residues: \n',residues_fln5)

# Load experimental data
exp_data_path = '../exp_data_rnc/'

dataC657 = pd.read_csv(exp_data_path+'31_A3A3_C657_V747_data.csv')
dataC671 = pd.read_csv(exp_data_path+'31_A3A3_C671_V747_data.csv')
dataC699 = pd.read_csv(exp_data_path+'31_A3A3_C699_V747_data.csv')
dataC706 = pd.read_csv(exp_data_path+'31_A3A3_C706_V747_data.csv')
dataC720 = pd.read_csv(exp_data_path+'31_A3A3_C720_V747_data.csv')
dataC734 = pd.read_csv(exp_data_path+'31_A3A3_C734_V747_data.csv')
dataC740 = pd.read_csv(exp_data_path+'31_A3A3_C740_V747_data.csv')
dataC744 = pd.read_csv(exp_data_path+'31_A3A3_C744_V747_data.csv')
dataC53 = pd.read_csv(exp_data_path+'31_A3A3_V747_L24_N53C_data.csv')
dataC90 = pd.read_csv(exp_data_path+'31_A3A3_V747_L23_G90C_data.csv')

# theta values used for reweighting (L-curve)
thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

# load effective tauC and tauT values for each theta
tau_C = {}
tau_T = {}
# internal correlation time
tau_I = 0.5e-9

tauc_path = '../all_data/results/eff_tauc_data/'
# use final tauC after 20 iterations
nit = 20

for theta in thetas:
    nit = 20


    if theta == 'prior':
        nit = 0

    # tauC data l-curve elbow

    if theta == 'prior':

        tau_C_intra = np.load(tauc_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(thetas[1],nit),allow_pickle=True).item()
        tau_C_inter= np.load(tauc_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(thetas[1],nit),allow_pickle=True).item()

    else:
        tau_C_intra = np.load(tauc_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()
        tau_C_inter= np.load(tauc_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()



    # tauC data
    tau_C_theta = {}

    for key,values in zip(tau_C_intra.keys(),tau_C_intra.values()):
        tau_C_theta[key] = values

    ribo_labels = ['L23_C90','L24_C53']
    ribo_keys = ['C3064', 'C3126']

    for l,k in zip(ribo_labels,ribo_keys):
        tau_C_theta[k] = tau_C_inter[l]

    # calculate total correlation time from tau_C
    spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']
    tau_T_theta = {}
    for spinlabel in spinlabels:
        tau_C_theta[spinlabel] = tau_C_theta[spinlabel]*1e-9
        tau_T_theta[spinlabel] = 1/((1/tau_C_theta[spinlabel])+(1/tau_I))


    tau_C[theta] = tau_C_theta
    tau_T[theta] = tau_T_theta


# Load weights from reweighting


thetas = [1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
nit = 20

weights_path = '../all_data/results/wopt_data/'

# Reference weights for models [UNIFORM]
N_models = np.load(weights_path+'res_{}_{}.npy'.format(thetas[0],nit),allow_pickle=True).item()['wopt'].shape[0]
w0 = np.ones(N_models)/N_models


weights_array = [w0]

for theta in thetas:

    nit = 20

    w_current = np.load(weights_path+'res_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()['wopt']

    weights_array.append(w_current)

##############################################
## Collating Experimental & Simulation Data ##
##############################################

# gaussian error restraints

gaussian = []
gaussian_errors = []
gaussian_residues = []

upper = []
upper_errors = []
upper_residues = []

lower = []
lower_errors = []
lower_residues = []


# defining constants
K = 1.23e-44 # m^6 s^-2
# dataset-specific magnetic field strength --> larmor proton frequency

larmor_dict = {'C657':800.284e6*2*math.pi,
          'C671':800.354e6*2*math.pi,
          'C699':800.284e6*2*math.pi,
          'C706':800.354e6*2*math.pi,
          'C720':800.354e6*2*math.pi,
          'C734':800.354e6*2*math.pi,
          'C740':800.354e6*2*math.pi,
          'C744':800.284e6*2*math.pi,
          'C3064':800.354e6*2*math.pi,
          'C3126':800.354e6*2*math.pi}


dfs = [dataC657,dataC671,dataC699,dataC706,dataC720,dataC734,dataC740,dataC744,dataC90,dataC53]

# path for rm6 and S2 data
PRE_path = '../'


# C3064 = L23 C90, C3126 = L24 C53
spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']

sim_gaussian_all = []
sim_upper_all = []
sim_lower_all = []


thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

print('Thetas:')
for theta in thetas:

    print('Theta = ',theta)

    widx = thetas.index(theta)

    for i,spinlabel in zip(range(len(dfs)),spinlabels):

        # setting dataframe
        df = dfs[i]

        # setting the larmor frequency based on the dataset specific magnetic field strength
        larmor_H = larmor_dict[spinlabel]

        # setting labelling site specific simulation data (distances in Angstrom)

        rm6 = np.load(PRE_path + 'rm6_data_final.npy',allow_pickle=True).item()[spinlabel]
        S2 = np.load(PRE_path +'S2_data_final.npy',allow_pickle=True).item()[spinlabel]

        sim_data = (S2*(K*rm6)*((4*tau_C[theta][spinlabel]+((3*tau_C[theta][spinlabel])/(1+(larmor_H**2)*(tau_C[theta][spinlabel]**2)))))
                        + (1-S2)*(K*rm6)*((4*tau_T[theta][spinlabel]+((3*tau_T[theta][spinlabel])/(1+(larmor_H**2)*(tau_T[theta][spinlabel]**2))))))

        # restraint string prefix
        prefix = str('c{}_'.format(spinlabel[1:]))

        # initialising residue list for this iteration
        gau_res = []
        up_res = []
        low_res = []

        for j in range(len(df[df.columns[-6]])):

            # only take restraints for the FLN5 region (i.e. exclude 645, has very high error)
            resid_number = df['Residue'][j]
            if resid_number not in residues_fln5:
                continue

            if df[df.columns[-6]][j] > 0:

                gaussian.append(df[df.columns[-6]][j])
                gaussian_errors.append(df[df.columns[-5]][j])
                gaussian_residues.append(prefix+str(df['Residue'][j]))
                gau_res.append(df['Residue'][j])

            elif df[df.columns[-4]][j] > 0:

                gaussian.append(df[df.columns[-4]][j])
                gaussian_errors.append(df[df.columns[-3]][j])
                gaussian_residues.append(prefix+str(df['Residue'][j]))
                gau_res.append(df['Residue'][j])

            elif df[df.columns[-2]][j] > 0:

                if df[df.columns[-2]][j] < 15.0: # upper bound restraint

                    upper.append(df[df.columns[-2]][j])
                    upper_errors.append(df[df.columns[-2]][j]*(df['Combined_error (%)'][j]/100))
                    upper_residues.append(prefix+str(df['Residue'][j]))
                    up_res.append(df['Residue'][j])


                else: # lower bound restraint

                    lower.append(df[df.columns[-2]][j])
                    lower_errors.append(df[df.columns[-2]][j]*(df['Combined_error (%)'][j])/100)
                    lower_residues.append(prefix+str(df['Residue'][j]))
                    low_res.append(df['Residue'][j])

            elif df[df.columns[-1]][j] > 0:

                if df[df.columns[-1]][j] < 15.0: # upper bound restraint

                    upper.append(df[df.columns[-1]][j])
                    upper_errors.append(df[df.columns[-1]][j]*(df['Combined_error (%)'][j]/100))
                    upper_residues.append(prefix+str(df['Residue'][j]))
                    up_res.append(df['Residue'][j])

                else: # lower bound restraint

                    lower.append(df[df.columns[-1]][j])
                    lower_errors.append(df[df.columns[-1]][j]*(df['Combined_error (%)'][j]/100))
                    lower_residues.append(prefix+str(df['Residue'][j]))
                    low_res.append(df['Residue'][j])

            else:

                print("Warning: No datapoint found!")




        # append simulation data

        gau_idxs = []
        up_idxs = []
        low_idxs = []

        for res in gau_res:
            gau_idxs.append(np.where(residues_fln5==res)[0][0])
        for res in up_res:
            up_idxs.append(np.where(residues_fln5==res)[0][0])
        for res in low_res:
            low_idxs.append(np.where(residues_fln5==res)[0][0])


        # if array empty use append function
        if i == 0:
            if len(gau_idxs)>0:
                sim_gaussian = sim_data[:,np.array(gau_idxs)]
            else:
                sim_gaussian = np.array([])
            if len(up_idxs)>0:
                sim_upper = sim_data[:,np.array(up_idxs)]
            else:
                sim_upper = np.array([])
            if len(low_idxs)>0:
                sim_lower = sim_data[:,np.array(low_idxs)]
            else:
                sim_lower = np.array([])

        # if array non-empty use concatenate function
        else:
            if len(gau_idxs)>0:
                sim_gaussian = np.concatenate((sim_gaussian,sim_data[:,np.array(gau_idxs)]),axis=1)
            else:
                continue
            if len(up_idxs)>0:
                sim_upper = np.concatenate((sim_upper,sim_data[:,np.array(up_idxs)]),axis=1)
            else:
                continue
            if len(low_idxs)>0:
                sim_lower = np.concatenate((sim_lower,sim_data[:,np.array(low_idxs)]),axis=1)
            else:
                continue


    # calculating the weighted ensemble average
    wopt = weights_array[widx]
    sim_gaussian_avg = np.average(sim_gaussian,weights=wopt,axis=0)
    sim_upper_avg = np.average(sim_upper,weights=wopt,axis=0)
    #sim_lower_avg = np.average(sim_lower,weights=wopt,axis=0)

    sim_gaussian_all.append(sim_gaussian_avg)
    sim_upper_all.append(sim_upper_avg)
    #sim_lower_all.append(sim_lower_avg)






gaussian = np.array(gaussian)
gaussian_errors = np.array(gaussian_errors)
gaussian_residues = np.array(gaussian_residues)

# set error for upper lower bound restraints based on average relativ uncertainty of these restraint classes

upper = np.array(upper)
upper_residues = np.array(upper_residues)
upper_errors = np.array(upper_errors)

#lower = np.array(lower)
#lower_residues = np.array(lower_residues)
#lower_errors = np.array(lower_errors)


# extract first N gaussian
gaussian = gaussian[:sim_gaussian_all[0].shape[0]]
upper = upper[:sim_upper_all[0].shape[0]]
#lower = lower[:sim_lower_all[0].shape[0]]

gaussian_errors = gaussian_errors[:sim_gaussian_all[0].shape[0]]
upper_errors = upper_errors[:sim_upper_all[0].shape[0]]
#lower_errors = lower_errors[:sim_lower_all[0].shape[0]]

gaussian_residues = gaussian_residues[:sim_gaussian_all[0].shape[0]]
upper_residues = upper_residues[:sim_upper_all[0].shape[0]]
#lower_residues = lower_residues[:sim_lower_all[0].shape[0]]


#################################
# Calculate simulated PRE rates #
#################################

# Step 1: Obtain array of R2H and R2MQ in the same order as gaussian, upper, lower

all_restraint_ids = np.concatenate((gaussian_residues,upper_residues,lower_residues),axis=0)

exp_R2H = []
exp_R2MQ = []
exp_ratio = []
exp_ratio_error = []

# column names of Iox/Ired and corresponding list of spinlabels
col_names = ['Adjusted_ratio_paramagnetic:diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic','Ratio paramagnetic/diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic',
'Adjusted_ratio_paramagnetic:diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic','Ratio paramagnetic/diamagnetic','Ratio paramagnetic/diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic',
'Adjusted_ratio_paramagnetic:diamagnetic']

for restraint in all_restraint_ids:

    labelling_site = restraint[1:-4]
    nucleus = restraint[-3:]

    # deterine dataset
    df_idx = spinlabels.index(str('C')+str(labelling_site))

    df_current = dfs[df_idx]

    # extract linewidths
    exp_R2H.append(float(df_current[df_current['Residue']==int(nucleus)]['R2_H_diamagnetic']))
    exp_R2MQ.append(float(df_current[df_current['Residue']==int(nucleus)]['R2_MQ_diamagnetic']))

    # column name of Iox/Ired
    col_name = col_names[df_idx]

    # extract experimental intensity ratios
    # values above 1 or below 0 are set to 1 and 0, respectively
    iox_ired = float(df_current[df_current['Residue']==int(nucleus)][col_name])
    if iox_ired>1.0:
        exp_ratio.append(float(1.0))
    elif iox_ired<0.0:
        exp_ratio.append(float(0.0))
    else:
        exp_ratio.append(iox_ired)

    # experimental errors
    exp_ratio_error.append(float(df_current[df_current['Residue']==int(nucleus)]['Combined_error']))

exp_R2H = np.array(exp_R2H)
exp_R2MQ = np.array(exp_R2MQ)
exp_ratio = np.array(exp_ratio)
exp_ratio_error = np.array(exp_ratio_error)



# Calculate PRE Ratios (list of PRE rates for each theta value)
simratios = []
# delay during HMQC experiment
DELTA = 5.4348e-3 # delay time in s


for theta in thetas:

    idx = thetas.index(theta)

    all_sim_rates = np.concatenate((sim_gaussian_all[idx],sim_upper_all[idx]),axis=0)

    ratios = (exp_R2H*np.exp(-2*DELTA*all_sim_rates)/(exp_R2H+all_sim_rates))*(exp_R2MQ/(exp_R2MQ+all_sim_rates))
    simratios.append(ratios)




#####################################
# Chi-Square functions for analysis #
#####################################

def chiSqrTerm(sim_m, exp_data_v, exp_error_v):
    v = sim_m - exp_data_v
    v *= (1/exp_error_v)
    chiSqr = np.sum(v*v)

    return chiSqr

# for upper bound restraints
def chiSqrTermU(sim_u, exp_data_u, exp_error_u):
    bool_vector = (sim_u>exp_data_u)*1 # creates a vector of 0 and 1. 1 = restraint not satisfied
    v = sim_u - exp_data_u
    v *= bool_vector
    v *= (1/exp_error_u)
    chiSqr = np.sum(v*v)

    return chiSqr

# for lower bound restraints
def chiSqrTermL(sim_l, exp_data_l, exp_error_l):
    bool_vector = (sim_l<exp_data_l)*1 # creates a vector of 0 and 1. 1 = restraint not satisfied
    v = sim_l - exp_data_l
    v *= bool_vector
    v *= (1/exp_error_l)
    chiSqr = np.sum(v*v)

    return chiSqr


# Calculate entropy - Kullback-Leibler divergence

def get_entropy(w0, weights):
    s = - np.sum(weights * np.log(weights / w0))
    return s



###################################################
# Calculate entropy - Kullback-Leibler divergence #
###################################################

# Array with entropy
S_array = [get_entropy(w0,i) for i in weights_array]

thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

total_restraints = gaussian.shape[0]+upper.shape[0]#+lower.shape[0]


#####################################
# Calculate chi2 during reweighting #
#####################################
# PRE Rates Chi2
tot_chi2_rates = []

for j in range(len(thetas)):

    m=sim_gaussian_all[j]
    u=sim_upper_all[j]
    #l=sim_lower_all[j]

    # Array with chisqr
    chisqrt = chiSqrTerm(m,gaussian,gaussian_errors)

    # Array with chisqr
    chisqrtU = chiSqrTermU(u,upper,upper_errors)

    # Array with chisqr
    #chisqrtL = chiSqrTermL(l,lower,lower_errors)

    total = chisqrt+chisqrtU#+chisqrtL

    tot_chi2_rates.append(total)


# PRE Ratios Chi2
tot_chi2_ratios = []

for j in range(len(thetas)):

    m=simratios[j]

    # Array with chisqr
    chisqrt = chiSqrTerm(m,exp_ratio,exp_ratio_error)

    tot_chi2_ratios.append(chisqrt)



#####################################################################
# Save residue specific chi2 for analysis: Prior ensemble PRE Rates #
#####################################################################

save_path = './results/'

chi2_resids = []

chi2_resids_vals = []

for item in all_restraint_ids:
    chi2_resids.append(item)


# residue specific chi2
for i in list(range(gaussian.shape[0])):

    restraint = gaussian[i]
    error = gaussian_errors[i]
    sim = sim_gaussian_all[0][i]
    v = ((sim - restraint)/error)**2
    chi2_resids_vals.append(v)

for i in list(range(upper.shape[0])):

    restraint = upper[i]
    error = upper_errors[i]
    sim = sim_upper_all[0][i]
    if sim > restraint:
        v = ((sim - restraint)/error)**2
        chi2_resids_vals.append(v)
    else:
        chi2_resids_vals.append(float(0))

#for i in list(range(lower.shape[0])):

    #restraint = lower[i]
    #error = lower_errors[i]
    #sim = sim_lower_all[0][i]
    #if sim < restraint:
        #v = ((sim - restraint)/error)**2
        #chi2_resids_vals.append(v)
    #else:
        #chi2_resids_vals.append(float(0))


plot_x = {}
plot_y = {}

for i in spinlabels:
    plot_x[i] = []
    plot_y[i] = []


for s in spinlabels:
    site = str('c')+s[1:]
    for resi in residues_fln5:
        plot_x[s].append(resi)

        restraint_identity = site+str('_')+str(resi)

        if restraint_identity in chi2_resids:
            j = chi2_resids.index(restraint_identity)
            plot_y[s].append(chi2_resids_vals[j])
        else:
            plot_y[s].append(float(0))

plt.figure(figsize = (8,6))
for s in spinlabels:
    print(s)
    idx = spinlabels.index(s)
    plt.subplot(5,2,idx+1)
    plt.title(s)
    plt.bar(plot_x[s],plot_y[s])
    plt.xlabel('Residue')
    plt.ylabel('Residue Chi2')

plt.subplots_adjust(wspace = 0.4,hspace = 0.3)
plt.tight_layout()
plt.savefig(save_path+'resi_chi2_rates_prior.png')


# save csv and numpy files for residue-specific analysis
df = pd.DataFrame()
df['Residue'] = plot_x['C657']
for i in spinlabels:
    if i == 'C3064':
        df['L23 G90C'] = plot_y[i]
    elif i == 'C3126':
        df['L24 N53C'] = plot_y[i]
    else:
        df[i] = plot_y[i]

df.to_csv(save_path+'residue_chi2_rates_prior.csv')

np.save(save_path+'chi2_resids_rates_prior.npy',chi2_resids)
np.save(save_path+'chi2_resids_vals_rates_prior.npy',chi2_resids_vals)



#########################################################################
# Save residue specific chi2 for analysis: Posterior ensemble PRE Rates #
#########################################################################

save_path = './results/'

# from previous analysis of this reweighting run
elbow_idx = 11

chi2_resids = []

chi2_resids_vals = []

for item in all_restraint_ids:
    chi2_resids.append(item)


# residue specific chi2
for i in list(range(gaussian.shape[0])):

    restraint = gaussian[i]
    error = gaussian_errors[i]
    sim = sim_gaussian_all[elbow_idx][i]
    v = ((sim - restraint)/error)**2
    chi2_resids_vals.append(v)

for i in list(range(upper.shape[0])):

    restraint = upper[i]
    error = upper_errors[i]
    sim = sim_upper_all[elbow_idx][i]
    if sim > restraint:
        v = ((sim - restraint)/error)**2
        chi2_resids_vals.append(v)
    else:
        chi2_resids_vals.append(float(0))

#for i in list(range(lower.shape[0])):

    #restraint = lower[i]
    #error = lower_errors[i]
    #sim = sim_lower_all[elbow_idx][i]
    #if sim < restraint:
        #v = ((sim - restraint)/error)**2
        #chi2_resids_vals.append(v)
    #else:
        #chi2_resids_vals.append(float(0))


plot_x = {}
plot_y = {}

for i in spinlabels:
    plot_x[i] = []
    plot_y[i] = []


for s in spinlabels:
    site = str('c')+s[1:]
    for resi in residues_fln5:
        plot_x[s].append(resi)

        restraint_identity = site+str('_')+str(resi)

        if restraint_identity in chi2_resids:
            j = chi2_resids.index(restraint_identity)
            plot_y[s].append(chi2_resids_vals[j])
        else:
            plot_y[s].append(float(0))

plt.figure(figsize = (8,6))
for s in spinlabels:
    print(s)
    idx = spinlabels.index(s)
    plt.subplot(5,2,idx+1)
    plt.title(s)
    plt.bar(plot_x[s],plot_y[s])
    plt.xlabel('Residue')
    plt.ylabel('Residue Chi2')

plt.subplots_adjust(wspace = 0.4,hspace = 0.3)
plt.tight_layout()
plt.savefig(save_path+'resi_chi2_rates_posterior.png')


# save csv and numpy files for residue-specific analysis
df = pd.DataFrame()
df['Residue'] = plot_x['C657']
for i in spinlabels:
    if i == 'C3064':
        df['L23 G90C'] = plot_y[i]
    elif i == 'C3126':
        df['L24 N53C'] = plot_y[i]
    else:
        df[i] = plot_y[i]
df.to_csv(save_path+'residue_chi2_rates_posterior.csv')

np.save(save_path+'chi2_resids_rates_posterior.npy',chi2_resids)
np.save(save_path+'chi2_resids_vals_rates_postrior.npy',chi2_resids_vals)





###########################################################################
# Save residue specific chi2 for analysis: Prior ensemble PRE Intensities #
###########################################################################

save_path = './results/'

chi2_resids = []

chi2_resids_vals = []

for item in all_restraint_ids:
    chi2_resids.append(item)


# residue specific chi2
for i in list(range(exp_ratio.shape[0])):

    restraint = exp_ratio[i]
    error = exp_ratio_error[i]
    sim = simratios[0][i]
    v = ((sim - restraint)/error)**2
    chi2_resids_vals.append(v)


plot_x = {}
plot_y = {}

for i in spinlabels:
    plot_x[i] = []
    plot_y[i] = []


for s in spinlabels:
    site = str('c')+s[1:]
    for resi in residues_fln5:
        plot_x[s].append(resi)

        restraint_identity = site+str('_')+str(resi)

        if restraint_identity in chi2_resids:
            j = chi2_resids.index(restraint_identity)
            plot_y[s].append(chi2_resids_vals[j])
        else:
            plot_y[s].append(float(0))

plt.figure(figsize = (8,6))
for s in spinlabels:
    print(s)
    idx = spinlabels.index(s)
    plt.subplot(5,2,idx+1)
    plt.title(s)
    plt.bar(plot_x[s],plot_y[s])
    plt.xlabel('Residue')
    plt.ylabel('Residue Chi2')

plt.subplots_adjust(wspace = 0.4,hspace = 0.3)
plt.tight_layout()
plt.savefig(save_path+'resi_chi2_ratios_prior.png')


# save csv and numpy files for residue-specific analysis
df = pd.DataFrame()
df['Residue'] = plot_x['C657']
for i in spinlabels:
    if i == 'C3064':
        df['L23 G90C'] = plot_y[i]
    elif i == 'C3126':
        df['L24 N53C'] = plot_y[i]
    else:
        df[i] = plot_y[i]
df.to_csv(save_path+'residue_chi2_ratios_prior.csv')

# save dataset-specific reduced chi2
df2 = pd.DataFrame()

for i in spinlabels:
    if i == 'C3064':
        t = np.array(plot_y[i])
        df2['L23 G90C'] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]
    elif i == 'C3126':
        t = np.array(plot_y[i])
        df2['L24 N53C'] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]
    else:
        t = np.array(plot_y[i])
        df2[i] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]

df2.to_csv(save_path+'dataset_chi2_ratios_prior.csv')

# save numpy files
np.save(save_path+'chi2_resids_ratios_prior.npy',chi2_resids)
np.save(save_path+'chi2_resids_vals_ratios_prior.npy',chi2_resids_vals)



###############################################################################
# Save residue specific chi2 for analysis: Posterior ensemble PRE Intensities #
###############################################################################

save_path = './results/'

chi2_resids = []

chi2_resids_vals = []

for item in all_restraint_ids:
    chi2_resids.append(item)


# residue specific chi2
for i in list(range(exp_ratio.shape[0])):

    restraint = exp_ratio[i]
    error = exp_ratio_error[i]
    sim = simratios[elbow_idx][i]
    v = ((sim - restraint)/error)**2
    chi2_resids_vals.append(v)


plot_x = {}
plot_y = {}

for i in spinlabels:
    plot_x[i] = []
    plot_y[i] = []



for s in spinlabels:
    site = str('c')+s[1:]
    for resi in residues_fln5:
        plot_x[s].append(resi)

        restraint_identity = site+str('_')+str(resi)

        if restraint_identity in chi2_resids:
            j = chi2_resids.index(restraint_identity)
            plot_y[s].append(chi2_resids_vals[j])
        else:
            plot_y[s].append(float(0))

plt.figure(figsize = (8,6))
for s in spinlabels:
    print(s)
    idx = spinlabels.index(s)
    plt.subplot(5,2,idx+1)
    plt.title(s)
    plt.bar(plot_x[s],plot_y[s])
    plt.xlabel('Residue')
    plt.ylabel('Residue Chi2')

plt.subplots_adjust(wspace = 0.4,hspace = 0.3)
plt.tight_layout()
plt.savefig(save_path+'resi_chi2_ratios_posterior.png')


# save csv and numpy files for residue-specific analysis
df = pd.DataFrame()
df['Residue'] = plot_x['C657']
for i in spinlabels:
    if i == 'C3064':
        df['L23 G90C'] = plot_y[i]
    elif i == 'C3126':
        df['L24 N53C'] = plot_y[i]
    else:
        df[i] = plot_y[i]
df.to_csv(save_path+'residue_chi2_ratios_posterior.csv')

# save dataset-specific reduced chi2
df2 = pd.DataFrame()
for i in spinlabels:
    if i == 'C3064':
        t = np.array(plot_y[i])
        df2['L23 G90C'] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]
    elif i == 'C3126':
        t = np.array(plot_y[i])
        df2['L24 N53C'] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]
    else:
        t = np.array(plot_y[i])
        df2[i] = [np.round(np.sum(t[t!=0.0])/t[t!=0.0].shape[0],2)]

df2.to_csv(save_path+'dataset_chi2_ratios_posterior.csv')

# save numpy files
np.save(save_path+'chi2_resids_ratios_posterior.npy',chi2_resids)
np.save(save_path+'chi2_resids_vals_ratios_posterior.npy',chi2_resids_vals)


###############
# Elbow index #
###############

# determined from previous analysis
elbow_idx = 11



#################
# L-curve plots #
#################

save_path = './results/'

# full L-curve intensity ratios
plt.figure(figsize = (4,3))
thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

for i in range(1,len(thetas)+1):
    plt.scatter(-S_array[i-1],tot_chi2_ratios[i-1]/total_restraints,color=cmap(i/len(thetas)),label=str(thetas[i-1]))

plt.scatter(-S_array[elbow_idx],tot_chi2_ratios[elbow_idx]/total_restraints,facecolors = 'none',edgecolors = 'black')

plt.xlabel("Entropy")
plt.ylabel("Reduced $\u03C7^2$")
plt.title('L-curve: Intensity Ratios')
plt.tight_layout()

plt.savefig(save_path+'lcurve_ratios_full.png')

# zoomed L-curve intensity ratios
plt.figure(figsize = (4,3))
thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
y0 = tot_chi2_ratios[0]/total_restraints

for i in range(1,len(thetas)+1):
    plt.scatter(-S_array[i-1],tot_chi2_ratios[i-1]/total_restraints,color=cmap(i/len(thetas)),label=str(thetas[i-1]))

plt.scatter(-S_array[elbow_idx],tot_chi2_ratios[elbow_idx]/total_restraints,facecolors = 'none',edgecolors = 'black')

plt.xlabel("Entropy")
plt.ylabel("Reduced $\u03C7^2$")
plt.title('L-curve: Intensity Ratios Zoomed')
plt.ylim(-0.2,y0*1.3)
plt.tight_layout()

plt.savefig(save_path+'lcurve_ratios_zoomed.png')


# full L-curve PRE rates
plt.figure(figsize = (4,3))
thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

for i in range(1,len(thetas)+1):
    plt.scatter(-S_array[i-1],tot_chi2_rates[i-1]/total_restraints,color=cmap(i/len(thetas)),label=str(thetas[i-1]))

plt.scatter(-S_array[elbow_idx],tot_chi2_rates[elbow_idx]/total_restraints,facecolors = 'none',edgecolors = 'black')
plt.xlabel("Entropy")
plt.ylabel("Reduced $\u03C7^2$")
plt.title('L-curve: All restraints PRE (Hz)')
plt.tight_layout()

plt.savefig(save_path+'lcurve_rates_full.png')


# zoomed L-curve PRE rates
plt.figure(figsize = (4,3))
thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
y0 = tot_chi2_rates[0]/total_restraints
for i in range(1,len(thetas)+1):
    plt.scatter(-S_array[i-1],tot_chi2_rates[i-1]/total_restraints,color=cmap(i/len(thetas)),label=str(thetas[i-1]))

plt.scatter(-S_array[elbow_idx],tot_chi2_rates[elbow_idx]/total_restraints,facecolors = 'none',edgecolors = 'black')
plt.xlabel("Entropy")
plt.ylabel("Reduced $\u03C7^2$")
plt.title('L-curve: All restraints PRE (Hz)')
plt.ylim(-30,y0*1.2)
plt.tight_layout()

plt.savefig(save_path+'lcurve_rates_zoomed.png')

#####################
# Save L-curve data #
#####################

lcurve_data = {}
lcurve_data['elbow_idx'] = elbow_idx
lcurve_data['entropy'] = -np.array(S_array)
lcurve_data['chi2 ratios'] = np.array(np.array(tot_chi2_ratios)/total_restraints)
lcurve_data['chi2 rates'] = np.array(np.array(tot_chi2_rates)/total_restraints)
np.save(save_path+'lcurve_data.npy',lcurve_data)


############################################
# Plot convergence of weights and eff_tauc #
############################################

# RMSD function

def rmsd(array1,array2):

    return np.sqrt(np.sum((array1-array2)**2)/array1.shape[0])


thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

elbow_theta = thetas[elbow_idx]


# load weights data

theta = elbow_theta
nit = 20

weights_path = '../all_data/results/wopt_data/'
weights_array = []

for i in range(1,nit+1):
    w_current = np.load(weights_path+'res_{}_{}.npy'.format(theta,i),allow_pickle=True).item()['wopt']

    weights_array.append(w_current)

# Calculate diff entropy

diff_entropy  = []

for i in range(1,nit):

    diff_entropy.append(-get_entropy(weights_array[i],weights_array[i-1]))


# plot
plt.figure()
plt.plot(list(range(2,nit+1)),diff_entropy)
plt.xlabel('Iteration')
plt.ylabel('Difference Entropy')
plt.xticks(list(range(2,nit+1)))
plt.title('Theta = {:.3e}'.format(theta))
plt.tight_layout()
plt.savefig(save_path+'weights_convergence_{}.png'.format(theta))




# load tauC data
theta = elbow_theta
nit = 20


tauc_path = '../all_data/results/eff_tauc_data/'

tauc_intra_array = []
tauc_inter_array = []

for i in range(0,nit+1):
    tauc_intra_array.append(np.load(tauc_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(theta,i),allow_pickle=True).item())
    tauc_inter_array.append(np.load(tauc_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(theta,i),allow_pickle=True).item())


tc_data_all = []

for i in range(0,nit+1):

    current_it = []

    for key1,key2 in zip(tauc_intra_array[0].keys(),tauc_inter_array[0].keys()):

        current_it += list(tauc_intra_array[i][key1])
        current_it += list(tauc_inter_array[i][key2])


    tc_data_all.append(current_it)


# Calculate tauC RMSD between subsequent iterations

tauc_rmsd  = []

for i in range(1,nit+1):

    tauc_rmsd.append(rmsd(np.array(tc_data_all[i]),np.array(tc_data_all[i-1])))

# plot
plt.figure()
plt.plot(list(range(1,nit+1)),tauc_rmsd)
plt.xlabel('Iteration')
plt.ylabel('tauC RMSD')
plt.xticks(list(range(1,nit+1)))
plt.title('Theta = {:.3e}'.format(theta))
plt.tight_layout()
plt.savefig(save_path+'tauc_convergence_{}.png'.format(theta))




###########################################################################
# Plot prior and posterior PRE intensity ratios against experimental data #
###########################################################################


# Reference weights for models [UNIFORM]
weights_path = '../all_data/results/wopt_data/'
N_models = np.load(weights_path+'res_{}_{}.npy'.format(thetas[-1],1),allow_pickle=True).item()['wopt'].shape[0]
w0 = np.ones(N_models)/N_models



# load weights data
thetas = [1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
nit = 20

weights_array = [w0]
for theta in thetas:

    nit = 20

    w_current = np.load(weights_path+'res_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()['wopt']

    weights_array.append(w_current)



print('Nmodels:',N_models)

# adjust data (only FLN5 residues 646-750)
dataC657 = dataC657[dataC657['Residue']!=645]
dataC671 = dataC671[dataC671['Residue']!=645]
dataC699 = dataC699[dataC699['Residue']!=645]
dataC706 = dataC706[dataC706['Residue']!=645]
dataC720 = dataC720[dataC720['Residue']!=645]
dataC734 = dataC734[dataC734['Residue']!=645]
dataC740 = dataC740[dataC740['Residue']!=645]
dataC744 = dataC744[dataC744['Residue']!=645]
dataC90 = dataC90[dataC90['Residue']!=645]
dataC53 = dataC53[dataC53['Residue']!=645]

dfs = [dataC657,dataC671,dataC699,dataC706,dataC720,dataC734,dataC740,dataC744,dataC90,dataC53]

# constants for PRE rate calculation from rm6 and S2 data
# defining constants
K = 1.23e-44 # m^6 s^-2
larmor_dict = {'C657':800.284e6*2*math.pi,
          'C671':800.354e6*2*math.pi,
          'C699':800.284e6*2*math.pi,
          'C706':800.354e6*2*math.pi,
          'C720':800.354e6*2*math.pi,
          'C734':800.354e6*2*math.pi,
          'C740':800.354e6*2*math.pi,
          'C744':800.284e6*2*math.pi,
          'C3064':800.354e6*2*math.pi,
          'C3126':800.354e6*2*math.pi}
tau_I = 0.5e-9
DELTA = 5.4348e-3 # delay time in s

elbow_idx = elbow_idx # including prior
thetas = [1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
elbow_theta = thetas[elbow_idx-1]

spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']


# path for rm6 and S2 data
PRE_path = '../'
rm6_all = np.load(PRE_path+'rm6_data_final.npy',allow_pickle=True).item()
S2_all = np.load(PRE_path+'S2_data_final.npy',allow_pickle=True).item()
w_opt = weights_array[elbow_idx]


R2H_C657 = dataC657['R2_H_diamagnetic']
R2H_C671 = dataC671['R2_H_diamagnetic']
R2H_C699 = dataC699['R2_H_diamagnetic']
R2H_C706 = dataC706['R2_H_diamagnetic']
R2H_C720 = dataC720['R2_H_diamagnetic']
R2H_C734 = dataC734['R2_H_diamagnetic']
R2H_C740 = dataC740['R2_H_diamagnetic']
R2H_C744 = dataC744['R2_H_diamagnetic']
R2H_C90 = dataC90['R2_H_diamagnetic']
R2H_C53 = dataC53['R2_H_diamagnetic']


R2MQ_C657 = dataC657['R2_MQ_diamagnetic']
R2MQ_C671 = dataC671['R2_MQ_diamagnetic']
R2MQ_C699 = dataC699['R2_MQ_diamagnetic']
R2MQ_C706 = dataC706['R2_MQ_diamagnetic']
R2MQ_C720 = dataC720['R2_MQ_diamagnetic']
R2MQ_C734 = dataC734['R2_MQ_diamagnetic']
R2MQ_C740 = dataC740['R2_MQ_diamagnetic']
R2MQ_C744 = dataC744['R2_MQ_diamagnetic']
R2MQ_C90 = dataC90['R2_MQ_diamagnetic']
R2MQ_C53 = dataC53['R2_MQ_diamagnetic']



idxs_657 = []
for res in dataC657['Residue']:
    idxs_657.append(np.where(residues_fln5==res)[0][0])

idxs_671 = []
for res in dataC671['Residue']:
    idxs_671.append(np.where(residues_fln5==res)[0][0])

idxs_699 = []
for res in dataC699['Residue']:
    idxs_699.append(np.where(residues_fln5==res)[0][0])

idxs_706 = []
for res in dataC706['Residue']:
    idxs_706.append(np.where(residues_fln5==res)[0][0])

idxs_720 = []
for res in dataC720['Residue']:
    idxs_720.append(np.where(residues_fln5==res)[0][0])

idxs_734 = []
for res in dataC734['Residue']:
    idxs_734.append(np.where(residues_fln5==res)[0][0])

idxs_740 = []
for res in dataC740['Residue']:
    idxs_740.append(np.where(residues_fln5==res)[0][0])

idxs_744 = []
for res in dataC744['Residue']:
    idxs_744.append(np.where(residues_fln5==res)[0][0])

idxs_90 = []
for res in dataC90['Residue']:
    idxs_90.append(np.where(residues_fln5==res)[0][0])

idxs_53 = []
for res in dataC53['Residue']:
    idxs_53.append(np.where(residues_fln5==res)[0][0])

# Calculating PRE rates and intensity ratios (ensemble average before reweighting



rates_init = {}
ratios_init = {}

R2H_list = [R2H_C657,R2H_C671,R2H_C699,R2H_C706,R2H_C720,R2H_C734,R2H_C740,R2H_C744,R2H_C90,R2H_C53]
R2MQ_list = [R2MQ_C657,R2MQ_C671,R2MQ_C699,R2MQ_C706,R2MQ_C720,R2MQ_C734,R2MQ_C740,R2MQ_C744,R2MQ_C90,R2MQ_C53]
idx_list = [idxs_657,idxs_671,idxs_699,idxs_706,idxs_720,idxs_734,idxs_740,idxs_744,idxs_90,idxs_53]

for [spinlabel,R2H,R2MQ,idxs] in zip(spinlabels,
                    R2H_list,R2MQ_list,idx_list):

    #setting the larmor frequency based on the dataset specific magnetic field strength
    larmor_H = larmor_dict[spinlabel]


    rm6 = rm6_all[spinlabel][:,np.array(idxs)]
    S2 = S2_all[spinlabel][:,np.array(idxs)]
    tC = tau_C['prior'][spinlabel][np.array(idxs)]
    tT = tau_T['prior'][spinlabel][np.array(idxs)]

    rates = (S2*(K*rm6)*((4*tC+((3*tC)/(1+(larmor_H**2)*(tC**2)))))
                    + (1-S2)*(K*rm6)*((4*tT+((3*tT)/(1+(larmor_H**2)*(tT**2))))))

    rate_avg = np.average(rates,weights = w0,axis = 0)
    rates_init[spinlabel] = rate_avg
    ratios_init[spinlabel] = (R2H*np.exp(-2*DELTA*rate_avg)/(R2H+rate_avg))*(R2MQ/(R2MQ+rate_avg))

# Calculating PRE rates and intensity ratios (ensemble average with optimised weights)


rates_opt = {}
ratios_opt = {}

wopt_theta = thetas

for [spinlabel,R2H,R2MQ,idxs] in zip(spinlabels,
                    R2H_list,R2MQ_list,idx_list):

    #setting the larmor frequency based on the dataset specific magnetic field strength
    larmor_H = larmor_dict[spinlabel]

    rm6 = rm6_all[spinlabel][:,np.array(idxs)]
    S2 = S2_all[spinlabel][:,np.array(idxs)]
    tC = tau_C[elbow_theta][spinlabel][np.array(idxs)]
    tT = tau_T[elbow_theta][spinlabel][np.array(idxs)]


    rates = (S2*(K*rm6)*((4*tC+((3*tC)/(1+(larmor_H**2)*(tC**2)))))
                    + (1-S2)*(K*rm6)*((4*tT+((3*tT)/(1+(larmor_H**2)*(tT**2))))))


    rate_avg = np.average(rates,weights = w_opt,axis = 0)
    rates_opt[spinlabel] = rate_avg
    ratios_opt[spinlabel] = (R2H*np.exp(-2*DELTA*rate_avg)/(R2H+rate_avg))*(R2MQ/(R2MQ+rate_avg))


fig = plt.figure(figsize = (12,16))

col_names = ['Adjusted_ratio_paramagnetic:diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic','Ratio paramagnetic/diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic',
'Adjusted_ratio_paramagnetic:diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic','Ratio paramagnetic/diamagnetic','Ratio paramagnetic/diamagnetic','Adjusted_ratio_paramagnetic:diamagnetic',
'Adjusted_ratio_paramagnetic:diamagnetic']

for i in range(len(idx_list)):

    plt.subplot(5,2,i+1)
    plt.title(spinlabels[i])

    df = dfs[i]
    idx = idx_list[i]
    spinlabel = spinlabels[i]
    col_name = col_names[i]

    plt.bar(np.array(df['Residue']),np.array(df[col_name]),color = '#0055ff',yerr = df['Combined_error'],
           edgecolor = '#0055ff',linewidth = 0.5,error_kw=dict(ecolor='black',elinewidth=1.5)
           ,label = 'Experimental Data')

    plt.plot(residues_fln5[idx],ratios_init[spinlabel],linewidth = 3,color = 'red',label = 'SBM: Starting Ensemble')
    plt.plot(residues_fln5[idx],ratios_opt[spinlabel],linewidth = 3,color = 'lightgreen',label = 'SBM: Optimised Ensemble')





    plt.xlabel('Residue')
    plt.ylabel('I$_{ox}$/I$_{red}$')

    plt.xlim(644,750)
    plt.ylim(-0.1,1.1)






fig.subplots_adjust(wspace = 0.7)
plt.tight_layout()
plt.savefig(save_path+'exp_vs_model.png')


# save data to plot
pre_profiles_plot_data = {}
pre_profiles_plot_data['idx_list']=idx_list
pre_profiles_plot_data['ratios_init']=ratios_init
pre_profiles_plot_data['ratios_opt']=ratios_opt
np.save(save_path+'pre_profiles_plot_data.npy',pre_profiles_plot_data)

# save optimised weights
np.save(save_path+'w_opt.npy',w_opt)


################################
# CORRELATION PLOT INTENSITIES #
################################

exp_ratios = []
exp_errors = []
prior_ratios = []
posterior_ratios = []


for i in range(len(idx_list)):


    df = dfs[i]
    df = df[df['Residue']!=645]
    idx = idx_list[i]
    spinlabel = spinlabels[i]
    col_name = col_names[i]

    for ele in np.array(df[col_name]):
        if ele > 1.0:
            exp_ratios.append(float(1.0))
        elif ele < 0.0:
            exp_ratios.append(float(0.0))
        else:
            exp_ratios.append(ele)
    for ele in np.array(df['Combined_error']):
        exp_errors.append(ele)

    for ele in ratios_init[spinlabel]:
        prior_ratios.append(ele)
    for ele in ratios_opt[spinlabel]:
        posterior_ratios.append(ele)

exp_ratios = np.array(exp_ratios)
exp_errors = np.array(exp_errors)
prior_ratios = np.array(prior_ratios)
posterior_ratios = np.array(posterior_ratios)

# Prior figure
plt.figure(figsize = (3,3))

xvals = np.arange(0,1,0.001)
yvals = xvals

corr_prior = pearsonr(exp_ratios,prior_ratios)[0]


plt.plot(xvals,yvals,color = 'black')
plt.errorbar(exp_ratios,prior_ratios,xerr = exp_errors,fmt = 'o',color = 'blue',alpha = 0.2)

plt.text(0.1,0.9,'$R^2$ = {:.3f}'.format(corr_prior))

plt.ylim(0,1)
plt.xlim(0,1)

plt.xticks(np.arange(0, 1.2, step=0.2))
plt.yticks(np.arange(0, 1.2, step=0.2))

plt.xlabel('Experiment')
plt.ylabel('Model')

plt.tight_layout()
plt.savefig(save_path+'prior_intensities_corr.png')



# Posterior figure
plt.figure(figsize = (3,3))

xvals = np.arange(0,1,0.001)
yvals = xvals

corr_posterior = pearsonr(exp_ratios,posterior_ratios)[0]


plt.plot(xvals,yvals,color = 'black')
plt.errorbar(exp_ratios,posterior_ratios,xerr = exp_errors,fmt = 'o',color = 'blue',alpha = 0.2)

plt.text(0.1,0.9,'$R^2$ = {:.3f}'.format(corr_posterior))

plt.ylim(0,1)
plt.xlim(0,1)

plt.xticks(np.arange(0, 1.2, step=0.2))
plt.yticks(np.arange(0, 1.2, step=0.2))

plt.xlabel('Experiment')
plt.ylabel('Model')

plt.tight_layout()
plt.savefig(save_path+'posterior_intensities_corr.png')


##############################
# RMSD STATISTIC INTENSITIES #
##############################

def rmsd(exp,calc):

    return np.sqrt(np.sum((exp-calc)**2)/exp.shape[0])

def rmsd_error(exp,errors,calc): # calculates RMSD only for restraints not within error, to the upper or lower bound

    deviations = []

    for i in range(exp.shape[0]):

        experiment = exp[i]
        error = errors[i]
        value = calc[i]

        if value < experiment:

            if value < experiment-error:

                deviations.append((experiment-error-value)**2)

        else:

            if value > experiment+error:

                deviations.append((value-experiment+error)**2)

    return np.sqrt(np.sum(deviations)/exp.shape[0])



exp_ratios = []
exp_errors = []
prior_ratios = []
posterior_ratios = []


for i in range(len(idx_list)):


    df = dfs[i]
    df = df[df['Residue']!=645]
    idx = idx_list[i]
    spinlabel = spinlabels[i]
    col_name = col_names[i]

    for ele in np.array(df[col_name]):
        if ele > 1.0:
            exp_ratios.append(float(1.0))
        elif ele < 0.0:
            exp_ratios.append(float(0.0))
        else:
            exp_ratios.append(ele)
    for ele in np.array(df['Combined_error']):
        exp_errors.append(ele)

    for ele in ratios_init[spinlabel]:
        prior_ratios.append(ele)
    for ele in ratios_opt[spinlabel]:
        posterior_ratios.append(ele)

exp_ratios = np.array(exp_ratios)
exp_errors = np.array(exp_errors)
prior_ratios = np.array(prior_ratios)
posterior_ratios = np.array(posterior_ratios)


prior_rmsd = np.round(rmsd(exp_ratios,prior_ratios),3)
posterior_rmsd = np.round(rmsd(exp_ratios,posterior_ratios),3)
prior_rmsd_errors = np.round(rmsd_error(exp_ratios,exp_errors,prior_ratios),3)
posterior_rmsd_errors = np.round(rmsd_error(exp_ratios,exp_errors,posterior_ratios),3)



############################################
# Plot prior and posterior eff tauC values #
############################################


thetas = ['prior',1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]

tau_C_penultimate = {}
tau_T_penultimate = {}

# constants for PRE rate calculation from rm6 and S2 data
# defining constants
K = 1.23e-44 # m^6 s^-2
larmor_dict = {'C657':800.284e6*2*math.pi,
          'C671':800.354e6*2*math.pi,
          'C699':800.284e6*2*math.pi,
          'C706':800.354e6*2*math.pi,
          'C720':800.354e6*2*math.pi,
          'C734':800.354e6*2*math.pi,
          'C740':800.354e6*2*math.pi,
          'C744':800.284e6*2*math.pi,
          'C3064':800.354e6*2*math.pi,
          'C3126':800.354e6*2*math.pi}
tau_I = 0.5e-9

tauc_path = '../all_data/results/eff_tauc_data/'



nit = 19

for theta in thetas:
    nit = 19



    if theta == 'prior':
        nit = 0

    # tauC data l-curve elbow

    if theta == 'prior':

        tau_C_intra = np.load(tauc_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(thetas[1],nit),allow_pickle=True).item()
        tau_C_inter= np.load(tauc_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(thetas[1],nit),allow_pickle=True).item()

    else:
        tau_C_intra = np.load(tauc_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()
        tau_C_inter= np.load(tauc_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(theta,nit),allow_pickle=True).item()



    # tauC data
    tau_C_theta = {}

    for key,values in zip(tau_C_intra.keys(),tau_C_intra.values()):
        tau_C_theta[key] = values

    ribo_labels = ['L23_C90','L24_C53']
    ribo_keys = ['C3064', 'C3126']

    for l,k in zip(ribo_labels,ribo_keys):
        tau_C_theta[k] = tau_C_inter[l]

    # calculate total correlation time from tau_C
    spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']
    tau_T_theta = {}
    for spinlabel in spinlabels:
        tau_C_theta[spinlabel] = tau_C_theta[spinlabel]*1e-9
        tau_T_theta[spinlabel] = 1/((1/tau_C_theta[spinlabel])+(1/tau_I))


    tau_C_penultimate[theta] = tau_C_theta
    tau_T_penultimate[theta] = tau_T_theta


# plot effective tauC final and the one before and prior

elbow_idx = elbow_idx # including prior
thetas = [1e12,1e7,5e6,1e6,7.5e5,5e5,2.5e5,1e5,7.5e4,5e4,2.5e4,1e4,7.5e3,
          5e3,2.5e3,1e3,5e2,4e2,3.5e2,3e2,2e2,1e2,20,0]
elbow_theta = thetas[elbow_idx-1]

cterm = 725
cterm_index = np.where(residues_fln5==725)[0][0]
spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']
fig = plt.figure(figsize = (12,16))



for i in range(len(idx_list)):

    plt.subplot(5,2,i+1)
    plt.title(spinlabels[i])

    plt.plot(residues_fln5[1:cterm_index],tau_C[elbow_theta][spinlabels[i]][:cterm_index-1]*1e9,label = 'Posterior')
    plt.plot(residues_fln5[1:cterm_index],tau_C['prior'][spinlabels[i]][:cterm_index-1]*1e9,label='Prior')
    plt.plot(residues_fln5[1:cterm_index],tau_C_penultimate[elbow_theta][spinlabels[i]][:cterm_index-1]*1e9,label='Penultimate')

    plt.xlabel('Residue')
    plt.ylabel('\u03C4$_{c,eff}$ (ns)')

    plt.legend()


fig.subplots_adjust(wspace = 0.7)
plt.tight_layout()

plt.savefig(save_path+'eff_tauc_plots_model.png')


############################################
# Save csv with prior and posterior values #
############################################

final_df = {}
final_df['elbow_idx']=[int(elbow_idx)]
final_df['entropy posterior'] = [np.round(-np.array(S_array)[elbow_idx],2)]
final_df['Neff posterior'] = [np.round(np.exp(S_array[elbow_idx]),3)]
final_df['Kish posterior'] = [np.round(np.log((np.sum(weights_array[elbow_idx])**2/np.sum(weights_array[elbow_idx]**2))/w0.shape[0]),3)]
final_df['chi2_ratios_prior'] = [np.round(np.array(np.array(tot_chi2_ratios)/total_restraints)[0],2)]
final_df['chi2_rates_prior'] = [np.around(np.array(np.array(tot_chi2_rates)/total_restraints)[0],2)]
final_df['chi2_ratios_posterior'] = [np.round(np.array(np.array(tot_chi2_ratios)/total_restraints)[elbow_idx],2)]
final_df['chi2_rates_posterior'] = [np.around(np.array(np.array(tot_chi2_rates)/total_restraints)[elbow_idx],2)]
final_df['R2_prior'] = [corr_prior]
final_df['R2_posterior'] = [corr_posterior]
final_df['RMSD_prior'] = [prior_rmsd]
final_df['RMSD_posterior'] = [posterior_rmsd]
final_df['RMSD_error_prior'] = [prior_rmsd_errors]
final_df['RMSD_error_posterior'] = [posterior_rmsd_errors]
final_df = pd.DataFrame(final_df)
final_df = final_df.T
final_df.to_csv(save_path+'results_final.csv')


# also print final results_final# Calculate Reweighting Statistics
# Calculate Reweighting Statistics
print('Total restraints:',total_restraints)
print('Red Chi2 Rates:',np.array(np.array(tot_chi2_rates)/total_restraints)[elbow_idx])
print('Red Chi2 Rates Prior:',np.array(np.array(tot_chi2_rates)/total_restraints)[0])
print('Red Chi2 Intensities:',np.array(np.array(tot_chi2_ratios)/total_restraints)[elbow_idx])
print('Red Chi2 Intensities Prior:',np.array(np.array(tot_chi2_ratios)/total_restraints)[0])
print('R2 prior',corr_prior)
print('R2 posterior:',corr_posterior)
print('RMSD prior:',prior_rmsd)
print('RMSD posterior:',posterior_rmsd)
print('RMSD prior (errors)',prior_rmsd_errors)
print('RMSD posterior (errors):',posterior_rmsd_errors)
print('Opt. Entropy:',-S_array[elbow_idx])
print('Neff opt:',np.exp(S_array[elbow_idx]))
print('Kish score:', np.log((np.sum(weights_array[elbow_idx])**2/np.sum(weights_array[elbow_idx]**2))/w0.shape[0]))



print("Analysis done!")
