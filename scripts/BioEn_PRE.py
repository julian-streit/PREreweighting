#### THIS SCRIPT IS A MODIFIED VERSION OF BIOEN ####
#### THE SCRIPT PERFORMS REWEIGHTING WITH PRE DATA INCLUDING UPPER AND LOWER BOUND RESTRAINTS ####
#### THE REWEIGHTING IS DONE IN AN ITERATIVE MANNER FOR THE RNC TO ITERATIVELY UPDATE AND CONVERGE THE EFFECTIVE TAUCs ####

# BioEn github: https://github.com/bio-phys/BioEn
# REFERENCES
#[Hummer2015] Hummer G. and Koefinger J., Bayesian Ensemble Refinement by Replica Simulations and Reweighting. J. Chem. Phys. 143(24):12B634_1 (2015). https://doi.org/10.1063/1.4937786
#[Rozycki2011] Rozycki B., Kim Y. C., Hummer G., SAXS Ensemble Refinement of ESCRT-III Chmp3 Conformational Transitions Structure 19 109–116 (2011). https://doi.org/10.1016/j.str.2010.10.006
#[Reichel2018] Reichel K., Stelzl L. S., Köfinger J., Hummer G., Precision DEER Distances from Spin-Label Reweighting, J. Phys. Chem. Lett. 9 19 5748-5752 (2018). https://doi.org/10.1021/acs.jpclett.8b02439
#[Köfinger2019] Koefinger J., Stelzl L. S. Reuter K., Allande C., Reichel K., Hummer G., Efficient Ensemble Refinement by Reweighting J. Chem. Theory Comput. Article ASAP https://doi.org/10.1021/acs.jctc.8b01231


# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import MDAnalysis as md
import math
import random
import pandas as pd
import os.path


# Note on nomenclature of saved isfile
# name_theta_checkpointit
# checkpointit is iteration number. checkpointit = 0 is the prior (uniform weights)

# Theta
thetas = [1e12]

# number of iterations to converge weights and effective tauC values
n_iterations = 20

# ensemble used here
u = md.Universe('../nc_modelrc_EM_1.gro')


residues_fln5 = u.select_atoms('resid 10:114 and name H').resids + 636 # FLN5 residues NMR

print('Maximum possible NMR-observable residues:',len(residues_fln5))

print('NMR residues: \n',residues_fln5)


# load experimental data
path = '../exp_data_rnc/'

dataC657 = pd.read_csv(path+'31_A3A3_C657_V747_data.csv')
dataC671 = pd.read_csv(path+'31_A3A3_C671_V747_data.csv')
dataC699 = pd.read_csv(path+'31_A3A3_C699_V747_data.csv')
dataC706 = pd.read_csv(path+'31_A3A3_C706_V747_data.csv')
dataC720 = pd.read_csv(path+'31_A3A3_C720_V747_data.csv')
dataC734 = pd.read_csv(path+'31_A3A3_C734_V747_data.csv')
dataC740 = pd.read_csv(path+'31_A3A3_C740_V747_data.csv')
dataC744 = pd.read_csv(path+'31_A3A3_C744_V747_data.csv')
dataC53 = pd.read_csv(path+'31_A3A3_V747_L24_N53C_data.csv')
dataC90 = pd.read_csv(path+'31_A3A3_V747_L23_G90C_data.csv')


# Iteration number (check if previous iteration exists to start from otherwise start at it = 1)

checkpoints_path = './results/checkpoints/'

if os.path.isfile(checkpoints_path+'checkpoint_it_{}.npy'.format(thetas[0])):

    checkpoint_it = int(np.load(checkpoints_path+'checkpoint_it_{}.npy'.format(thetas[0]))[1])

else:

    checkpoint_it = 1


######################################################
### While loop: iterate BioEn until nit is reached ###
######################################################

while checkpoint_it <= n_iterations:

    # Load starting weights array for this iteration (if it = 0, starting from uniform). Else, starting with weights from previous iteration

    weights_path = './results/wopt_data/'

    N_models = np.load('../rm6_data_final.npy',allow_pickle=True).item()['C657'].shape[0]

    if checkpoint_it ==1:

        w0 = np.ones(N_models)/N_models
        w_init = np.ones(N_models)/N_models

    else:

        w0 = np.load(weights_path+'res_{}_{}.npy'.format(thetas[0],checkpoint_it-1),allow_pickle=True).item()['wopt']
        w_init = w0


    ##############################################
    ## Collating Experimental & Simulation Data ##
    ##############################################

    # Calculate effective tauC (residue-specific) from order parameters if this is the first iterations
    # Else load order parameters from previous iteration result

    S2_data_path = '../../S2_calc/'

    # Frame indices of final frames (MTSL compatible) in 100k ensemble)

    frame_idxs = []
    with open('../final_frames.ndx','r') as f:
        lines = f.readlines()

    for l in lines[1:]:
        frame_idxs.append(int(l)-1) # convert back to 0-indexed

    frame_idxs = np.array(frame_idxs)

    # load S2 data and calculate weighted ensemble average

    s2_data_intra_modelrc = np.load(S2_data_path+'S2_data_intra.npy',allow_pickle=True).item()
    s2_data_inter_modelrc = np.load(S2_data_path+'S2_data_inter.npy',allow_pickle=True).item()

    # rotational correlation time bounds for ribosome at 10C in water in ns
    ribo_tauR = 4300

    # isolated tauC in ns
    tauC_iso = 3

    # if this is the first iteration calculate starting order parameters and tauC values
    if checkpoint_it == 1:

        S2_intra_modelrc = {}

        for spinlabel in s2_data_intra_modelrc['rm6'].keys():

            rminus6_avg = np.average(s2_data_intra_modelrc['rm6'][spinlabel][frame_idxs],weights = w0,axis=0)
            y2m2_avg = np.average(s2_data_intra_modelrc['y2m2'][spinlabel][frame_idxs],weights = w0,axis=0)
            y2m1_avg = np.average(s2_data_intra_modelrc['y2m1'][spinlabel][frame_idxs],weights = w0,axis=0)
            y20_avg = np.average(s2_data_intra_modelrc['y20'][spinlabel][frame_idxs],weights = w0,axis=0)
            y21_avg = np.average(s2_data_intra_modelrc['y21'][spinlabel][frame_idxs],weights = w0,axis=0)
            y22_avg = np.average(s2_data_intra_modelrc['y22'][spinlabel][frame_idxs],weights = w0,axis=0)

            S2_intra_modelrc[spinlabel] = 4*np.pi/5 / rminus6_avg * (np.abs(y2m2_avg)**2+np.abs(y2m1_avg)**2+np.abs(y20_avg)**2+np.abs(y21_avg)**2+np.abs(y22_avg)**2)

        S2_inter_modelrc = {}

        for spinlabel in s2_data_inter_modelrc['rm6'].keys():

            rminus6_avg = np.average(s2_data_inter_modelrc['rm6'][spinlabel][frame_idxs],weights = w0,axis=0)
            y2m2_avg = np.average(s2_data_inter_modelrc['y2m2'][spinlabel][frame_idxs],weights = w0,axis=0)
            y2m1_avg = np.average(s2_data_inter_modelrc['y2m1'][spinlabel][frame_idxs],weights = w0,axis=0)
            y20_avg = np.average(s2_data_inter_modelrc['y20'][spinlabel][frame_idxs],weights = w0,axis=0)
            y21_avg = np.average(s2_data_inter_modelrc['y21'][spinlabel][frame_idxs],weights = w0,axis=0)
            y22_avg = np.average(s2_data_inter_modelrc['y22'][spinlabel][frame_idxs],weights = w0,axis=0)

            S2_inter_modelrc[spinlabel] = 4*np.pi/5 / rminus6_avg * (np.abs(y2m2_avg)**2+np.abs(y2m1_avg)**2+np.abs(y20_avg)**2+np.abs(y21_avg)**2+np.abs(y22_avg)**2)


        avg_S2_data_path = './results/NC_S2_data/'

        np.save(avg_S2_data_path+'S2_avg_intra_modelrc_lcurve_{}_{}.npy'.format(thetas[0],0),S2_intra_modelrc)
        np.save(avg_S2_data_path+'S2_avg_inter_modelrc_lcurve_{}_{}.npy'.format(thetas[0],0),S2_inter_modelrc)

        # convert weighted S2 averages to effective tauC values

        # convert to tauC and append to dict
        eff_tauC_path = './results/eff_tauc_data/'
        eff_tauC_intra_modelrc = {}

        for spinlabel in list(S2_intra_modelrc.keys()):

            # calculate effective tauC

            eff_tauC = S2_intra_modelrc[spinlabel]*ribo_tauR+(1-S2_intra_modelrc[spinlabel])*tauC_iso

            # append
            eff_tauC_intra_modelrc[spinlabel] = eff_tauC

        np.save(eff_tauC_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(thetas[0],0),eff_tauC_intra_modelrc)


        eff_tauC_inter_modelrc = {}

        for spinlabel in list(S2_inter_modelrc.keys()):

                # calculate effective tauC

                eff_tauC = S2_inter_modelrc[spinlabel]*ribo_tauR+(1-S2_inter_modelrc[spinlabel])*tauC_iso

                # append
                eff_tauC_inter_modelrc[spinlabel] = eff_tauC

        np.save(eff_tauC_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(thetas[0],0),eff_tauC_inter_modelrc)

    # if this is not the first iteration load the order parameters and tauC values from the optimised solution in the previous iteration
    else:

        S2_intra_modelrc = np.load(avg_S2_data_path+'S2_avg_intra_modelrc_lcurve_{}_{}.npy'.format(thetas[0],checkpoint_it-1),allow_pickle=True).item()
        S2_inter_modelrc = np.load(avg_S2_data_path+'S2_avg_inter_modelrc_lcurve_{}_{}.npy'.format(thetas[0],checkpoint_it-1),allow_pickle=True).item()
        eff_tauC_intra_modelrc = np.load(eff_tauC_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(thetas[0],checkpoint_it-1),allow_pickle=True).item()
        eff_tauC_inter_modelrc =np.load(eff_tauC_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(thetas[0],checkpoint_it-1),allow_pickle=True).item()


    # load effective tauC values (residue-specific)

    tau_C_intra = eff_tauC_intra_modelrc
    tau_C_inter = eff_tauC_inter_modelrc

    tau_C = {}

    for key,values in zip(tau_C_intra.keys(),tau_C_intra.values()):
        tau_C[key] = values


    ribo_labels = ['L23_C90','L24_C53']
    ribo_keys = ['C3064', 'C3126']

    for l,k in zip(ribo_labels,ribo_keys):
        tau_C[k] = tau_C_inter[l]


    ## DONE TO HERE, 05042021 ##

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


    # constants for PRE rate calculation from rm6 and S2 data
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

    tau_I = 0.5e-9

    # calculate total correlation time from tau_C
    spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']
    tau_T = {}
    for spinlabel in spinlabels:
        tau_C[spinlabel] = tau_C[spinlabel]*1e-9
        tau_T[spinlabel] = 1/((1/tau_C[spinlabel])+(1/tau_I))




    dfs = [dataC657,dataC671,dataC699,dataC706,dataC720,dataC734,dataC740,dataC744,dataC90,dataC53]



    # C3064 = L23 C90, C3126 = L24 C53
    spinlabels = ['C657','C671','C699','C706','C720','C734','C740','C744','C3064', 'C3126']

    for i,spinlabel in zip(range(len(dfs)),spinlabels):

        # setting dataframe
        df = dfs[i]

        # setting the larmor frequency based on the dataset specific magnetic field strength
        larmor_H = larmor_dict[spinlabel]

        # setting labelling site specific simulation data (distances in Angstrom)

        rm6 = np.load('../rm6_data_final.npy',allow_pickle=True).item()[spinlabel]
        S2 = np.load('../S2_data_final.npy',allow_pickle=True).item()[spinlabel]

        sim_data = (S2*(K*rm6)*((4*tau_C[spinlabel]+((3*tau_C[spinlabel])/(1+(larmor_H**2)*(tau_C[spinlabel]**2)))))
                        + (1-S2)*(K*rm6)*((4*tau_T[spinlabel]+((3*tau_T[spinlabel])/(1+(larmor_H**2)*(tau_T[spinlabel]**2))))))

        # restraint string prefix
        prefix = str('c{}_'.format(spinlabel[-3:]))




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
                    lower_errors.append(df[df.columns[-2]][j]*(df['Combined_error (%)'][j]/100))
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
                pass
            if len(up_idxs)>0:
                sim_upper = np.concatenate((sim_upper,sim_data[:,np.array(up_idxs)]),axis=1)
            else:
                pass
            if len(low_idxs)>0:
                sim_lower = np.concatenate((sim_lower,sim_data[:,np.array(low_idxs)]),axis=1)
            else:
                pass



    # assumed uncertainty in back-calculation --> 0


    gaussian = np.array(gaussian)
    gaussian_errors = np.array(gaussian_errors)
    gaussian_residues = np.array(gaussian_residues)

    # set error for upper lower bound restraints based on average relativ uncertainty of these restraint classes

    upper = np.array(upper)
    upper_residues = np.array(upper_residues)
    upper_errors = np.array(upper_errors)

    lower = np.array(lower)
    lower_residues = np.array(lower_residues)
    lower_errors = np.array(lower_errors)


    print('Number of Restraints with Gaussian Error:',len(gaussian))
    print('Number of Restraints with Upper Bound:',len(upper))
    print('Number of Restraints with Lower Bound:',len(lower))
    print('Number of labelling sites used:',len(dfs))
    total_restraints = len(gaussian)+len(upper)+len(lower)
    print('Total number of restraints:',total_restraints)
    print('Avg. number of restraints per residue:',(len(gaussian)+len(upper)+len(lower))/105)

    ################
    # Reading data #
    ################

    #####################
    # Experimental Data #
    #####################


    # Vector with experimental data: normal restraints
    exp_data_v = gaussian

    # Vector with experimental data: upper bound restraints:
    exp_data_u = upper


    # Vector of experimental data: lower bound restraints
    exp_data_l = lower

    # Vector with experimental errors
    exp_error_v = gaussian_errors
    exp_error_u = upper_errors
    exp_error_l = lower_errors




    #########################
    # Data from simulations #
    #########################



    # PREs corresponding to normal restraints
    sim_pre = sim_gaussian

    # PREs corresponding to upper bound restraints
    sim_preU = sim_upper


    # PREs corresponding to lower bound restraints
    sim_preL = sim_lower


    # Matrix with data from simulations
    # Order has to be the same as in experimental data
    sim_data = []
    for i in range(sim_pre.shape[1]):
        sim_data.append(sim_pre[:,i])

    # Matrix with data from simulations
    # Order has to be the same as in experimental data
    sim_dataU = []
    for i in range(sim_preU.shape[1]):
        sim_dataU.append(sim_preU[:,i])

    # Matrix with data from simulations
    # Order has to be the same as in experimental data
    #sim_dataL = []
    #for i in range(sim_preL.shape[1]):
        #sim_dataL.append(sim_preL[:,i])


    sim_data_m = np.array(sim_data)
    sim_data_u = np.array(sim_dataU)
    #sim_data_l = np.array(sim_dataL)



    # Number of models from trajectory
    N_models = np.shape(sim_data_m)[1]




    #################
    # Log - Weights #
    #################

    # Reference log-weights, with one vaue set to 0
    # Actually for uniform reference weights this sets the whole log-weight vector to zero - is this a problem?
    g0 = np.log(w0)
    g0 -= g0[-1]

    # Log-weights for initialization of the optimization protocol
    g_init = np.log(w_init)
    g_init -= g_init[-1]

    # Returns proper weights from log-weights after normalisation
    def getWeights(g):
        tmp = np.exp(g)
        s = tmp.sum()
        w = np.array(tmp / s)
        return w,s


    #########################################
    # Parameters for optimization algorithm #
    #########################################

    # For now we can only use BFGS algorithm as is coded in SCIPY

    epsilon = 0.1
    gtol = 0.00001
    maxiter = 1e8
    maxfun = 1e8


    # Log Prior base funtion
    # Log Prior in the log-weights representation:
    # theta * ((g.T * w) - (g0.T * w) + np.log(s0) - np.log(s))

    def bioen_log_prior(w, s, g, g0, theta):
        w0,s0 = getWeights(g0)
        g_ave = np.sum(g * w)
        g0_ave = np.sum(g0 * w)
        log_prior = theta * (g_ave - g0_ave + np.log(s0) - np.log(s))

        return log_prior

    # Chi-Square functions



    def chiSqrTerm(w, sim_data_m, exp_data_v, exp_error_v):
        v = np.sum(sim_data_m * w, 1) - exp_data_v
        v *= (1/exp_error_v)
        chiSqr = 0.5 * np.sum(v*v)

        return chiSqr

    # for upper bound restraints
    def chiSqrTermU(w, sim_data_u, exp_data_u, exp_error_u):
        avg = np.sum(sim_data_u * w, 1)
        bool_vector = (avg>exp_data_u)*1 # creates a vector of 0 and 1. 1 = restraint not satisfied
        v = avg - exp_data_u
        v *= bool_vector
        v *= (1/exp_error_u)
        chiSqr = 0.5 * np.sum(v*v)

        return chiSqr

    # for lower bound restraints
    def chiSqrTermL(w, sim_data_l, exp_data_l, exp_error_l):
        avg = np.sum(sim_data_l * w, 1)
        bool_vector = (avg<exp_data_l)*1 # creates a vector of 0 and 1. 1 = restraint not satisfied
        v = avg - exp_data_l
        v *= bool_vector
        v *= (1/exp_error_l)
        chiSqr = 0.5 * np.sum(v*v)

        return chiSqr



    # Log Posterior base function

    def bioen_log_posterior_base(g, g0, sim_data_m, exp_data_v, exp_error_v,
                                 sim_data_u, exp_data_u, exp_error_u, theta):
        w, s = getWeights(g)
        log_prior = bioen_log_prior(w, s, g, g0, theta)
        chiSqr = chiSqrTerm(w, sim_data_m, exp_data_v, exp_error_v)
        chiSqrU = chiSqrTermU(w, sim_data_u, exp_data_u, exp_error_u)
        #chiSqrL = chiSqrTermL(w, sim_data_l, exp_data_l, exp_error_l)
        log_posterior = chiSqr + chiSqrU + log_prior

        return log_posterior









    # Gradient of log Posterior base function in the log-weights representation




    def grad_bioen_log_posterior_base(g, g0, sim_data_m, exp_data_v, exp_error_v,
                sim_data_u, exp_data_u, exp_error_u, theta):

        # constant term
        w, s = getWeights(g)

        # chi square-like terms
        tmp = np.zeros(w.shape[0])
        tmpU = np.zeros(w.shape[0])
        #tmpL = np.zeros(w.shape[0])

        # for normal restraints
        sim_ave = np.sum(sim_data_m * w,1)
        for mu in range(w.shape[0]):
            w_mu = w[mu]
            diff1 = (sim_ave - exp_data_v)*(1/exp_error_v)
            sim_mu = sim_data_m[:,mu]
            diff2 = (sim_mu - sim_ave)*(1/exp_error_v)
            tmp[mu] = w_mu * np.sum(diff1 * diff2)

        # for upper bound restraints
        sim_ave = np.sum(sim_data_u * w,1)
        bool_vector = (sim_ave>exp_data_u)*1 # creates a vector of 0 and 1. 1 = restraint not satisfie
        for mu in range(w.shape[0]):
            w_mu = w[mu]
            diff1 = (sim_ave - exp_data_u)*(1/exp_error_u)
            diff1 *= bool_vector
            sim_mu = sim_data_u[:,mu]
            diff2 = (sim_mu - sim_ave)*(1/exp_error_u)
            tmpU[mu] = w_mu * np.sum(diff1 * diff2)

        # for lower bound restraints
        #sim_ave = np.sum(sim_data_l * w,1)
        #bool_vector = (sim_ave<exp_data_l)*1 # creates a vector of 0 and 1. 1 = restraint not satisfie
        #for mu in range(w.shape[0]):
            #w_mu = w[mu]
            #diff1 = (sim_ave - exp_data_l)*(1/exp_error_l)
            #diff1 *= bool_vector
            #sim_mu = sim_data_l[:,mu]
            #diff2 = (sim_mu - sim_ave)*(1/exp_error_l)
            #tmpL[mu] = w_mu * np.sum(diff1 * diff2)


        gradient = w * theta * (g - np.sum(g*w) - g0 + np.sum(g0*w)) + tmp + tmpU

        return gradient






    # ITERATIONS THROUGHT Thetas LBFGS
    fmin_initial_array=[]

    for theta in thetas:
        print('Theta =',theta)
        g = g_init
        fmin_initial = bioen_log_posterior_base(g, g0, sim_data_m, exp_data_v, exp_error_v, sim_data_u, exp_data_u, exp_error_u, theta)
        fmin_initial_array.append(fmin_initial)
        res=sopt.minimize(bioen_log_posterior_base,g,args = (g0, sim_data_m, exp_data_v, exp_error_v, sim_data_u, exp_data_u, exp_error_u,theta),jac = grad_bioen_log_posterior_base,method='L-BFGS-B', tol=gtol, options = {'maxiter':maxiter,'maxfun':maxfun,'disp':True})


        print('Sucess:',res.success)
        print('Current function value:',res.fun)
        print('Number of iterations:',res.nit)
        print('Number of function calls:',res.nfev)

        # extract w_opt
        gopt = res.x[()]
        wopt = getWeights(gopt)[0]

        # save results dict
        res_dict = {}
        res_dict['Success'] = res.success
        res_dict['Fun val'] = res.fun
        res_dict['Nit'] = res.nit
        res_dict['N fun calls'] = res.nfev
        res_dict['wopt'] = wopt

        np.save(weights_path+'res_{}_{}.npy'.format(thetas[0],checkpoint_it),res_dict)

        # save new order parameters
        S2_intra_modelrc = {}

        for spinlabel in s2_data_intra_modelrc['rm6'].keys():

            rminus6_avg = np.average(s2_data_intra_modelrc['rm6'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y2m2_avg = np.average(s2_data_intra_modelrc['y2m2'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y2m1_avg = np.average(s2_data_intra_modelrc['y2m1'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y20_avg = np.average(s2_data_intra_modelrc['y20'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y21_avg = np.average(s2_data_intra_modelrc['y21'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y22_avg = np.average(s2_data_intra_modelrc['y22'][spinlabel][frame_idxs],weights = wopt,axis=0)

            S2_intra_modelrc[spinlabel] = 4*np.pi/5 / rminus6_avg * (np.abs(y2m2_avg)**2+np.abs(y2m1_avg)**2+np.abs(y20_avg)**2+np.abs(y21_avg)**2+np.abs(y22_avg)**2)

        S2_inter_modelrc = {}

        for spinlabel in s2_data_inter_modelrc['rm6'].keys():

            rminus6_avg = np.average(s2_data_inter_modelrc['rm6'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y2m2_avg = np.average(s2_data_inter_modelrc['y2m2'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y2m1_avg = np.average(s2_data_inter_modelrc['y2m1'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y20_avg = np.average(s2_data_inter_modelrc['y20'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y21_avg = np.average(s2_data_inter_modelrc['y21'][spinlabel][frame_idxs],weights = wopt,axis=0)
            y22_avg = np.average(s2_data_inter_modelrc['y22'][spinlabel][frame_idxs],weights = wopt,axis=0)

            S2_inter_modelrc[spinlabel] = 4*np.pi/5 / rminus6_avg * (np.abs(y2m2_avg)**2+np.abs(y2m1_avg)**2+np.abs(y20_avg)**2+np.abs(y21_avg)**2+np.abs(y22_avg)**2)


        avg_S2_data_path = './results/NC_S2_data/'

        np.save(avg_S2_data_path+'S2_avg_intra_modelrc_lcurve_{}_{}.npy'.format(thetas[0],checkpoint_it),S2_intra_modelrc)
        np.save(avg_S2_data_path+'S2_avg_inter_modelrc_lcurve_{}_{}.npy'.format(thetas[0],checkpoint_it),S2_inter_modelrc)




        # convert to tauC and save new tauC values
        eff_tauC_path = './results/eff_tauc_data/'
        eff_tauC_intra_modelrc = {}

        for spinlabel in list(S2_intra_modelrc.keys()):

            # calculate effective tauC

            eff_tauC = S2_intra_modelrc[spinlabel]*ribo_tauR+(1-S2_intra_modelrc[spinlabel])*tauC_iso

            # append
            eff_tauC_intra_modelrc[spinlabel] = eff_tauC

        np.save(eff_tauC_path+'eff_tauc_intra_modelrc_{}_{}.npy'.format(thetas[0],checkpoint_it),eff_tauC_intra_modelrc)


        eff_tauC_inter_modelrc = {}

        for spinlabel in list(S2_inter_modelrc.keys()):

                # calculate effective tauC

                eff_tauC = S2_inter_modelrc[spinlabel]*ribo_tauR+(1-S2_inter_modelrc[spinlabel])*tauC_iso

                # append
                eff_tauC_inter_modelrc[spinlabel] = eff_tauC

        np.save(eff_tauC_path+'eff_tauc_inter_modelrc_{}_{}.npy'.format(thetas[0],checkpoint_it),eff_tauC_inter_modelrc)


    # save checkpoint_it
    checkpoint_item_to_save = ['Iteration',checkpoint_it]
    np.save(checkpoints_path+'checkpoint_it_{}.npy'.format(thetas[0]),checkpoint_item_to_save)

    # add 1 to checkpoint_it
    checkpoint_it += 1

    #########################
    ### End of while loop ###
    #########################
