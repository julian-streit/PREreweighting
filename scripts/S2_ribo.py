#!/usr/bin/env python

#### THIS SCRIPT CALCULATES AN ORDER PARAMETER FOR THE INTERACTION VECTORS BETWEEN THE NITROXIDE AND AMIDES ####

# Import modules

import numpy as np
import MDAnalysis as md
import math as math


path = '../'

u = md.Universe(path+'nc.pdb',path+'nc_traj.xtc')

HN_selection = u.select_atoms('resid 10:114 and name HN and not resname PRO')
dataHN = np.array([HN_selection.ts.positions for ts in u.trajectory])

# spinlabel position approximated by CA atom
spinlabels = {
    'C657': 'resid 21 and name CA',
    'C671': 'resid 35 and name CA',
    'C699': 'resid 63 and name CA',
    'C706': 'resid 70 and name CA',
    'C720': 'resid 84 and name CA',
    'C734': 'resid 98 and name CA',
    'C740': 'resid 104 and name CA',
    'C744': 'resid 108 and name CA'
    }


# functions to calculate S2
Y2m2 = lambda x: 0.386274 * (x[0]-1j*x[1])**2 / (x[0]**2+x[1]**2+x[2]**2)**2.5
Y2m1 = lambda x: 0.772548 * (x[0]-1j*x[1])*x[2] / (x[0]**2+x[1]**2+x[2]**2)**2.5
Y20 = lambda x: 0.31539 * (2*x[2]**2-x[0]**2-x[1]**2) / (x[0]**2+x[1]**2+x[2]**2)**2.5
Y21 = lambda x: 0.772548 * (x[0]+1j*x[1])*x[2] / (x[0]**2+x[1]**2+x[2]**2)**2.5
Y22 = lambda x: 0.386274 * (x[0]+1j*x[1])**2 / (x[0]**2+x[1]**2+x[2]**2)**2.5

# dictionaries for data

rm6_dict = {}
y2m2_dict = {}
y2m1_dict = {}
y20_dict = {}
y21_dict = {}
y22_dict = {}

for spinlabel, selector in spinlabels.items():

    print("Reading in spin label positions for {}...".format(spinlabel))
    SL = u.select_atoms(selector)
    dataSL = np.array([SL.ts.positions for ts in u.trajectory])


    print("Calculating distances between amide nitrogens and spin label...")
    dx = dataHN - dataSL

    # apply a little function to calculate r^-6 for each (dx, dy, dz)
    calc_rminus6 = lambda dx: (np.linalg.norm(dx))**-6
    rminus6 = np.apply_along_axis(calc_rminus6, 2, dx)

    print("Calculating order parameters...")
    y2m2 = np.apply_along_axis(Y2m2, 2, dx)
    y2m1 = np.apply_along_axis(Y2m1, 2, dx)
    y20 = np.apply_along_axis(Y20, 2, dx)
    y21 = np.apply_along_axis(Y21, 2, dx)
    y22 = np.apply_along_axis(Y22, 2, dx)

    print("Appending data to dict...")
    rm6_dict[spinlabel] = rminus6
    y2m2_dict[spinlabel] = y2m2
    y2m1_dict[spinlabel] = y2m1
    y20_dict[spinlabel] = y20
    y21_dict[spinlabel] = y21
    y22_dict[spinlabel] = y22


print("Saving data...")

S2_data = {}
S2_data['rm6'] = rm6_dict
S2_data['y2m2'] = y2m2_dict
S2_data['y2m1'] = y2m1_dict
S2_data['y20'] = y20_dict
S2_data['y21'] = y21_dict
S2_data['y22'] = y22_dict

np.save('S2_data_intra.npy',S2_data)

print("Done!")
