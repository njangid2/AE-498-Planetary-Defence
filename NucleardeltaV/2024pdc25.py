"""
2024 PDC25: Orbit Determination and Propagation
This script performs orbit determination and propagation for the 2024 PDC25 object using the GRSS library.
The script is structured as follows:
1. Define initial conditions and settings for orbit determination (OD)
2. Load optical observations and run the OD filter to estimate the orbit and covariance
3. Save the OD results and print a summary
4. Define settings for parallel propagation
5. Sample from the OD solution covariance to create initial conditions for parallel propagation
6. Run parallel propagation for the sampled initial conditions and save the results

Author: Rahil Makadia, August 2024
"""
# pylint: disable=invalid-name
import os
import gzip
import json
import numpy as np
from astropy.time import Time

from grss import fit, prop, utils

body_id = '2024PDC25'

################## Epoch 2: horizons header from https://ssd.jpl.nasa.gov/horizons/app.html#/?id=-937020
# EPOCH= 2461002.5
#    EC= .3903977728972607  QR= 1.005918824517983  TP= 2461215.136873333567
#    OM= 214.4170705068864  W = 359.941369522211  IN= 10.68653275799602
#    SRC= -4.729401494307664E-10 4.999699999910745E-9  -8.857690106814095E-9
#          1.768819076233289E-9 -2.647063337774079E-9  -1.094039784814247E-6
#         -1.420690625988797E-8  2.48148634199217E-8   -1.311424078242986E-6
#         -3.353689722153545E-8 -9.430736211018382E-11  3.390839818277143E-10
#         -3.022218031075639E-7  1.206343342257748E-7  -1.19473371420242E-7
#         -5.364100246208384E-9  9.054632576841123E-9  -4.548772648982771E-8
#         -2.407314566673439E-7  2.265029335178455E-7  -8.8170973597317E-9
#         -4.887982511228615E-10 1.345541782988126E-10 -1.065924436219504E-6
#          9.09374621143591E-10 -4.023622604787796E-9   2.750785802423358E-11
#          3.287557722527732E-14
#   EST= A2
#    A2= 1.004247567426106E-13  R0= 1.  ALN= 1.  NM= 2.  NN= 0.  NK= 0.
#   H= 21.9  G= 0.15
init_sol = {
    't': 2461002.5-2400000.5,
    'e': 3.903977728972607E-01,
    'q': 1.005918824517983,
    'tp': 2461215.136873333567-2400000.5,
    'om': 214.4170705068864*np.pi/180,
    'w': 359.941369522211*np.pi/180,
    'i': 10.68653275799602*np.pi/180,
    'a2': 1.004247567426106E-13,
}
src = [-4.729401494307664E-10, 4.999699999910745E-9, -8.857690106814095E-9,
        1.768819076233289E-9, -2.647063337774079E-9, -1.094039784814247E-6,
        -1.420690625988797E-8, 2.48148634199217E-8, -1.311424078242986E-6,
        -3.353689722153545E-8, -9.430736211018382E-11, 3.390839818277143E-10,
        -3.022218031075639E-7, 1.206343342257748E-7, -1.19473371420242E-7,
        -5.364100246208384E-9, 9.054632576841123E-9, -4.548772648982771E-8,
        -2.407314566673439E-7, 2.265029335178455E-7, -8.8170973597317E-9,
        -4.887982511228615E-10, 1.345541782988126E-10, -1.065924436219504E-6,
        9.09374621143591E-10, -4.023622604787796E-9, 2.750785802423358E-11,
        3.287557722527732E-14
]

def upper_tri_src2full(vec):
    """
    Convert upper triangular square root covariance vector to full covariance matrix

    Parameters
    ----------
    vec : list/np.array
        Upper triangular square root covariance vector

    Returns
    -------
    cov : np.array
        Full covariance matrix
    """
    # length of cov vec is n*(n+1)/2. Solve for n using quadratic formula
    # n^2 + n - 2*len(src) = 0
    # a = 1, b = 1, c = -2*len(src)
    # n1 = (-b + sqrt(b^2 - 4ac)) / 2a
    # n2 = (-b - sqrt(b^2 - 4ac)) / 2a
    # n2 will always be negative, so take n1 as the solution
    n_val = int((-1 + np.sqrt(1 - 4*1*-2*len(vec))) / 2)
    src = np.zeros((n_val, n_val))
    k = 0
    for i in range(1, n_val+1):
        for j in range(i):
            src[j, i-1] = vec[k]
            k += 1
    cov = src @ src.T
    return cov

init_cov = upper_tri_src2full(src)
nongrav_info = {
    'a1': 0.0,
    'a2': init_sol['a2'],
    'a3': 0.0,
    'alpha': 1.0,
    'k': 0.0,
    'm': 2.0,
    'n': 0.0,
    'r0_au': 1.0,
    'radius': 75.0,
}
mass = 0.0

###################################################################################################
############################################# OD time #############################################
###################################################################################################
# OD settings
current_file_path = os.path.dirname(os.path.abspath(__file__))
optical_obs_file = f"{current_file_path}/data/2024pdc25_epoch2.xml"
t_min_tdb = None
t_max_tdb = None
debias_lowres = None
deweight = False
eliminate = False
num_obs_per_night = 4
verbose = False
accept_weights = True
n_iter_max = 25
prior_est = {
    'a2': 1.0E-13,
}
prior_sig = {
    'a2': 3.3E-14,
}
# get obs dataframe
obs_df = fit.get_optical_obs(body_id, optical_obs_file, t_min_tdb, t_max_tdb, debias_lowres,
                                deweight, eliminate, num_obs_per_night, verbose, accept_weights)
obs_df['permID'] = body_id
# find indices of obs_df where sigRA or sigDec is nan and drop them
obs_df['sigRA'] = obs_df['rmsRA'].values
obs_df['sigDec'] = obs_df['rmsDec'].values
# find indices of obs_df where sigCorr is nan and fill them with 0.0
nan_sigCorr = obs_df['sigCorr'].isna()
nan_sigCorr_idx = obs_df.index[nan_sigCorr].tolist()
obs_df.loc[nan_sigCorr_idx, 'sigCorr'] = 0.0
# fill nan values in sigTime with 1.0 second time uncertainty
nan_sigTime = obs_df['sigTime'].isna()
nan_sigTime_idx = obs_df.index[nan_sigTime].tolist()
obs_df.loc[nan_sigTime_idx, 'sigTime'] = 1.0

# create fit simulation and run filter
fit_sim = fit.FitSimulation(init_sol, obs_df, init_cov, n_iter_max=n_iter_max,
                            de_kernel=440, nongrav_info=nongrav_info)
fit_sim.name = body_id
fit_sim.prior_est = prior_est
fit_sim.prior_sig = prior_sig
fit_sim.analytic_partials = True
fit_sim.filter_lsq()

# if od_log_file exists, delete it
od_log_file = f'./{body_id}_od.log'
if os.path.exists(od_log_file):
    os.remove(od_log_file)
fit_sim.save(od_log_file)

# print summary and save solution and covariance for propagation
fit_sim.print_summary()
grss_sol = {'t': fit_sim.t_sol} | fit_sim.x_nom
grss_cov = fit_sim.covariance
with open('./data/sol.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(grss_sol, indent=4))
np.savetxt('./data/cov.txt', grss_cov, fmt='%30.20e')

###################################################################################################
#################################### parallel propagation time ####################################
###################################################################################################
# set end time to 1 day after impact
t_end = Time('2041-04-25', scale='tdb', format='iso').tdb.mjd
# define number of samples for parallel propagation
NUM_SAMPLES = int(1e3)
# define number of threads for parallel propagation
NUM_THREADS = 64

# read solution and covariance
with open('./data/sol.json', 'r', encoding='utf-8') as f:
    init_sol = json.load(f)
init_cov = np.loadtxt('./data/cov.txt')

# remove time from init_sol for sampling
elem_keys = [key for key in init_sol.keys() if key != 't']
init_sol_arr = np.array([init_sol[key] for key in elem_keys])
samples = np.random.multivariate_normal(
    init_sol_arr, init_cov, NUM_SAMPLES
)
all_points_dict = []
# convert samples back to list of dicts with time from init_sol
for i in range(NUM_SAMPLES):
    sample_dict = {'t': init_sol['t']}
    for j, key in enumerate(elem_keys):
        sample_dict[key] = samples[i, j]
    all_points_dict.append(sample_dict)

# write all samples to file for parallel propagation
sample_fname = f'./data/samples.json.gz'
with gzip.open(sample_fname, 'wt', encoding='utf-8') as file:
    file.write(json.dumps(all_points_dict, ensure_ascii=True, indent=4))
print(f'Wrote {NUM_SAMPLES} samples to {sample_fname}')

# read samples back from file for parallel propagation
with gzip.open(sample_fname, 'rt', encoding='utf-8') as file:
    prop_samples_dict = json.load(file)

# set up reference propagation simulation
prop_sim = prop.PropSimulation(
    name=body_id,
    t0=init_sol['t'],
    defaultSpiceBodies=440,
    DEkernelPath=utils.default_kernel_path)
prop_sim.set_integration_parameters(t_end)

# set up reference nongrav parameters
ng_params = prop.NongravParameters()
ng_params.a1 = init_sol.get('a1', nongrav_info['a1'])
ng_params.a1Est = 'a1' in init_sol
ng_params.a2 = init_sol.get('a2', nongrav_info['a2'])
ng_params.a2Est = 'a2' in init_sol
ng_params.a3 = init_sol.get('a3', nongrav_info['a3'])
ng_params.a3Est = 'a3' in init_sol
ng_params.alpha = nongrav_info['alpha']
ng_params.k = nongrav_info['k']
ng_params.m = nongrav_info['m']
ng_params.n = nongrav_info['n']
ng_params.r0_au = nongrav_info['r0_au']

# run parallel propagation
result = prop.parallel_propagate(
    init_sol, ng_params, prop_sim,
    prop_samples_dict, NUM_THREADS,
    reconstruct=True
)

# Run this in the .py file after parallel_propagate with reconstruct=True
# Save the nominal (first clone = nominal solution) b-plane data

ca_nom = result[0][0][0]   # first clone = nominal
 
nominal_bplane = {
    't_ca'  : ca_nom.t,          # MJD of close approach
    'bMag'  : ca_nom.bMag,       # |b| in AU
    'xi'    : ca_nom.opik.x,     # ξ in AU
    'zeta'  : ca_nom.opik.y,     # ζ in AU
    'vInf'  : ca_nom.vInf,       # U∞ in AU/day
    'xRel'  : list(ca_nom.xRel), # relative state [x,y,z,vx,vy,vz]
}

# Also save the full 1000-clone cloud for uncertainty ellipse
cloud = []
for i in range(len(result[0])):
    if result[0][i]:
        ca = result[0][i][0]
        cloud.append({
            'xi'  : ca.opik.x,
            'zeta': ca.opik.y,
            'bMag': ca.bMag,
        })

with open('./data/nominal_bplane.json', 'w') as f:
    json.dump({'nominal': nominal_bplane, 'cloud': cloud}, f, indent=2)

print(f'Saved nominal b-plane + {len(cloud)} clone cloud')