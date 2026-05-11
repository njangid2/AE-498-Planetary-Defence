"""
Nuclear Standoff Deflection – Monte Carlo Pipeline  (v2 — live solid angle)
============================================================================
Pipeline:
  1. Load OD solution + covariance (from Epoch 2 OD script)
  2. Load physical-property MC CSV  (5 000 triaxial ellipsoid realizations)
  3. Sample N orbital clones from OD covariance
  4. Propagate all clones to t_blast (intercept date) using grss
  5. At t_blast, define blast point behind nominal in -T direction
  6. For each clone i, sample ALL uncertainty sources and compute ΔV:

       Source 1 — Orbital            clone's own (r_i, v_i) → T_hat_i
       Source 2 — Mass               log-normal  (P25=3.30e9, P75=4.58e9 kg)
       Source 3 — Diameter           Gaussian    (P25=148 m, P75=152 m)
       Source 4 — Yield              Gaussian    ±9.1 %  (mission design spec)
       Source 5 — Standoff height    Gaussian    ±5 %
       Source 6 — Coupling eff. η    Gaussian    ±30 %, clipped [0.02, 0.10]
       Source 7 — Ejecta velocity    Gaussian    ±30 %, clipped [1000, 4000]
       Source 8 — Shape / solid Ω   direct from CSV row  (bootstrap resample)
                  f_geo_i = π·b_i·c_i / (4π · r_burst_i²)
                  where r_burst_i = R_i + H_i  (Sources 3 + 5)

  7. Apply per-clone ΔV to each clone's velocity at t_blast
  8. Propagate deflected and undeflected clouds to
     t_end_extended = t_impact + 30,000 days (~82 yr) to capture all
     future resonant close approaches — matches old nominal code which
     used: mjd_end_defl = nom['t_ca'] + 30000
  9. Extract worst Earth b-plane (ξ, ζ) per clone and plot overlay

SOLID ANGLE NOTE
----------------
Ω is NOT a hardcoded Gaussian any more.  Each clone draws its shape directly
from the physical-property CSV (bootstrap resampled), so the shape uncertainty,
standoff uncertainty, and radius uncertainty are all propagated coherently
through a single expression:

    Ω_i  =  π · b_i · c_i  /  r_burst_i²       [sr]
    f_geo_i  =  Ω_i  /  (4π)

EXTENDED PROPAGATION NOTE
--------------------------
Both clouds are integrated 30,000 days (~82 yr) past the nominal impact date.
propagate_to_epoch selects the WORST (min bMag) Earth event across ALL CAs
and impacts in the full window — same logic as the old nominal code's
get_worst_earth_ca().  caTol = 0.1 AU captures strongly deflected clones.

Author: adapted from Rahil Makadia OD script + PDC25 Epoch 2 physical MC
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from grss import prop, utils

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = './data/run_1.log'
os.makedirs('./data', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

RUN_START = time.perf_counter()
log.info('='*70)
log.info('Nuclear Standoff Deflection MC Pipeline  v2b (live solid angle + extended propagation)')
log.info('='*70)

# ── Constants ─────────────────────────────────────────────────────────────────
AU_KM = 1.495978707e8   # km per AU
AU_M  = AU_KM * 1e3     # m  per AU
DAY_S = 86400.0         # seconds per day

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = './data'
SOL_FILE  = f'{DATA_DIR}/sol.json'
COV_FILE  = f'{DATA_DIR}/cov.txt'
PHYS_CSV  = f'{DATA_DIR}/pdc-25-epoch2-ellipse-subset.csv'
OUT_UNDEF = f'{DATA_DIR}/bplane_undeflected_1.json'
OUT_DEF   = f'{DATA_DIR}/bplane_deflected_1.json'
PLOT_FILE = f'{DATA_DIR}/bplane_overlay_1.png'

# ── Monte Carlo settings ───────────────────────────────────────────────────────
NUM_SAMPLES = 100000
NUM_THREADS = 64
RANDOM_SEED = 42

# ── Mission / blast nominal settings ──────────────────────────────────────────
T_BLAST_ISO = '2033-11-01'   # intercept date (TDB ISO)
T_END_ISO   = '2041-04-25'   # nominal impact date

# Extended propagation: 30,000 days past nominal impact (~82 years)
T_END_EXTENDED_DAYS = 21200

# ── Nominal asteroid physical properties ──────────────────────────────────────
R_AST_M  = 75.0     # nominal radius [m]
MASS_AST = 3.97e9   # nominal mass   [kg]

# ── Nominal nuclear parameters ────────────────────────────────────────────────
Y_MT         = 1.0      # yield [megatons]
H_STANDOFF_M = 1100.0   # standoff height above surface [m]
ETA_ABLATION = 0.05     # X-ray → momentum coupling efficiency (S-type)
V_EJ_MS      = 2000.0   # ablation ejecta velocity [m/s]

# ── Nongrav parameters (must match OD script) ─────────────────────────────────
NONGRAV_INFO = {
    'a1': 0.0, 'a2': 1.004247567426106e-13, 'a3': 0.0,
    'alpha': 1.0, 'k': 0.0, 'm': 2.0, 'n': 0.0,
    'r0_au': 1.0, 'radius': 75.0,
}
BODY_ID = '2024PDC25'

log.info('Configuration:')
log.info('  NUM_SAMPLES=%d  NUM_THREADS=%d  RANDOM_SEED=%d',
         NUM_SAMPLES, NUM_THREADS, RANDOM_SEED)
log.info('  T_BLAST=%s  T_END=%s  EXTENDED_DAYS=%d',
         T_BLAST_ISO, T_END_ISO, T_END_EXTENDED_DAYS)
log.info('  Y_MT=%.2f  H_STANDOFF=%.0f m  ETA=%.3f  V_EJ=%.0f m/s',
         Y_MT, H_STANDOFF_M, ETA_ABLATION, V_EJ_MS)


###############################################################################
# ── Solid angle helpers ───────────────────────────────────────────────────────
###############################################################################

def solid_angle_from_axes(b_m, c_m, r_burst_m):
    """Solid angle [sr] of a triaxial ellipsoid seen broadside (LOS ∥ a)."""
    return np.pi * b_m * c_m / r_burst_m**2


def solid_angle_distribution(b_arr, c_arr, d_surface_km,
                              view_axis='a', a_arr=None):
    """Compute Ω distribution over an array of ellipsoid shapes."""
    d_m = d_surface_km * 1000.0
    if view_axis == 'a':
        A_proj = np.pi * b_arr * c_arr
        d_ctr  = d_m + (a_arr if a_arr is not None else 0.0)
    elif view_axis == 'b':
        A_proj = np.pi * a_arr * c_arr
        d_ctr  = d_m + b_arr
    else:
        A_proj = np.pi * a_arr * b_arr
        d_ctr  = d_m + c_arr
    return A_proj / d_ctr**2


###############################################################################
# ── Other helpers ─────────────────────────────────────────────────────────────
###############################################################################

def make_ng_params(sol_dict):
    ng = prop.NongravParameters()
    ng.a1 = sol_dict.get('a1', NONGRAV_INFO['a1']); ng.a1Est = 'a1' in sol_dict
    ng.a2 = sol_dict.get('a2', NONGRAV_INFO['a2']); ng.a2Est = 'a2' in sol_dict
    ng.a3 = sol_dict.get('a3', NONGRAV_INFO['a3']); ng.a3Est = 'a3' in sol_dict
    ng.alpha = NONGRAV_INFO['alpha']; ng.k = NONGRAV_INFO['k']
    ng.m     = NONGRAV_INFO['m'];     ng.n = NONGRAV_INFO['n']
    ng.r0_au = NONGRAV_INFO['r0_au']
    return ng


def rtn_frame(r_vec, v_vec):
    """RTN unit vectors from position and velocity (AU, AU/day)."""
    R_hat = np.array(r_vec) / np.linalg.norm(r_vec)
    N_hat = np.cross(r_vec, v_vec); N_hat /= np.linalg.norm(N_hat)
    T_hat = np.cross(N_hat, R_hat); T_hat /= np.linalg.norm(T_hat)
    return R_hat, T_hat, N_hat


def ablation_dv_ms(E_total_J, m_ast_kg, v_ej_ms, f_geo, eta=ETA_ABLATION):
    """Ablation ΔV [m/s]:  ΔV = 2·eta·f_geo·E / (m·v_ej)"""
    E_dep = eta * f_geo * E_total_J
    return (2.0 * E_dep) / (m_ast_kg * v_ej_ms)


def lognormal_from_percentiles(p25, p75):
    """Fit log-normal (mu_ln, sigma_ln) exactly from P25 and P75."""
    mu_ln    = (np.log(p75) + np.log(p25)) / 2.0
    sigma_ln = (np.log(p75) - np.log(p25)) / (2.0 * 0.6745)
    return mu_ln, sigma_ln


def propagate_to_epoch(init_sol, ng_params, prop_sim_ref,
                       samples_list, num_threads, t_end_mjd):
    """
    Wrapper around grss parallel_propagate.
    Selects WORST (min bMag) Earth event per clone across full integration window.
    """
    result = prop.parallel_propagate(
        init_sol, ng_params, prop_sim_ref,
        samples_list, num_threads, reconstruct=True)

    cloud = []
    for i in range(len(result[0])):
        impact_list = result[1][i]
        ca_list     = result[0][i]

        all_events   = list(impact_list) + list(ca_list)
        earth_events = [
            e for e in all_events
            if 'Earth' in str(getattr(e, 'centralBody', ''))
            or getattr(e, 'centralBodySpiceId', 0) == 399
        ]

        if not earth_events:
            cloud.append(None)
            continue

        worst = min(earth_events, key=lambda e: e.bMag)
        cloud.append({
            'xi'  : worst.opik.x,
            'zeta': worst.opik.y,
            'bMag': worst.bMag,
            't_ca': worst.t,
        })
    return cloud, result


###############################################################################
# STEP 1 — Load physical-property MC (shape realizations)
###############################################################################
log.info('-'*70)
log.info('STEP 1 — Loading physical-property MC (ellipsoid shape realizations)')
t0_step = time.perf_counter()

phys_df = pd.read_csv(PHYS_CSV)
csv_a   = phys_df['a'].values
csv_b   = phys_df['b'].values
csv_c   = phys_df['c'].values
N_CSV   = len(phys_df)

a_nom, b_nom, c_nom = csv_a.mean(), csv_b.mean(), csv_c.mean()
r_burst_nom_m = R_AST_M + H_STANDOFF_M

omega_nom = solid_angle_from_axes(b_nom, c_nom, r_burst_nom_m)
f_geo_nom = omega_nom / (4.0 * np.pi)

sin_t_sph = np.clip(R_AST_M / r_burst_nom_m, 0.0, 1.0)
f_geo_sph = 0.5 * (1.0 - np.sqrt(1.0 - sin_t_sph**2))

omega_csv_dist = solid_angle_from_axes(csv_b, csv_c, r_burst_nom_m)

log.info('  CSV rows           : %d', N_CSV)
log.info('  Mean semi-axes     : a=%.1f m  b=%.1f m  c=%.1f m', a_nom, b_nom, c_nom)
log.info('  r_burst (nominal)  : %.1f m', r_burst_nom_m)
log.info('  Ω (nominal shape)  : %.5f sr', omega_nom)
log.info('  Ω dist (CSV)       : mean=%.5f  std=%.5f  (%.1f%%)',
         omega_csv_dist.mean(), omega_csv_dist.std(),
         100*omega_csv_dist.std()/omega_csv_dist.mean())
log.info('  f_geo (solid angle): %.6f', f_geo_nom)
log.info('  f_geo (sphere leg.): %.6f  → solid-angle is %.2f× larger',
         f_geo_sph, f_geo_nom/f_geo_sph)

log.info('  Ω vs surface distance (broadside, LOS ∥ a-axis):')
log.info('  %12s  %12s  %10s  %10s', 'd_surf [km]', 'mean Ω [sr]', '5th %ile', '95th %ile')
for d_km in [0.5, 1.0, 2.0, 5.0, 10.0]:
    omega_d = solid_angle_distribution(csv_b, csv_c, d_km,
                                        view_axis='a', a_arr=csv_a)
    log.info('  %12.1f  %12.5f  %10.5f  %10.5f',
             d_km, omega_d.mean(),
             np.percentile(omega_d, 5), np.percentile(omega_d, 95))

log.info('  Step 1 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 2 — Load OD solution and covariance
###############################################################################
log.info('-'*70)
log.info('STEP 2 — Loading OD solution and covariance')
t0_step = time.perf_counter()

with open(SOL_FILE, 'r', encoding='utf-8') as f:
    init_sol = json.load(f)
init_cov = np.loadtxt(COV_FILE)

t0_mjd             = init_sol['t']
t_end_mjd          = Time(T_END_ISO, scale='tdb', format='iso').tdb.mjd
t_blast_mjd        = Time(T_BLAST_ISO, scale='tdb', format='iso').tdb.mjd
t_end_extended_mjd = t_end_mjd + T_END_EXTENDED_DAYS

assert t_blast_mjd > t0_mjd,    "Blast date must be after OD epoch"
assert t_blast_mjd < t_end_mjd, "Blast date must be before impact epoch"

log.info('  OD epoch        : MJD %.2f', t0_mjd)
log.info('  Blast date      : %s  (MJD %.2f)', T_BLAST_ISO, t_blast_mjd)
log.info('  Nominal impact  : %s  (MJD %.2f)', T_END_ISO, t_end_mjd)
log.info('  Extended end    : MJD %.2f  (+%d days, ~%.0f yr past impact)',
         t_end_extended_mjd, T_END_EXTENDED_DAYS, T_END_EXTENDED_DAYS/365.25)
log.info('  Covariance matrix shape: %s', init_cov.shape)
log.info('  Step 2 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 3 — Sample N orbital clones from OD covariance
###############################################################################
log.info('-'*70)
log.info('STEP 3 — Sampling %d orbital clones from OD covariance', NUM_SAMPLES)
t0_step = time.perf_counter()

np.random.seed(RANDOM_SEED)
elem_keys    = [k for k in init_sol.keys() if k != 't']
init_sol_arr = np.array([init_sol[k] for k in elem_keys])
raw_samples  = np.random.multivariate_normal(init_sol_arr, init_cov, NUM_SAMPLES)

samples_list = [{'t': t0_mjd, **{k: raw_samples[i, j]
                 for j, k in enumerate(elem_keys)}}
                for i in range(NUM_SAMPLES)]

log.info('  Elements sampled : %s', elem_keys)
log.info('  Sample array shape: %s', raw_samples.shape)
log.info('  Step 3 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 4 — Propagate all clones to t_blast
###############################################################################
log.info('-'*70)
log.info('STEP 4 — Propagating %d clones to blast epoch', NUM_SAMPLES)
t0_step = time.perf_counter()

def state_at_t(sol_dict, t_query_mjd):
    """Propagate one clone to t_query_mjd, return [x,y,z,vx,vy,vz] AU/AU/day."""
    sim = prop.PropSimulation(BODY_ID, sol_dict['t'], 440,
                               utils.default_kernel_path)
    ng  = make_ng_params(sol_dict)
    sim.set_integration_parameters(t_query_mjd,
        tEval=[t_query_mjd], tEvalUTC=False,
        evalApparentState=False, convergedLightTime=False)
    cometary = [sol_dict[k] for k in ['e', 'q', 'tp', 'om', 'w', 'i']]
    sim.add_integ_body(prop.IntegBody(
        BODY_ID, sol_dict['t'], 0.0,
        NONGRAV_INFO['radius'] / AU_KM, cometary, ng))
    sim.integrate()
    return np.array(sim.xIntegEval[0][:6])

log.info('  Computing nominal state at t_blast ...')
nom_state = state_at_t(init_sol, t_blast_mjd)
r_nom, v_nom = nom_state[:3], nom_state[3:]
log.info('  Nominal pos at blast: %s AU', r_nom)

log.info('  Computing %d clone states ...', NUM_SAMPLES)
clone_states_at_blast = []
n_failed = 0
for i, sample in enumerate(samples_list):
    if i % 10000 == 0:
        log.info('    Clone %d/%d  (elapsed %.1f s)',
                 i, NUM_SAMPLES, time.perf_counter() - t0_step)
    try:
        clone_states_at_blast.append(state_at_t(sample, t_blast_mjd))
    except Exception as e:
        log.warning('    Clone %d failed (%s) — using nominal state', i, e)
        clone_states_at_blast.append(None)
        n_failed += 1

log.info('  Done. %d/%d succeeded  (%d failed)',
         NUM_SAMPLES - n_failed, NUM_SAMPLES, n_failed)
log.info('  Step 4 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 5 — Blast geometry in RTN frame
###############################################################################
log.info('-'*70)
log.info('STEP 5 — Defining blast geometry in RTN frame')

R_hat, T_hat, N_hat = rtn_frame(r_nom, v_nom)
blast_pos = r_nom - (r_burst_nom_m / AU_M) * T_hat

log.info('  T_hat (RTN)  : %s', T_hat)
log.info('  Blast point  : %s AU  (-T direction)', blast_pos)
log.info('  Burst dist   : %.1f m  (= %.0f m surface + %.0f m standoff)',
         r_burst_nom_m, R_AST_M, H_STANDOFF_M)

###############################################################################
# STEP 6 — Sample all uncertainty sources  +  compute per-clone ΔV
###############################################################################
log.info('-'*70)
log.info('STEP 6 — Sampling uncertainty sources 1–8 and computing per-clone ΔV')
t0_step = time.perf_counter()

J_per_MT = 4.184e15

# Source 2 — Mass
mass_mu, mass_sig = lognormal_from_percentiles(3.30e9, 4.58e9)
np.random.seed(RANDOM_SEED + 1)
m_samples = np.exp(np.random.normal(mass_mu, mass_sig, NUM_SAMPLES))

# Source 3 — Diameter → radius
diam_mu    = (148.0 + 152.0) / 2.0
diam_sigma = (152.0 - 148.0) / (2.0 * 0.6745)
np.random.seed(RANDOM_SEED + 2)
diam_samples = np.random.normal(diam_mu, diam_sigma, NUM_SAMPLES)
R_samples    = np.clip(diam_samples / 2.0, 70.0, 80.0)

# Source 4 — Yield
np.random.seed(RANDOM_SEED + 3)
Y_samples = np.clip(
    np.random.normal(Y_MT, 0.09 * Y_MT, NUM_SAMPLES), 0.9, None)

# Source 5 — Standoff height
np.random.seed(RANDOM_SEED + 4)
H_samples = np.clip(
    np.random.normal(H_STANDOFF_M, 0.05 * H_STANDOFF_M, NUM_SAMPLES), 1000.0, 1400.0)

# Source 6 — Coupling efficiency η
np.random.seed(RANDOM_SEED + 5)
eta_samples = np.clip(
    np.random.normal(ETA_ABLATION, 0.3 * ETA_ABLATION, NUM_SAMPLES),
    0.02, 0.10)

# Source 7 — Ejecta velocity
np.random.seed(RANDOM_SEED + 6)
vej_samples = np.clip(
    np.random.normal(V_EJ_MS, 0.30 * V_EJ_MS, NUM_SAMPLES),
    1000.0, 3000.0)

# Source 8 — Shape: bootstrap resample from CSV
shape_rng = np.random.default_rng(RANDOM_SEED + 7)
shape_idx = shape_rng.integers(0, N_CSV, size=NUM_SAMPLES)
b_samples = csv_b[shape_idx]
c_samples = csv_c[shape_idx]

dv_nom_ms = ablation_dv_ms(Y_MT * J_per_MT, MASS_AST, V_EJ_MS,
                            f_geo_nom, eta=ETA_ABLATION)
log.info('  Nominal ΔV (solid-angle f_geo=%.6f) : %.5f m/s  (%.4f mm/s)',
         f_geo_nom, dv_nom_ms, dv_nom_ms * 1e3)
log.info('  Comparison — legacy sphere f_geo=%.6f would give : %.5f m/s',
         f_geo_sph,
         ablation_dv_ms(Y_MT * J_per_MT, MASS_AST, V_EJ_MS, f_geo_sph))

r_burst_all = R_samples + H_samples
omega_all   = np.pi * b_samples * c_samples / r_burst_all**2
f_geo_all   = omega_all / (4.0 * np.pi)

log.info('  Sampled distributions (mean ± std):')
for lbl, arr in [
    ('Mass [kg]',    m_samples),
    ('Diameter [m]', diam_samples),
    ('Yield [MT]',   Y_samples),
    ('Standoff [m]', H_samples),
    ('Eta',          eta_samples),
    ('v_ej [m/s]',   vej_samples),
    ('b [m]',        b_samples),
    ('c [m]',        c_samples),
    ('r_burst [m]',  r_burst_all),
    ('Ω [sr]',       omega_all),
    ('f_geo',        f_geo_all),
]:
    log.info('    %-16s: %.4g ± %.4g  (%.1f%%)',
             lbl, arr.mean(), arr.std(), 100*arr.std()/arr.mean())

# ── Per-clone ΔV loop ─────────────────────────────────────────────────────────
dv_magnitudes_ms  = []
dv_vectors_au_day = []
f_geo_record      = []

for i, state in enumerate(clone_states_at_blast):
    if state is None:
        dv_magnitudes_ms.append(0.0)
        dv_vectors_au_day.append(np.zeros(3))
        f_geo_record.append(np.nan)
        continue

    r_clone = np.array(state[:3])
    v_clone = np.array(state[3:])
    _, T_hat_i, _ = rtn_frame(r_clone, v_clone)

    m_i   = m_samples[i]
    Y_i   = Y_samples[i] * J_per_MT
    eta_i = eta_samples[i]
    vej_i = vej_samples[i]
    R_i   = R_samples[i]
    H_i   = H_samples[i]

    r_burst_i = R_i + H_i
    omega_i   = np.pi * b_samples[i] * c_samples[i] / r_burst_i**2
    f_geo_i   = omega_i / (4.0 * np.pi)

    dv_ms_i     = ablation_dv_ms(Y_i, m_i, vej_i, f_geo_i, eta=eta_i)
    dv_au_day_i = (dv_ms_i / 1e3) / (AU_KM / DAY_S) * T_hat_i

    dv_magnitudes_ms.append(dv_ms_i)
    dv_vectors_au_day.append(dv_au_day_i)
    f_geo_record.append(f_geo_i)

dv_magnitudes_ms = np.array(dv_magnitudes_ms)
f_geo_record     = np.array(f_geo_record)
valid            = dv_magnitudes_ms > 0

log.info('  ΔV stats across clones (N_valid=%d):', valid.sum())
log.info('    mean   : %.4f mm/s', dv_magnitudes_ms[valid].mean() * 1e3)
log.info('    std    : %.4f mm/s  (%.1f%%)',
         dv_magnitudes_ms[valid].std() * 1e3,
         100*dv_magnitudes_ms[valid].std()/dv_magnitudes_ms[valid].mean())
log.info('    5th %%  : %.4f mm/s', np.percentile(dv_magnitudes_ms[valid], 5) * 1e3)
log.info('    95th %% : %.4f mm/s', np.percentile(dv_magnitudes_ms[valid], 95) * 1e3)
log.info('    nominal: %.4f mm/s', dv_nom_ms * 1e3)
log.info('  Step 6 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 7 — Build deflected / undeflected clone initial conditions
###############################################################################
log.info('-'*70)
log.info('STEP 7 — Applying ΔV to each clone velocity at t_blast')
t0_step = time.perf_counter()

deflected_samples   = []
undeflected_samples = []

for i in range(NUM_SAMPLES):
    state = clone_states_at_blast[i] \
            if clone_states_at_blast[i] is not None else nom_state
    dv    = dv_vectors_au_day[i]
    base  = {'t': t_blast_mjd,
             'x': state[0], 'y': state[1], 'z': state[2]}
    deflected_samples.append({**base,
        'vx': state[3] + dv[0],
        'vy': state[4] + dv[1],
        'vz': state[5] + dv[2]})
    undeflected_samples.append({**base,
        'vx': state[3], 'vy': state[4], 'vz': state[5]})

nom_sol_blast = {
    't':  t_blast_mjd,
    'x':  nom_state[0], 'y':  nom_state[1], 'z':  nom_state[2],
    'vx': nom_state[3], 'vy': nom_state[4], 'vz': nom_state[5],
}
log.info('  Built %d deflected and %d undeflected initial conditions',
         len(deflected_samples), len(undeflected_samples))
log.info('  Step 7 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 8 — Propagate both clouds to EXTENDED end epoch
###############################################################################
log.info('-'*70)
log.info('STEP 8 — Propagating both clouds to extended epoch')
log.info('  t_end_extended = MJD %.0f  (+%d days, ~%.0f yr past nominal impact)',
         t_end_extended_mjd, T_END_EXTENDED_DAYS, T_END_EXTENDED_DAYS/365.25)
t0_step = time.perf_counter()

ng_blast = make_ng_params(nom_sol_blast)

def make_impact_sim(run_label):
    sim = prop.PropSimulation(
        name=f'{BODY_ID}_{run_label}',
        t0=t_blast_mjd, defaultSpiceBodies=440,
        DEkernelPath=utils.default_kernel_path)
    sim.set_integration_parameters(t_end_extended_mjd)
    for body in sim.spiceBodies:
        if 'Earth' in body.name or body.spiceId == 399:
            body.caTol = 0.1
    return sim

log.info('  Propagating UNDEFLECTED clones ...')
cloud_undef, result_undef = propagate_to_epoch(
    nom_sol_blast, ng_blast, make_impact_sim('undeflected_1'),
    undeflected_samples, NUM_THREADS, t_end_extended_mjd)

log.info('  Propagating DEFLECTED clones ...')
cloud_def, result_def = propagate_to_epoch(
    nom_sol_blast, ng_blast, make_impact_sim('deflected_1'),
    deflected_samples, NUM_THREADS, t_end_extended_mjd)

ca_u   = sum(1 for x in result_undef[0] if x)
imp_u  = sum(1 for x in result_undef[1] if x)
ca_d   = sum(1 for x in result_def[0]   if x)
imp_d  = sum(1 for x in result_def[1]   if x)
none_u = sum(1 for i in range(len(result_undef[0]))
             if not result_undef[0][i] and not result_undef[1][i])
none_d = sum(1 for i in range(len(result_def[0]))
             if not result_def[0][i] and not result_def[1][i])
log.info('  Undeflected — CA: %d  impacts: %d  neither: %d', ca_u, imp_u, none_u)
log.info('  Deflected   — CA: %d  impacts: %d  neither: %d', ca_d, imp_d, none_d)
log.info('  Step 8 completed in %.2f s', time.perf_counter() - t0_step)

###############################################################################
# STEP 9 — Save, plot, and summary
###############################################################################
log.info('-'*70)
log.info('STEP 9 — Saving outputs and generating plots')
t0_step = time.perf_counter()

def filter_cloud(cloud):
    return [c for c in cloud if c is not None]

with open(OUT_UNDEF, 'w') as f:
    json.dump(filter_cloud(cloud_undef), f, indent=2)
with open(OUT_DEF, 'w') as f:
    json.dump(filter_cloud(cloud_def), f, indent=2)

n_undef_found = len(filter_cloud(cloud_undef))
n_def_found   = len(filter_cloud(cloud_def))
log.info('  Saved: %s  (%d entries)', OUT_UNDEF, n_undef_found)
log.info('  Saved: %s  (%d entries)', OUT_DEF, n_def_found)

R_E_AU  = 4.2635e-5
xi_u_re = np.array([c['xi']   for c in filter_cloud(cloud_undef)]) / R_E_AU
ze_u_re = np.array([c['zeta'] for c in filter_cloud(cloud_undef)]) / R_E_AU
xi_d_re = np.array([c['xi']   for c in filter_cloud(cloud_def)])   / R_E_AU
ze_d_re = np.array([c['zeta'] for c in filter_cloud(cloud_def)])   / R_E_AU

all_xi = np.concatenate([xi_u_re, xi_d_re])
all_ze = np.concatenate([ze_u_re, ze_d_re])
xi_c, ze_c = np.nanmean(all_xi), np.nanmean(all_ze)
half = max(np.nanmax(all_xi) - np.nanmin(all_xi),
           np.nanmax(all_ze) - np.nanmin(all_ze), 2.0) * 0.7
xlim = (xi_c - half, xi_c + half)
ylim = (ze_c - half, ze_c + half)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#1a1a2e')

ax = axes[0]
ax.set_facecolor('#1a1a2e')
ax.add_patch(plt.Circle((0, 0), 1.0, color='deepskyblue', zorder=5,
                         label='Earth (1 R⊕)'))
ax.scatter(xi_u_re, ze_u_re, s=6, alpha=0.45, color='tomato',
           label=f'Undeflected (N={len(xi_u_re)})', zorder=3)
ax.scatter(xi_d_re, ze_d_re, s=6, alpha=0.45, color='limegreen',
           label=f'Deflected (N={len(xi_d_re)})', zorder=4)
ax.scatter([np.nanmean(xi_u_re)], [np.nanmean(ze_u_re)],
           s=120, color='red',  marker='*', zorder=6, label='Nominal (undefl.)')
ax.scatter([np.nanmean(xi_d_re)], [np.nanmean(ze_d_re)],
           s=120, color='lime', marker='*', zorder=7, label='Nominal (defl.)')
ax.annotate('',
    xy=(np.nanmean(xi_d_re), np.nanmean(ze_d_re)),
    xytext=(np.nanmean(xi_u_re), np.nanmean(ze_u_re)),
    arrowprops=dict(arrowstyle='->', color='white', lw=2.0))
shift_re = np.sqrt((np.nanmean(xi_d_re) - np.nanmean(xi_u_re))**2 +
                   (np.nanmean(ze_d_re) - np.nanmean(ze_u_re))**2)
ax.text(np.nanmean(xi_u_re) + 0.05*half,
        0.5*(np.nanmean(ze_u_re) + np.nanmean(ze_d_re)),
        f'Δb = {shift_re:.3f} R⊕', color='white', fontsize=9, va='center')
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_xlabel('ξ  [R⊕]', fontsize=12, color='white')
ax.set_ylabel('ζ  [R⊕]', fontsize=12, color='white')
ax.set_title(
    f'B-plane: 2024 PDC25  (worst Earth CA, +{T_END_EXTENDED_DAYS}d window)\n'
    f'{Y_MT} MT standoff @ {H_STANDOFF_M:.0f} m | Blast: {T_BLAST_ISO}',
    fontsize=10, color='white')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('white')
ax.grid(True, alpha=0.2, color='white')
ax.legend(loc='upper right', fontsize=9, facecolor='#2a2a4e', labelcolor='white')

ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
valid_dv = dv_magnitudes_ms[valid] * 1e3
ax2.hist(valid_dv, bins=60, color='limegreen', alpha=0.7, edgecolor='none')
ax2.axvline(np.mean(valid_dv), color='white', lw=1.5, ls='--',
            label=f'Mean = {np.mean(valid_dv):.4f} mm/s')
ax2.axvline(dv_nom_ms * 1e3, color='gold', lw=1.5, ls=':',
            label=f'Nominal = {dv_nom_ms*1e3:.4f} mm/s')
ax2.axvline(np.percentile(valid_dv, 5),  color='tomato', lw=1, ls='--',
            label=f'5th pct = {np.percentile(valid_dv,5):.4f} mm/s')
ax2.axvline(np.percentile(valid_dv, 95), color='tomato', lw=1, ls='--',
            label=f'95th pct = {np.percentile(valid_dv,95):.4f} mm/s')
ax2.set_xlabel('ΔV  [mm/s]', fontsize=11, color='white')
ax2.set_ylabel('Count', fontsize=11, color='white')
ax2.set_title(
    f'ΔV Distribution — 8 uncertainty sources\n'
    f'f_geo from CSV shape: mean={f_geo_all.mean():.5f}  '
    f'std={f_geo_all.std():.5f}  ({100*f_geo_all.std()/f_geo_all.mean():.1f}%)',
    fontsize=9, color='white')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_edgecolor('white')
ax2.grid(True, alpha=0.2, color='white')
ax2.legend(fontsize=9, facecolor='#2a2a4e', labelcolor='white')

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
log.info('  Plot saved → %s', PLOT_FILE)
plt.show()

log.info('  Step 9 completed in %.2f s', time.perf_counter() - t0_step)

# ── Summary ───────────────────────────────────────────────────────────────────
n_impact_u = np.sum(np.sqrt(xi_u_re**2 + ze_u_re**2) < 1.0)
n_impact_d = np.sum(np.sqrt(xi_d_re**2 + ze_d_re**2) < 1.0)

log.info('')
log.info('='*70)
log.info('DEFLECTION SUMMARY')
log.info('='*70)
log.info('  Blast date             : %s', T_BLAST_ISO)
log.info('  Lead time              : %.0f days  (%.2f yr)',
         t_end_mjd - t_blast_mjd, (t_end_mjd - t_blast_mjd)/365.25)
log.info('  Propagation window     : +%d days past impact  (~%.0f yr)',
         T_END_EXTENDED_DAYS, T_END_EXTENDED_DAYS/365.25)
log.info('  Nominal yield          : %s MT', Y_MT)
log.info('  Nominal standoff       : %.0f m', H_STANDOFF_M)
log.info('  Nominal η_ablation     : %s  (S-type silicaceous)', ETA_ABLATION)
log.info('  Nominal v_ejecta       : %.0f m/s', V_EJ_MS)
log.info('  ── Solid Angle ─────────────────────────────────────────────────')
log.info('  Shape source           : %s  (%d rows)', PHYS_CSV, N_CSV)
log.info('  Mean semi-axes         : a=%.1f m  b=%.1f m  c=%.1f m', a_nom, b_nom, c_nom)
log.info('  Ω (nominal shape)      : %.5f sr', omega_nom)
log.info('  f_geo (solid angle)    : %.6f', f_geo_nom)
log.info('  f_geo (sphere, legacy) : %.6f  (%.2f× larger)', f_geo_sph, f_geo_nom/f_geo_sph)
log.info('  f_geo MC (all clones)  : %.6f ± %.6f  (%.1f%%)',
         f_geo_all.mean(), f_geo_all.std(),
         100*f_geo_all.std()/f_geo_all.mean())
log.info('  ── ΔV ──────────────────────────────────────────────────────────')
log.info('  Nominal ΔV             : %.4f mm/s', dv_nom_ms * 1e3)
log.info('  Mean ΔV  (MC)          : %.4f mm/s', dv_magnitudes_ms[valid].mean() * 1e3)
log.info('  Std  ΔV  (MC)          : %.4f mm/s  (%.1f%%)',
         dv_magnitudes_ms[valid].std() * 1e3,
         100*dv_magnitudes_ms[valid].std()/dv_magnitudes_ms[valid].mean())
log.info('  90%% CI ΔV              : [%.4f, %.4f] mm/s',
         np.percentile(dv_magnitudes_ms[valid], 5) * 1e3,
         np.percentile(dv_magnitudes_ms[valid], 95) * 1e3)
log.info('  ── B-plane ─────────────────────────────────────────────────────')
log.info('  B-plane found          : undefl. %d/%d  defl. %d/%d',
         n_undef_found, NUM_SAMPLES, n_def_found, NUM_SAMPLES)
log.info('  Δb (centroid shift)    : %.4f R⊕', shift_re)
log.info('  Clones ≤ 1R⊕ (undefl.) : %d/%d  (%.1f%%)',
         n_impact_u, len(xi_u_re), 100*n_impact_u/max(len(xi_u_re), 1))
log.info('  Clones ≤ 1R⊕ (defl.)   : %d/%d  (%.1f%%)',
         n_impact_d, len(xi_d_re), 100*n_impact_d/max(len(xi_d_re), 1))
log.info('='*70)

###############################################################################
# COMPLETE EARTH ENCOUNTER SUMMARY — all clones, all years
###############################################################################
R_e_km = 6371.0

def collect_all_earth_events(result):
    """
    For each clone, collect every unique Earth CA and impact event.
    Deduplicates by time (round to 2 decimal places in MJD).
    Returns list of dicts: clone index, year, b [R⊕], impact flag.
    """
    events = []
    for i in range(len(result[0])):
        all_e   = list(result[1][i]) + list(result[0][i])
        earth_e = [e for e in all_e
                   if 'Earth' in str(getattr(e, 'centralBody', ''))
                   or getattr(e, 'centralBodySpiceId', 0) == 399]
        seen_times = set()
        for e in earth_e:
            t_key = round(e.t, 2)
            if t_key not in seen_times:
                seen_times.add(t_key)
                events.append({
                    'clone' : i,
                    'year'  : Time(e.t, format='mjd').decimalyear,
                    'b_re'  : e.bMag * AU_KM / R_e_km,
                    'impact': getattr(e, 'impact', False),
                })
    return events

log.info('')
log.info('='*78)
log.info('ALL CLONES — COMPLETE EARTH ENCOUNTER SUMMARY')
log.info('='*78)

for label, result in [('UNDEFLECTED', result_undef),
                       ('DEFLECTED',   result_def)]:
    events = collect_all_earth_events(result)

    if not events:
        log.info('  %s: no Earth events found', label)
        continue

    impacts = [e for e in events if e['impact']]
    misses  = [e for e in events if not e['impact']]

    year_bins = defaultdict(list)
    for e in events:
        year_bins[int(e['year'])].append(e)

    log.info('  %s:', label)
    log.info('  Total Earth events (all clones) : %d', len(events))
    log.info('  Clones with ≥1 impact           : %d/%d',
             len(set(e['clone'] for e in impacts)), NUM_SAMPLES)
    log.info('  Clones with only misses         : %d/%d',
             len(set(e['clone'] for e in misses) - set(e['clone'] for e in impacts)),
             NUM_SAMPLES)
    log.info('  Encounter breakdown by year:')
    log.info('  %6s  %10s  %8s  %15s  %14s  %14s',
             'Year', 'N clones', 'Impacts', 'Mean |b| [R⊕]',
             'Min |b| [R⊕]', 'Max |b| [R⊕]')
    log.info('  %s', '-'*73)
    for key in sorted(year_bins.keys()):
        bucket = year_bins[key]
        n      = len(bucket)
        n_imp  = sum(1 for e in bucket if e['impact'])
        b_vals = [e['b_re'] for e in bucket]
        log.info('  %6d  %10d  %8d  %15.3f  %14.3f  %14.3f',
                 key, n, n_imp, np.mean(b_vals), np.min(b_vals), np.max(b_vals))

log.info('='*78)

total_elapsed = time.perf_counter() - RUN_START
log.info('Total pipeline wall time: %.1f s', total_elapsed)
log.info('Log written to: %s', os.path.abspath(LOG_FILE))
