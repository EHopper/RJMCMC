# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:27:30 2018

@author: emily
"""

# This code is largely based on...
# B&c12:    Bodin et al., 2012, JGR, doi:10.1029/2011JB008560
# KL14:     Kolb and Lekic, 2014, GJI, doi:10.1093/gji/ggu079
# GA:       Geoff Abers' code given to me by Helen Janizewski


#import collections
import typing
import numpy as np
import random
from scipy.interpolate import interp1d
import matlab


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class RecvFunc(typing.NamedTuple):
    amp: np.array  # assumed to be processed in same way as synthetics are here
    std: np.array  # same length as amp, comes from bootstrapping
    dt: float      # constant timestep in s
    rf_phase: str
    ray_param: float  # tested assuming ray_param = 0.0618
    filter_corners: list # Should be the short and long PERIOD
                                  # corners of the appropriate filter band (s)
    std_sc: float # This is trying to account for near-surface misrotation
                            # scale std_rf by std_sc for the first 0.5s of the signal
                            # i.e. set to 1 to not bodge things!
    weight_by: str # set to 'even' to weight RF and SWD misfit evenly
                    #    to 'rf2'  to weight RF 2x as much as SWD misfit
# Receiver function assumed to start at t=0s with a constant timestep, dt
# required to be stretched to constant RP (equivalent to distance of 60 degrees)
# This formulation should allow multiple input RFs

class BodyWaveform(typing.NamedTuple):
    amp_R: np.array # horizontal (radial) energy
    amp_Z: np.array # vertical energy
    dt: float

class RotBodyWaveform(typing.NamedTuple):
    parent: np.array # P energy for Ps, SV energy for Sp
    daughter: np.array # SV energy for Ps, P energy for Sp
    dt: float

class SurfaceWaveDisp(typing.NamedTuple):
    period: np.array    # dominant period of dispersion measurement, seconds
    c: np.array  # phase velocity in km/s
# Surface wave dispersion measurements are allowed at any series of periods (s)

class Model(typing.NamedTuple):
    vs: np.array
    all_deps: np.array    # all possible depth nodes
    idep: np.array        # indices of used depth nodes
    std_rf: np.array      #  length == number of input RFs
    lam_rf: np.array      #  length == number of input RFs
    std_swd: float
# Inputs: dep=[list of depths, km]; vs=[list of Vs, km/s];
# std_rf=RF standard deviation; lam_rf=RF lambda; std_swd=SWD dispersion std
# Note that std_rf, lam_rf, std_swd are all inputs to the covariance matrix,
# i.e. allow greater wiggle room in the misfit - [B&c12]
# I'm also including a parameter all_deps which defines the depth grid

class Limits(typing.NamedTuple):
    vs: tuple
    dep: tuple
    std_rf: tuple
    lam_rf: tuple
    std_swd: tuple
    crustal_thick: tuple
# Reasonable min and max values for all model parameters define the prior
# distribution - i.e. uniform probability within some reasonable range of values
# These limits are on the output vel model, so will not depend on inputs

class CovarianceMatrix(typing.NamedTuple):
    Covar: np.array
    invCovar: np.array
    detCovar: float
    R: np.array

class ModelChange(typing.NamedTuple):
    theta: float
    old_param: float
    new_param: float
    which_change: str

class SynthModel(typing.NamedTuple):
    vs: np.array
    vp: np.array
    rho: np.array
    thickness: np.array
    layertops: np.array
    avdep: np.array

class GlobalState(typing.NamedTuple):
    model: Model
    fullmodel: SynthModel
    cov: CovarianceMatrix
    rf_synth: RecvFunc
    swd_synth: SurfaceWaveDisp
    misfit_to_obs: float

class Error(Exception): pass

# =============================================================================
# Overall process for the joint inversion
# =============================================================================

def JointInversion(rf_obs: list, swd_obs: SurfaceWaveDisp, lims: Limits,
                   max_iter: int, random_seed: int,
                   save_name:  str) -> Model:

    # N.B. Input more than one RF by making a list of RecvFunc

    savename = open('failed_prior.txt', mode = 'w')

    # N.B.  The variable random_seed ensures that the process is repeatable.
    random.seed(a = random_seed)
    np.random.seed(seed = random_seed+1)

    state = InitialState(rf_obs, swd_obs, lims)

# =============================================================================
#           Define parameters for convergence
# =============================================================================

    ddeps=np.zeros(state.model.all_deps.size*2-1)
    ddeps[::2] = state.model.all_deps
    ddeps[1::2] = state.model.all_deps[:-1]+np.diff(state.model.all_deps)/2
    save_every = 100
    all_models = np.zeros((ddeps.size,int((max_iter)/save_every)+2))
    all_models[:,0] = ddeps
    all_models[:,1] = SaveModel(state.fullmodel, ddeps)
    all_hyperparameters = np.zeros((int(max_iter/save_every)+1,
                                    1+2*len(rf_obs)))
    all_phi = np.zeros(save_every) # all misfits
    all_alpha = np.zeros(save_every) # all likelihoods
    all_keep = np.zeros(save_every)
    mean_phi = np.zeros(int(max_iter/save_every)+1)
    mean_alpha = np.zeros(int(max_iter/save_every)+1)
    mean_keep = np.zeros(int(max_iter/save_every)+1)

    n_save = 0
    n_mwin = 0

    # =========================================================================
    #       Iterate by Reverse Jump Markov Chain Monte Carlo
    # =========================================================================
    for itr in range(1,max_iter):
        if not itr % 20:
            print("Iteration {}..".format(itr))

        state, keep_yn, all_alpha[n_mwin] = NextState(
            itr, rf_obs, swd_obs, lims, state, savename)

        all_phi[n_mwin] = state.misfit_to_obs
        all_keep[n_mwin] = keep_yn

        # Save every XXXth (save_every) iteration to see progress
        n_mwin += 1

        if itr / save_every > n_save:
            mean_phi[n_save] = np.mean(all_phi[all_phi>0])
            mean_alpha[n_save] = np.mean(all_alpha[all_phi>0])
            mean_keep[n_save] = np.mean(all_keep[all_phi>0])
            all_phi *= 0; all_alpha *= 0; all_keep *= 0
            all_hyperparameters[n_save,:] = np.hstack([state.model.std_rf,
                               state.model.lam_rf, np.array([state.model.std_swd])])
            all_models[:,n_save+1] = SaveModel(state.fullmodel, all_models[:,0])
            np.save((save_name+'_AllModels'),all_models[:,:n_save+1])
            np.save((save_name+'_Misfit'),np.vstack((mean_phi[:n_save],
                    mean_alpha[:n_save], mean_keep[:n_save])))
            np.save((save_name+'_Hyperparams'),all_hyperparameters)
            n_save += 1
            n_mwin = 0

    return (None, all_models, all_phi, all_alpha, all_keep, state.fullmodel)


def InitialState(rf_obs: list, swd_obs: SurfaceWaveDisp,
                 lims: Limits) -> GlobalState:
    # =========================================================================
    #      Start by calculating for some (arbitrary) initial model
    # =========================================================================
    model_0 = InitialModel(rf_obs)
    if not CheckPrior(model_0, lims):
        raise Error("Starting model does not fit prior distribution")

    # Calculate synthetic data
        # First, need to define the Vp and rho
    fullmodel_0 = MakeFullModel(model_0)
    rf_synth_m0 = SynthesiseRF(fullmodel_0, rf_obs) # forward model the RF data
    swd_synth_m0 = SynthesiseSWD(fullmodel_0, swd_obs.period, 0) # forward model the SWD data

    # Calculate covariance matrix (and inverse, and determinant)
    cov_m0 = CalcCovarianceMatrix(model_0,rf_synth_m0,swd_synth_m0)

    # Calculate fit
    misfit_to_obs_m0 = Mahalanobis(
            rf_obs, rf_synth_m0, # RecvFunc
            swd_obs, swd_synth_m0,  # SurfaceWaveDisp
            cov_m0.invCovar,             # CovarianceMatrix
            )
    return GlobalState(
        model=model_0,
        cov=cov_m0,
        fullmodel=fullmodel_0,
        rf_synth=rf_synth_m0,
        swd_synth=swd_synth_m0,
        misfit_to_obs=misfit_to_obs_m0,
    )


def NextState(itr:int, rf_obs: list, swd_obs: SurfaceWaveDisp, lims: Limits,
              prev_state: GlobalState, savename) -> (GlobalState, bool, float):
    # Generate new model by perturbing the old model
    model, changes_model = Mutate(prev_state.model, itr)

    if not CheckPrior(model, lims): # check if mutated model compatible with prior distr.
#        print('Failed Prior (', itr, '): ', changes_model.which_change,
#              ' change to ', changes_model.new_param, ' (from ',
#              changes_model.old_param, ')', file = savename)
        #print('Failed Prior')
        return prev_state, False, 0  # if not, continue to next iteration


    fullmodel = MakeFullModel(model)
    if changes_model.which_change == 'Noise':
        # only need to recalculate covariance matrix if changed hyperparameter
        rf_synth_m = prev_state.rf_synth
        swd_synth_m = prev_state.swd_synth
        cov_m = CalcCovarianceMatrix(model,prev_state.rf_synth,prev_state.swd_synth)

    else:
        #print(changes_model.which_change)
        cov_m = prev_state.cov
        rf_synth_m = SynthesiseRF(fullmodel, prev_state.rf_synth) # forward model the RF data
        swd_synth_m = SynthesiseSWD(fullmodel, swd_obs.period, itr) # forward model the SWD data

    # Calculate fit
    misfit_to_obs_m = Mahalanobis(
        rf_obs, rf_synth_m, # RecvFunc
        swd_obs, swd_synth_m,  # SurfaceWaveDisp
        cov_m.invCovar,             # CovarianceMatrix
    )

    # Calculate probability of keeping mutation
    keep_yn, alpha = AcceptFromLikelihood(
        misfit_to_obs_m, prev_state.misfit_to_obs, # float
        cov_m, prev_state.cov, # CovarianceMatrix
        changes_model, # form depends on which change
    )

    if not keep_yn:
        return prev_state, False, alpha

    return GlobalState(
        model=model,
        cov=cov_m,
        fullmodel=fullmodel,
        rf_synth=rf_synth_m,
        swd_synth=swd_synth_m,
        misfit_to_obs=misfit_to_obs_m,
    ), True, alpha


# =============================================================================
#       Setting up the model.
# =============================================================================
# Our model variables consist of (1) Depth of nuclei; (2) Vs of nuclei;
# and two hyperparameters that define the noise of the RFs [std_rf, lam_rf];
# and one hyperparameter defining the noise of the SWD [std_SWD]
# Note that RF noise is assumed to be correlated in some way, dependent on the
# lag time between two samples (according to an expression involving lambda)
# but SWD noise is assume to be uncorrelated, so only one hyperparameter is
# inverted for the dispersion data.
def InitialModel(rf_obs: list) -> Model:
    vs = np.array([round(random.uniform(0.5,4.5),2)])   # arbitrary
    all_deps = np.concatenate((np.arange(0,10,0.2),
                                np.arange(10,60,1), np.arange(60,201,5)))
    idep = np.array([random.randrange(0,len(all_deps))])  # arbitrary
    std_rf = np.array([round(np.std(rf_obs[0].amp,ddof=1),2)]) # start off assuming all data is noise (KL14)
    for irf in range(1, len(rf_obs)):
        std_rf = np.hstack((std_rf, round(np.std(rf_obs[irf].amp,ddof=1),2)))
    lam_rf = 0.2 * np.ones(len(rf_obs),) # this is as the example given by KL14
    std_swd = 0.05 # arbitrary - the max allowed in Geoff Abers' code = 0.15
    return Model(vs, all_deps, idep, std_rf, lam_rf, std_swd)


# =============================================================================
#       Check against prior distribution (check reasonable limits)
# =============================================================================
#     For simplicity, we assume a uniform probability distribution between two
#     limits for all the parameters
def CheckPrior(model: Model, limits: Limits) -> bool:
    return (
        _InBounds(model.idep, (0,model.all_deps.size-2)) and
        _InBounds(model.vs, limits.vs) and
        _InBounds(model.all_deps[model.idep], limits.dep) and
        _InBounds(model.std_rf, limits.std_rf) and
        _InBounds(model.lam_rf, limits.lam_rf) and
        _InBounds(model.std_swd, limits.std_swd) and
        model.vs.size == model.idep.size and
        (np.hstack([0,model.vs[model.all_deps[model.idep]<=
                               limits.crustal_thick[0]]])  <= 4.5).all()
    )

def _InBounds(value, limit):
    min_v, max_v = limit
    if type(value) == np.ndarray: # and value.size > 1:
        return all(np.logical_and(min_v <= value, value <= max_v))
    else:
        return min_v <= value <= max_v

# =============================================================================
#       Calculate Rest of model required for synthetics
# =============================================================================

def MakeFullModel(model) -> SynthModel:

    # Assuming waveforms are incident from 60 degrees for default value of p
    # For P waves, this means horizontal slowness = 0.0618
    # For incident S waves, this means p = 0.1157 (using AK135)
    # phase_vel = 1/horizontal_slowness


    # Define Vp, Rho, Thickness
    dep = model.all_deps[model.idep]
    vp = np.zeros_like(model.vs) # in km/s
    rho = np.zeros_like(model.vs) # in Mg/m^3 (or g/cm^3)


    #  Now calculate the thickness and average depth of the layers
    layertops = np.concatenate([[0],(dep[1:]+dep[0:-1])/2])
    if layertops.size == 1:
        thickness = dep # arbitrary
    else:
        thickness = np.diff(layertops)
        thickness = np.concatenate([thickness, [thickness[-1]]])
    avdep = layertops + thickness/2

    #   1.  Assume that any values of Vs under 4.5 km/s are crustal
    Moho_ind, = np.where(4.5 <= model.vs)
    if Moho_ind.size:  # if this array isn't empty
        Moho_ind = Moho_ind[0]
    else:
        Moho_ind = model.vs.size

    crust_inds = np.arange(Moho_ind)

    # Equations from Brocher et al., 2005, BSSA (doi: 10.1785/0120050077)
    vp[crust_inds] = (
                0.9409+2.0947*model.vs[crust_inds] -
                0.8206*model.vs[crust_inds]**2 +
                0.2683*model.vs[crust_inds]**3 - 0.0251*model.vs[crust_inds]**4
                )
    rho[crust_inds] = (
                1.6612*vp[crust_inds] - 0.4721*vp[crust_inds]**2 +
                0.0671*vp[crust_inds]**3 - 0.0043*vp[crust_inds]**4 +
                0.000106*vp[crust_inds]**5
                )

    #     2.  And any value deeper than the shallowest Vs>4.5 (Moho) is mantle
    if Moho_ind < model.vs.size:
        vp[Moho_ind:] = model.vs[Moho_ind:]*1.75

        # Density scaling comes from Forte et al., 2007, GRL (doi:10.1029/2006GL027895)
        #   This relates d(ln(rho))/d(ln(vs)) to z and comes from fitting
        #   linear/quadratics to values picked from their Fig. 1b.
        #   We will assume an oceanic setting (suitable for orogenic zones too)
        #   although they give a different scaling for shields if necessary
        rho_scaling = np.zeros(model.vs.size)
        del_rho = np.zeros(model.vs.size)
        i_uppermost_mantle = np.logical_and(avdep[Moho_ind] <= avdep, avdep < 135)
        i_upper_mantle = np.logical_and(135 <= avdep, avdep <= 300)
        rho_scaling[i_uppermost_mantle] = 2.4e-4*avdep[i_uppermost_mantle] + 5.6e-2
        rho_scaling[i_upper_mantle] = (
                2.21e-6*avdep[i_upper_mantle]**2 - 6.6e-4*avdep[i_upper_mantle] + 0.075
                )
        # for shields, would be...
        #    rho_scaling[Moho_ind:] = 3.3e-4*avdep[Moho_ind:] - 3.3e-2

        # rho_scaling is d(ln(rho))/d(ln(vs)), i.e. d(rho)/rho / d(vs)/vs
        # We'll take as our reference value the 120km value of AK135
        ref_vs_rho = (4.5, 3.4268)
        del_rho[Moho_ind:] = (
                rho_scaling[Moho_ind:]*  # d(ln(rho))/d(ln(vs))
                (model.vs[Moho_ind:]-ref_vs_rho[0])/ref_vs_rho[0]* # d(vs)/vs
                ref_vs_rho[1] # ref rho
                )
        rho[Moho_ind:]=ref_vs_rho[1]+del_rho[Moho_ind:]


    vp = np.round(vp, 3)
    rho = np.round(rho, 3)

    return SynthModel(model.vs, vp, rho, thickness, layertops, avdep)


# =============================================================================
#       Calculate the covariance matrix, its inverse and its determinant
# =============================================================================
# Here, we are assuming that noise is not correlated for the surface waves
# i.e. covariance matrix is diagonal, with every value equal to std_swd (B&c12)
# For receiver functions, noise IS correlated (as we are taking every point in
# time series as an adjacent data point).  There are many ways to represent
# this correlated noise, including C_ab = r^(abs(a-b)) (B&c12 method 1).  The
# advantage of this method is there is an analytic solution to the inverse and
# the determinant.  However, KL14 point out this does a poor job of actually
# representing the noise in the dataset.  Instead, they recommend...
    # R_ij = exp(-lambda*(i-j))*cos(lambda*w0*(i-j)); C_ij = std_rf^2 * R_ij
    # where lambda is inverted for (lam_rf), w0 is a constant (KL14 set to 4.4)
    # and (i-j) is actually dt*abs(i-j) i.e. the timelag between two samples
def CalcCovarianceMatrix(model,rf_obs,swd_obs) -> CovarianceMatrix:
    #    First, calculate R based on lam_rf
    # RF part of covariance: square matrix with length = number of timesteps in RF
    w0=4.4 # constant, as KL14
    n_rft = rf_obs[0].amp.size # Assume all RF inputs are same length
    n_data = swd_obs.period.size + len(rf_obs)*n_rft
    R = np.zeros((n_rft*len(rf_obs), n_rft*len(rf_obs)))
    covar = np.zeros((n_data, n_data))

    for irf in range(len(rf_obs)):
        i1 = n_rft*irf
        i2 = n_rft*(irf+1)
        for a in range(i1, i2):
            for b in range(i1, i2):
                R[a,b]=(np.exp(-(model.lam_rf[irf]*rf_obs[irf].dt*abs(a-b)))*
                 np.cos(model.lam_rf[irf]*w0*rf_obs[irf].dt*abs(a-b)))
        covar[i1:i2,i1:i2] = R[i1:i2, i1:i2] * model.std_rf[irf]**2

        # Try to workaround potential mis-rotation near surface by increasing the
        # std_rf near the surface, e.g. first 0.5 s
        if rf_obs[irf].rf_phase == 'Ps': ind = int(0.5/rf_obs[irf].dt) + 1  # <= 0.5 s
        if rf_obs[irf].rf_phase == 'Sp': ind = int(3/rf_obs[irf].dt) + 1 # <= 3s for Sp
        covar[i1:i1+ind,i1:i1+ind] *= rf_obs[irf].std_sc


    covar[-swd_obs.period.size:,-swd_obs.period.size:]=(
            np.identity(swd_obs.period.size)*model.std_swd**2)

    #print(model.lam_rf, model.std_rf, model.std_swd)
    invc=np.linalg.inv(covar)
    #  Note: Need to take the determinant for use in calculating the acceptance
    #        probability if changing a hyperparameter.  However, given that
    #        this is a 130x130 odd matrix with all values <1, the determinant
    #        is WAY too small.  However, we actually only care about the RATIO
    #        between the new and old determinant, so it is fine to just scale
    #        this up so that it doesn't get lost in the rounding error.
    #  detc=np.linalg.det(covar)

    if len(rf_obs) == 1: sc_by = 1e4
    if len(rf_obs) == 2: sc_by = 5e2
    detc = np.linalg.det(covar*sc_by)
    if detc>1e300: detc=1e300
    if detc<1e-300: detc=1e-300

    #detc = detc / (10**int(np.log10(detc)))

    return CovarianceMatrix(covar,invc,detc,R)



# =============================================================================
#       Perturb the old model
# =============================================================================
# To perturb the model, we randomly pick one of the model parameters and then
# we vary it according to a Gaussian probability distribution around the
# existing value.  The exact form varies according to the parameter we vary (B&c12)
# KL14 suggest downweighting the probability of changing lam_rf (or whichever
# parameters require recalculating the determinant/inverse of the covariance matrix)
# but it seems like any of the other parameters are going to require recalculating
# the synthetic data, so that will be just as bad.  Anyway, keep that in mind.
# KL14 also suggest limiting the number of possible nodes in your model at first
# to ensure the largest discontinuities are fit first.
def Mutate(model,itr) -> (Model, ModelChange): # and ModelChange
    # First, we choose which model parameter to perturb
    # There are model.vs.size + model.dep.size + 3 (std_rf, lam_rf, std_swd)
    # We can perturb by either (1) changing Vs, (2) changing nucleus depth,
    # (3) having a layer birth, (4) having a layer depth, or
    # (5) changing a hyperparameter.
    # Choose one uniformly - although...
    # KL14 suggest a limit of k Gaussians in the first 1000k(k+1) iterations
    # (where k Gaussians == k+1 nuclei here)
    if model.idep.size==1: # and don't kill your only layer...
        perturbs=('Vs','Dep','Birth','Hyperparameter')
    elif itr <= 750*(model.idep.size-1)*(model.idep.size) or model.idep.size >=15: # as KL14
        perturbs=('Vs','Dep','Death','Hyperparameter')
    else:
        perturbs=('Vs','Dep','Birth','Death','Hyperparameter')
    perturb = random.sample(perturbs, 1)[0]

    if perturb=='Vs':          # Change Vs in a layer
        i_vs = random.randint(0,model.vs.size-1)
        old = model.vs[i_vs]
        theta =_GetStdForGaussian(perturb, itr)
        new = np.round(np.random.normal(old,theta),4)
        changes_model = ModelChange(theta, old, new, which_change = perturb)
        new_vs = model.vs.copy()
        new_vs[i_vs] = new
        new_model = model._replace(vs = new_vs)

    elif perturb=='Dep':       # Change Depth of one node
        # Remember, we assume possible model points follow the following sequence
        # 1 km spacing from 0-60 km; 5 km spacing from 60-200 km
        idep = model.idep.copy()
        i_id = random.randrange(0,idep.size)
        # perturb around index in alldeps array
        theta =_GetStdForGaussian(perturb, itr)
        idep[i_id] = round(np.random.normal(idep[i_id],theta))
        changes_model = ModelChange(theta,model.idep[i_id],idep[i_id],
                                    which_change = perturb)
        new_model = model._replace(idep=idep)

    elif perturb=='Birth':     # Layer Birth
        # choose unoccupied position, unused_d (index i_d), with uniform probability
        idep = model.idep.copy()
        vs = model.vs.copy()
        # We'll try and do this with uniform probability across the depth range
        i_d = -1
        while i_d < 0:
            target_depth = random.random()*np.max(model.all_deps)
            test_i = np.argmin(np.abs(model.all_deps-target_depth))
            if test_i not in model.idep: i_d = test_i

        #unused_idep = [idx for idx,val in enumerate(model.all_deps)
        #        if idx not in idep]
        #i_d = random.sample(unused_idep, 1)[0]
        unused_d = model.all_deps[i_d]
        theta = _GetStdForGaussian(perturb, itr)
        # identify index of closest nucleus for old Vs value
        i_old = abs(model.all_deps[idep]-unused_d).argmin()
        # Make new array of depth indices, and identify new value index, i_new
        idep = np.sort(np.append(idep,i_d))
        i_new = np.where(idep==i_d)[0]
        vs = np.insert(vs,i_new,np.round(np.random.normal(vs[i_old],theta),4))
        changes_model = ModelChange(theta,model.vs[i_old],vs[i_new],
                                    which_change = perturb)
        new_model=model._replace(idep=idep,vs=vs)

    elif perturb=='Death':     # Layer Death
        # choose occupied position, i_id, with uniform probability
        idep = model.idep.copy()
        vs = model.vs.copy()
        i_id = random.randrange(0,idep.size) # this is a [min, max) range
        theta = _GetStdForGaussian(perturb, itr)
        kill_d = model.all_deps[idep[i_id]]
        idep_new = np.delete(idep,i_id)
        vs = np.delete(vs,i_id)
        i_new = abs(model.all_deps[idep_new]- kill_d).argmin()
        changes_model = ModelChange(theta,model.vs[i_id],vs[i_new],
                                    which_change = perturb)
        new_model = model._replace(idep=idep_new,vs=vs)
    else: # perturb=='Hyperparameter', Change a hyperparameter
        hyperparams = ['Std_RF','Lam_RF','Std_SWD']
        hyperparam = random.sample(hyperparams, 1)[0]
        theta = _GetStdForGaussian(hyperparam, itr)

        if hyperparam == 'Std_RF':
            ih = random.randint(0,len(model.std_rf)-1)
            old = model.std_rf[ih]
            new = np.round(np.random.normal(old,theta),5)
            new_noise = model.std_rf.copy()
            new_noise[ih] = new
            new_model = model._replace(std_rf = new_noise)
        elif hyperparam == 'Lam_RF':
            ih = random.randint(0, len(model.lam_rf)-1)
            old = model.lam_rf[ih]
            new = np.round(np.random.normal(old,theta),4)
            new_noise = model.lam_rf.copy()
            new_noise[ih] = new
            new_model = model._replace(lam_rf = new_noise)
        elif hyperparam == 'Std_SWD':
            old = model.std_swd
            new = np.round(np.random.normal(old,theta),4)
            new_model = model._replace(std_swd = new)

        changes_model = ModelChange(theta, old, new,
                                    which_change = 'Noise')

    return new_model, changes_model

def _GetStdForGaussian(perturb, itr) -> float:
    if perturb=='Vs':          # Change Vs in a layer
        theta=0.05             # B&c12's theta1
    elif perturb=='Dep':       # Change Depth of one node
        theta=2                # B&c12's theta3
    elif perturb=='Birth' or perturb=='Death': # Layer Birth/Death
        theta=0.5              # half B&c12's theta2 for new velocity
    elif perturb=='Std_RF':    # Change hyperparameter - std_rf
        theta=1e-3             # B&c12's theta_h_j
    elif perturb=='Std_SWD':   # Change hyperparameter - std_swd
        theta=1e-2             # B&c12's theta_h_j
    elif perturb=='Lam_RF':    # Change hyperparameter - lam_swd
        theta=1e-2

    if itr < 250:
        theta = theta*2

    return theta


# =============================================================================
#       Generate synthetic receiver functions based on model input
# =============================================================================
# This is based on propagator matrix code given to me by Helen Janiszewski
# Her citation is Haskell [1962], although the notation follows Ben-Menahan
# and Singh [1980] very closely.  We also need to calculate Vp and density
# from the Vs structure.  There are various (empirical) ways of doing this.
# We also assume that the synthetic ray is incident from 60 degrees, so input
# receiver function (observed) should also be stretched accordingly
def SynthesiseRF(fullmodel, rf_all) -> list: # of RecvFunc
    for irf in range(len(rf_all)):
        rf_in = rf_all[irf]

        # Make synthetic waveforms for Sp or Ps or both
        wv = _SynthesiseWV(fullmodel, rf_in)
        # Multiply by rotation matrix
        wv = _RotateToPSV(wv, fullmodel, rf_in)
        # Prep data (filter, crop, align)
        wv = _PrepWaveform(wv, Ts = [1, 100])

        # And deconvolve it
        rf_out = _CalculateRF(wv, rf_in)

        if irf == 0: rfs_out = [rf_out]
        else:        rfs_out.append(rf_out)

    return rfs_out


def _SynthesiseWV(synthmodel, rf_in) -> BodyWaveform:
    # And the filter band
    T=[1,100] # corner periods of filter, sec

    # Now set up the length, frequency etc info for the synthetic
    dt = 0.05 # timestep in seconds
    tmax = 50 # maximum length of signal, seconds
    n_fft = 2**(int(tmax/dt).bit_length()+1) # 2048 samples, for 50s at dt=0.05
                 # this is num_timesteps for FFT
    max_loop = 350 # this defines the frequencies looped through to calculate
                    # the synthetics.  Should be n_fft/2+1, but given that we
                    # are filtering the synthetics, calculating all those really
                    # high frequencies is just a waste of time
    tot_time = dt*n_fft # 102.4 seconds, for 50s at dt=0.05
    dom_freq = (1/tot_time)*(np.arange(1,max_loop))
    transfer_P_horz = np.zeros(n_fft, dtype = np.complex_)
    transfer_P_vert = np.zeros(n_fft, dtype = np.complex_)
    transfer_S_horz = np.zeros(n_fft, dtype = np.complex_)
    transfer_S_vert = np.zeros(n_fft, dtype = np.complex_)
    c=1/rf_in.ray_param
    vp = synthmodel.vp[-1] # Vp in halfspace
    vs = synthmodel.vs[-1] # Vs in halfspace

    for i in range(1,max_loop):
        freq = i/tot_time
        wavenumber = 2*np.pi*(freq/c) # 2*pi/wavelength
        J = _CalcPropagatorMatrix(synthmodel,c,wavenumber,calc_from_layer = 0)
        D = (J[0][0]-J[1][0])*(J[2][1]-J[3][1])-(J[0][1]-J[1][1])*(J[2][0]-J[3][0])
        # remember when indexing J that Python is zero-indexed!
        # Also that Z component polarity is reversed from BM&S because Z increases down!
        transfer_P_horz[i]=2*c/(D*vp)*(J[3][1]-J[2][1])
        transfer_P_vert[i]=2*c/(D*vp)*(J[3][0]-J[2][0])
        transfer_S_horz[i]=-c/(D*vs)*(J[0][1]-J[1][1])
        transfer_S_vert[i]=-c/(D*vs)*(J[0][0]-J[1][0])


    if rf_in.rf_phase == 'Ps':
        P_horz = _IFFTSynth(transfer_P_horz, max_loop, dom_freq, T, dt, tmax)
        P_vert = _IFFTSynth(transfer_P_vert, max_loop, dom_freq, T, dt, tmax)
        P_horz, P_vert = _NormaliseSynth(P_horz, P_vert, rf_in.rf_phase)
        return BodyWaveform(P_horz,P_vert,dt)

    elif rf_in.rf_phase == 'Sp':
        S_horz = _IFFTSynth(transfer_S_horz, max_loop, dom_freq, T, dt, tmax)
        S_vert = _IFFTSynth(transfer_S_vert, max_loop, dom_freq, T, dt, tmax)
        S_horz, S_vert = _NormaliseSynth(S_horz, S_vert, rf_in.rf_phase)
        return BodyWaveform(S_horz,S_vert,dt)

    elif rf_in.rf_phase == 'Both':
        P_horz = _IFFTSynth(transfer_P_horz, max_loop, dom_freq, T, dt, tmax)
        P_vert = _IFFTSynth(transfer_P_vert, max_loop, dom_freq, T, dt, tmax)
        P_horz, P_vert = _NormaliseSynth(P_horz, P_vert, 'Ps')

        S_horz = _IFFTSynth(transfer_S_horz, max_loop, dom_freq, T, dt, tmax)
        S_vert = _IFFTSynth(transfer_S_vert, max_loop, dom_freq, T, dt, tmax)
        S_horz, S_vert = _NormaliseSynth(S_horz, S_vert, 'Sp')

        return BodyWaveform(P_horz,P_vert,dt), BodyWaveform(S_horz,S_vert,dt)


def _NormaliseSynth(horz, vert, rf_phase):
    max_val = np.max(np.abs(np.concatenate([horz,vert])))
    horz = horz/max_val
    vert = vert/max_val

    if rf_phase == 'Sp':  # flip the time axis
        horz = horz[-1::-1]
        vert = vert[-1::-1]

    return horz, vert

def _IFFTSynth(transfer_comp, max_loop, dom_freq, T, dt, tmax):
    # apply a Gaussian filter
    transfer_comp[1:max_loop] *= np.exp(-dom_freq**2*T[0]/2)

    # Make symmetric for IFFT
    transfer_comp[-2:-max_loop:-1] = transfer_comp[1:max_loop-1]

    comp = np.real(np.fft.ifft(transfer_comp))/dt

    # Filter and cut to size
    comp = matlab.BpFilt(comp,T[0],T[1],dt)[:round(tmax/dt)]

    return comp


def _CalcPropagatorMatrix(synthmodel, c, wavenumber, calc_from_layer):
    # We can speed this up (fractionally!) by making sure we only do each
    # calculation once - hence this looks a bit confusing!
    vp = synthmodel.vp[calc_from_layer]
    vs = synthmodel.vs[calc_from_layer]
    rho = synthmodel.rho[calc_from_layer]

    rhoc2 = rho*c*c
    one_rhoc2 = 1/rhoc2
    rhovp2 = rho*vp*vp
    one_rhovp2 = 1/rhovp2
    vs_vp = vs/vp
    c_vp2 = c/vp*c/vp

    gamma = 2*vs/c*vs/c
    eta_vp = np.sqrt(c_vp2-1)
    eta_vs = np.sqrt(c/vs*c/vs-1)

    gamma1 = gamma-1
    gamma1_etavp = gamma1/eta_vp
    gamma1_etavs = gamma1/eta_vs
    gammaetavs = gamma*eta_vs
    gamma2 = gamma*gamma
    gamma12 = gamma1*gamma1

    if calc_from_layer == synthmodel.vs.size - 1:
        # Calculate propagation through the half space
        return np.array([[-2*vs_vp*vs_vp, 0, one_rhovp2, 0],
                         [0, c_vp2*gamma1_etavp, 0, one_rhovp2/eta_vp],
                         [gamma1_etavs/gamma, 0, -one_rhoc2/gammaetavs, 0],
                         [0, 1, 0, one_rhoc2/gamma]])

#        np.array([[-2*(vs/vp)**2, 0, (rho*vp**2)**-1, 0], # Slightly easier to read
#                [0,c**2*(gamma-1)*(vp**2*eta_vp)**-1,0,(rho*vp**2*eta_vp)**-1],
#                [(gamma-1)*(gamma*eta_vs)**-1,0,-(rho*c**2*gamma*eta_vs)**-1,0],
#                [0,1,0,(rho*c**2*gamma)**-1]])

    else:
        # Calculate propagation through each layer
        thick = synthmodel.thickness[calc_from_layer]
        P = wavenumber*eta_vp*thick
        Q = wavenumber*eta_vs*thick

        cosP = np.cos(P)
        sinP = np.sin(P)
        cosQ = np.cos(Q)
        sinQ = np.sin(Q)
        gammacosP = gamma*cosP
        gammacosQ = gamma*cosQ
        sinP_etavp = sinP/eta_vp
        sinQ_etavs = sinQ/eta_vs
        sinQetavs = sinQ*eta_vs
        sinPetavp = sinP*eta_vp

        a11 = gammacosP - (gammacosQ - cosQ)
        a12 = 1j*(gamma1_etavp*sinP + gammaetavs*sinQ)
        a13 = -(cosP-cosQ)/rhoc2
        a14 = 1j*((sinP_etavp + sinQetavs)/rhoc2)
        a21 = -1j*(gamma*sinPetavp + gamma1_etavs*sinQ)
        a22 = -gammacosP + cosP + gammacosQ
        a23 = 1j*((sinPetavp + sinQ_etavs)/rhoc2)
        a31 = rhoc2*gamma*(gammacosP-gammacosQ-cosP+cosQ)
        a32 = 1j*rhoc2*(sinP_etavp*gamma12 + gamma2*sinQetavs)
        a41 = 1j*rhoc2*(gamma2*sinPetavp + gamma12*sinQ_etavs)

        a_n = np.array([[a11, a12, a13, a14],
                        [a21, a22, a23, a13],
                        [a31, a32, a22, a12],
                        [a41, a31, a21, a11]])

#        a_n = np.array([
#                [
#                    gamma*np.cos(P)-(gamma-1)*np.cos(Q),
#                    1j*((gamma-1)*eta_vp**-1*np.sin(P)+gamma*eta_vs*np.sin(Q)),
#                    -(rho*c**2)**-1*(np.cos(P)-np.cos(Q)),
#                    1j*(rho*c**2)**-1*(eta_vp**-1*np.sin(P)+eta_vs*np.sin(Q)),
#                    ],
#                [
#                    -1j*(gamma*eta_vp*np.sin(P)+(gamma-1)*eta_vs**-1*np.sin(Q)),
#                    -(gamma-1)*np.cos(P)+gamma*np.cos(Q),
#                    1j*(rho*c**2)**-1*(eta_vp*np.sin(P)+eta_vs**-1*np.sin(Q)),
#                    -(rho*c**2)**-1*(np.cos(P)-np.cos(Q)),
#                    ],
#                [
#                    rho*c**2*gamma*(gamma-1)*(np.cos(P)-np.cos(Q)),
#                    1j*rho*c**2*((gamma-1)**2*eta_vp**-1*np.sin(P)+gamma**2*eta_vs*np.sin(Q)),
#                    -(gamma-1)*np.cos(P)+gamma*np.cos(Q),
#                    1j*((gamma-1)*eta_vp**-1*np.sin(P)+gamma*eta_vs*np.sin(Q)),
#                    ],
#                [
#                    1j*rho*c**2*(gamma**2*eta_vp*np.sin(P)+(gamma-1)**2*eta_vs**-1*np.sin(Q)),
#                    rho*c**2*gamma*(gamma-1)*(np.cos(P)-np.cos(Q)),
#                    -1j*(gamma*eta_vp*np.sin(P)+(gamma-1)*eta_vs**-1*np.sin(Q)),
#                    gamma*np.cos(P)-(gamma-1)*np.cos(Q),
#                    ],
#                ])
        out= np.matmul(
                _CalcPropagatorMatrix(synthmodel,c,wavenumber,calc_from_layer+1),
                a_n)
        return out

# This is a free surface transform, used to rotate from RTZ to P-SV-SH
# while surpressing the effects of the free surface
#   Bostock, M.G., 1998. Mantle stratigraphy and evolution of the Slave province.
#       J. Geophys. Res. 103, 21183–21200.
#   Kennett, B.L.N., 1991. The removal of free surface interactions from
#       three-component seismograms. Geophys. J. Int. 104, 153–163.
def _RotateToPSV(waveform, fullmodel, rf_in) -> RotBodyWaveform:
    vp_surface = fullmodel.vp[0]
    vs_surface = fullmodel.vs[0]
    ray_param = rf_in.ray_param
    rf_phase = rf_in.rf_phase


    a = np.sqrt(vp_surface**-2 - ray_param**2)
    b = np.sqrt(vs_surface**-2 - ray_param**2)
    rot_mat = np.array([[ray_param * vs_surface**2 / vp_surface,
        (vs_surface**2 * ray_param**2 - 1/2) / (a * vp_surface)],
        [(1/2 - vs_surface**2 * ray_param**2) / (b * vs_surface),
         ray_param * vs_surface]])
    rotated = np.matmul(rot_mat, np.vstack((waveform.amp_R, -waveform.amp_Z)))
    # Remove the mean line (can't just detrend as is skewed by the peaks)
    amp_P = rotated[0]
    amp_SV = rotated[1]
    amp_P = amp_P - np.linspace(np.median(amp_P[:50]),
           np.median(amp_P[-50:]), amp_P.size)
    amp_SV = amp_SV - np.linspace(np.median(amp_SV[:50]),
           np.median(amp_SV[-50:]), amp_SV.size)

    if rf_phase == 'Ps':
        return RotBodyWaveform(parent = amp_P, daughter = amp_SV, dt = waveform.dt)
    elif rf_phase == 'Sp':
        return RotBodyWaveform(parent = amp_SV, daughter = amp_P, dt = waveform.dt)

#  This follows the way that the real data is prepped and deconvolved.
#   Abt, D.L., Fischer, K.M., French, S.W., Ford, H.A., Yuan, H., Romanowicz,
#       B., 2010.  North American lithospheric discontinuity structure imaged
#       by Ps and Sp receiver functions. J. Geophys. Res. 115, B09301.
#   Lekić, V., French, S. W., & Fischer, K. M. (2011). Lithospheric Thinning
#       Beneath Rifted Regions of Southern California. Science, 334(6057),
#       783–787. https://doi.org/10.1126/science.1208898
#   Mancinelli, N. J., Fischer, K. M., & Dalton, C. A. (2017). How Sharp Is the
#        Cratonic Lithosphere-Asthenosphere Transition? Geophysical Research
#       Letters, 44(20), 2017GL074518. https://doi.org/10.1002/2017GL074518
def _PrepWaveform(waveform, Ts) -> RotBodyWaveform:
    # N.B. Ts are the filter corners (as PERIOD, in seconds)
    parent = waveform.parent.copy()
    daughter = waveform.daughter.copy()
    dt = waveform.dt

    # Identify the direct arrival (max peak) - INDEX
    tshift = np.argmax(np.abs(parent))

    # Filter, crop, and align
    #   Set the window length to 100 ish sec, centred on incident phase by
    #   padding with zeros (actually 2048, so a power of 2 close to 100s
    #   ASSUMING dt == 0.05 s !!)
    # N.B.  Synthesised signal length has to be <= half the padded signal length
    #       here (n_fft) for the padding to work under all circumstances
    tot_time = 100
    n_fft = 2**(int(tot_time/dt).bit_length()) # next power of 2 (2048)
    i_arr = int(n_fft/2)
    parent = np.concatenate([ # pad the beginning with zeros so incident phase
                np.zeros(i_arr - tshift), parent]) # at tot_time/2
    parent = np.concatenate([ # pad the end with zeros
            parent, np.zeros(n_fft - parent.size)])
    daughter = np.concatenate([ # pad the beginning with zeros
            np.zeros(i_arr - tshift), daughter])
    daughter = np.concatenate([ # pad the end with zeros
            daughter, np.zeros(n_fft - daughter.size)])

    # Bandpass filter
    parent = matlab.BpFilt(parent, Ts[0], Ts[1], dt)
    daughter = matlab.BpFilt(daughter, Ts[0], Ts[1], dt)

    # Taper the data
    #    This is written so all positions are INDICES not actual time
    # Want to taper the parent much more severely than the daughter

    # First, taper the parent
    taper_width = 5  # Arbitrary (ish), but works for real data processing
    taper_length = 30 # Arbitrary (ish), but works for real data processing
    parent = matlab.Taper(parent, int(taper_width/dt), i_arr - int(taper_length/2/dt),
                  i_arr + int(taper_length/2/dt))
    # And the daughter
    taper_width = 20
    daughter = matlab.Taper(daughter, int(taper_width/dt), int(taper_width/dt),
                            daughter.size - int(taper_width/dt))

    return RotBodyWaveform(parent, daughter, dt)



def _CalculateRF(waveform, rf_in) -> RecvFunc:
    # For output file, we want a relatively sparsely sampled RF (for inversion)
    #   say 30 s long with a dt of 0.25 s?
    rf_tmax = 30
    rf_dt = 0.25

    # Define the Slepians for the MTM
    #   Use a 50 second window, num_Slepian = 4, num_tapers = 7
    win_length = 50 # Have 100s long signal
    i_shift = 0.1  # these windows will overlap by 90% (10% shift each time)
    num_tapers = 7
    n_samp = 2**(int(win_length / waveform.dt).bit_length()) # for FFT - 1024 samples
    tapers = matlab.slepian(n_samp, num_tapers)


    # We have already required (in PrepWaveform) that tot_time = 100

    # Pad with zeros to give wiggle room on large ETMTM windows
    pad = np.zeros(int(np.ceil(n_samp*i_shift)))
    Daughter = np.concatenate([pad, waveform.daughter, pad])
    i_starts = np.arange(0,Daughter.size - (n_samp - 1), int(i_shift*n_samp))

    i_t0 = int(waveform.parent.size/2)
    incident_win = i_t0+int(n_samp/2)*np.array([-1, 1])
    Parent = waveform.parent[incident_win[0]:incident_win[1]]

    # Normalise to amplitude of 5 (this should be normalised by S2N)
    norm_by = 5/np.max(np.abs(Parent))
    Parent = Parent*norm_by
    Daughter = Daughter*norm_by

    P_fft = _ETMTMSumFFT(tapers, Parent, Parent, num_tapers)
    norm_by = np.max(np.abs(np.real(np.fft.ifft(P_fft/(P_fft + 10)))))
    w_RFs = np.zeros((i_starts.size, Daughter.size))
    max_RF = 0

    for i in range(i_starts.size):
        # Isolate and taper the daughter window
        Daughter_win = matlab.Taper( Daughter[i_starts[i] : i_starts[i] + n_samp],
                             int(n_samp/5), int(n_samp/5), int(4*n_samp/5))
        w_D_fft = _ETMTMSumFFT(tapers, Parent, Daughter_win, num_tapers)
        w_RF = np.real(np.fft.ifft(w_D_fft/(P_fft + 10)))/norm_by # Arbitrary water level = 10
        w_RF = np.fft.fftshift(w_RF)
        w_RFs[i, i_starts[i]:i_starts[i]+n_samp] = w_RF
        max_RF = np.max([max_RF, np.max(np.abs(w_RF))])

    i_t0 = i_t0 + pad.size - 1
    RF = np.mean(w_RFs, 0)

    # Filter as the original RF (corners specified in input file)
    RF =  matlab.BpFilt(RF, rf_in.filter_corners[0],
                        rf_in.filter_corners[1], waveform.dt)

    # Normalise to max amp of an individual window
    RF = RF/np.max(np.abs(RF)) * max_RF


    # Want time window from incident phase to + 30s
    # and resample with larger dt
    RF = RF[i_t0 : i_t0 + int(rf_tmax/waveform.dt)]

    n_jump = int(rf_dt / waveform.dt)
    if n_jump == rf_dt / waveform.dt:
        RF = RF[::n_jump]
    else:
        RF = interp1d(np.arange(0, rf_tmax, waveform.dt),
                      RF)(np.arange(0, rf_tmax, rf_dt))


    return RecvFunc(amp = RF, dt = rf_dt, ray_param = rf_in.ray_param,
                    std_sc = rf_in.std_sc, rf_phase = rf_in.rf_phase,
                    filter_corners = rf_in.filter_corners,
                    weight_by = rf_in.weight_by)

#  Extended time multi-taper method of deconvolution:
#   Helffrich, G., 2006. Extended-time multitaper frequency domain
#       cross-correlation receiver-function estimation. Bull. Seismol. Soc. Am.
#        96, 344–347.
def _ETMTMSumFFT(slepian_tapers, data_1, data_2, num_window):
    which_window = num_window-1 # as zero-indexed
    if which_window == 0:
        return (np.conj(np.fft.fft(slepian_tapers[:,which_window]*data_1)) *
                np.fft.fft(slepian_tapers[:,which_window]*data_2))
    else:
        return (np.conj(np.fft.fft(slepian_tapers[:,which_window]*data_1)) *
                np.fft.fft(slepian_tapers[:,which_window]*data_2) +
                _ETMTMSumFFT(slepian_tapers, data_1, data_2, num_window-1))


# =============================================================================
#       Synthesise Surface Wave Dispersion measurements
# =============================================================================
# This is based on Matlab code from Bill Menke via Natalie Accardo
#         - MATLAB code includes following copyright info
#    Copyright 2003 by Glenn J. Rix and Carlo G. Lai
#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU General Public License
#    as published by the Free Software Foundation
#
#   The algorithms are based on:
#   Hisada, Y., (1994). "An Efficient Method for Computing Green's Functions for
#   a Layered Half-Space with Sources and Receivers at Close Depths," Bulletin of
#   the Seismological Society of America, Vol. 84, No. 5, pp. 1456-1472.
#
#   Lai, C.G., (1998). "Simultaneous Inversion of Rayleigh Phase Velocity and
#   Attenuation for Near-Surface Site Characterization," Ph.D. Dissertation,
#   Georgia Institute of Technology.

def SynthesiseSWD(model, period, itr) -> SurfaceWaveDisp:
    # Save time in early iterations when just trying to get the ballpark answer
    # by using a coarser estimate for the forward modelled dispersion
    if itr < 1e4:
        ifcoarse = True
    else:
        ifcoarse = False



    freq = 1/period

    if model.vs.size == 1:
        cr = _CalcRaylPhaseVelInHalfSpace(model.vp[0], model.vs[0])
        return SurfaceWaveDisp(period = period,
                               c = np.ones(freq.size)*cr)
        # no freq. dependence in homogeneous half space

    # Set bounds of tested Rayleigh wave phase velocity
    cr_max = np.max(model.vs)
    cr_min = 0.98 * _CalcRaylPhaseVelInHalfSpace(np.min(model.vp),
                                                 np.min(model.vs))
    omega = 2*np.pi*freq
    n_ksteps = 20 # assume this is finely spaced enough for our purposes
        # was set to 15 when including findmin # Original code had 200, but this should speed things up
    cr = np.zeros(omega.size)

    # Look for the wavenumber (i.e. vel) with minimum secular function value
    k_lims = np.vstack((omega/cr_max, omega/cr_min))


    mu = model.rho * model.vs**2
    for i_om in range(omega.size):
        # Limit wavenumber search range in order to speed things up??
        # Tried this and it definitely broke everything...

        if ifcoarse:
            cr[i_om] = _FindMinValueSecularFunctionCoarse(omega[i_om], k_lims[:,i_om],
              n_ksteps, model.thickness, model.rho, model.vp, model.vs, mu)
        else:
            cr[i_om] = _FindMinValueSecularFunctionFine(omega[i_om], k_lims[:,i_om],
              n_ksteps, model.thickness, model.rho, model.vp, model.vs, mu)

    return SurfaceWaveDisp(period = period, c = cr)

def _CalcRaylPhaseVelInHalfSpace(vp, vs):
    # Note: they actually do this in a cleverer way (see commented MATLAB code)
    #       but it is 5x faster to just do it this way, and it seems to give
    #       answers within about 0.5% (for Vp/Vs > 1.5).
    vp2=vp*vp
    vs2=vs*vs
    nu = (0.5*vp2-vs2)/(vp2-vs2) # Poisson's ratio

    phase_vel_rayleigh = vs*((0.862+1.14*nu)/(1+nu)); # the estimated velocity (Achenbach, 1973)

    #% Define Coefficients of Rayleigh's Equation
    #a =  1;
    #b = -8;
    #c =  8*(3-2*(cvs*cvs)/(cvp*cvp));
    #d = 16*((cvs*cvs)/(cvp*cvp)-1);
    #
    #% Solution of Rayleigh Equation
    #p   = [a b c d];
    #x   = roots(p);
    #cr  = cvs*sqrt(x);
    #
    #% Determine which of the roots is correct using the estimated velocity (Achenbach, 1973)
    #crest = cvs*((0.862+1.14*nu)/(1+nu));
    #cvr = cr(abs(cr-crest) == min(abs(cr-crest)));

    return phase_vel_rayleigh

def _FindMinValueSecularFunctionCoarse(omega, k_lims, n_ksteps, thick, rho, vp, vs, mu):
    #tol_s = 0.1 # This is as original code

    # Define vector of possible wavenumbers to try
    wavenumbers = np.linspace(k_lims[0], k_lims[1], n_ksteps)


    f1 = 1e-10  # arbitrarily low values so f2 < f1 & f2 < f3 never true for 2 rounds
    f2 = 1e-9
    k2 = 0 # irrelevant, will not be used unless enter IF statement below
           # and should be replaced with real values before then
    c = 0
    for i_k in range(-1, -wavenumbers.size-1,-1):
        k3 = wavenumbers[i_k]

        f3 = _Secular(k3, omega, thick, mu, rho, vp, vs)

        if f2 < f1 and f2 < f3: # if f2 has minimum of the 3 values
            # Doing it properly (as _FindMinValueSecularFunctionFine) is REALLY SLOW
            c = omega/k2
            break
        else:
             f1 = f2  # cycle through wavenumber values
             f2 = f3
             k2 = k3

    if c == 0 and n_ksteps <= 250:
        print(n_ksteps)
        c = _FindMinValueSecularFunctionCoarse(omega, k_lims, n_ksteps+100, thick,
                                         rho, vp, vs, mu)

    return c

def _FindMinValueSecularFunctionFine(omega, k_lims, n_ksteps, thick, rho, vp, vs, mu):
    tol_s = 0.1 # This is as original code

    # Define vector of possible wavenumbers to try
    wavenumbers = np.linspace(k_lims[0], k_lims[1], n_ksteps)


    f1 = 1e-10  # arbitrarily low values so f2 < f1 & f2 < f3 never true for 2 rounds
    f2 = 1e-9
    k1 = 0 # irrelevant, will not be used unless enter IF statement below
    k2 = 0 # and should be replaced with real values before then
    c = 0
    for i_k in range(-1, -wavenumbers.size-1,-1):
        k3 = wavenumbers[i_k]

        f3 = _Secular(k3, omega, thick, mu, rho, vp, vs)

        if f2 < f1 and f2 < f3: # if f2 has minimum of the 3 values
            # Find minimum more finely
            kmin, fmin = matlab.findmin(_Secular, brack = (k3, k2, k1),
                                 args = (omega, thick, mu, rho, vp, vs))
            if fmin < tol_s:
                c = omega/kmin
                break

        else:
             f1 = f2  # cycle through wavenumber values
             f2 = f3
             k1 = k2
             k2 = k3

    if c == 0 and n_ksteps <= 250:
        print(n_ksteps)
        c = _FindMinValueSecularFunctionFine(omega, k_lims, n_ksteps+100, thick,
                                         rho, vp, vs, mu)

    return c

def _m3D(x):
    return np.reshape(x,(1,1,x.size))

def _Secular(k, om, thick, mu, rho, vp, vs):
    # Ok.  This is written in kind of a weird way (following the MATLAB code)
    # but that's because it minimises repetition of any multiplication etc -
    # so it should be faster to run, even if it seems like an odd way of doing
    # things...

    # Check to see if the trial phase velocity is equal to the shear wave velocity
    # or compression wave velocity of one of the layers
    epsilon = 0.0001;
    while np.any(np.abs(om/k-vs)<epsilon) or np.any(np.abs(om/k-vp)<epsilon):
        k = k * (1+epsilon)


    # First, calculate some commonly used variables
    k = k+0j
    nu_s = np.sqrt(k**2 - om**2/vs**2) # 1 x nl
    inds = np.imag(-1j*nu_s)>0
    nu_s[inds] = -nu_s[inds]
    gamma_s = _m3D(nu_s/k) # 1 x nl
    nu_p = np.sqrt(k**2 - om**2/vp**2) # 1 x nl
    inds = np.imag(-1j*nu_p)>0
    nu_p[inds] = -nu_p[inds]
    gamma_p = _m3D(nu_p/k) # 1 x nl
    chi = 2*k - (om**2/vs**2)/k  # nk x nl
    thick = _m3D(thick)



    # First, calculate the E and Lambda matrices (up-going and
    # down-going matrices) for the P-SV case.
    ar1 = np.ones((1,1,mu.size))
    ar0 = np.zeros((1,1,mu.size))
    muchi = _m3D(mu*chi)
    munup = _m3D(2*mu*nu_p)
    munus = _m3D(2*mu*nu_s)

    E11 = np.vstack((np.hstack((-ar1,gamma_s)),np.hstack((-gamma_p, ar1))))
    E12 = np.vstack((np.hstack((-ar1, gamma_s)), np.hstack((gamma_p, -ar1))))
    E21 = np.vstack((np.hstack((munup, -muchi)), np.hstack((muchi, -munus))))
    E22 = np.vstack((np.hstack((-munup, muchi)), np.hstack((muchi, -munus))))
    du  = np.vstack((np.hstack((np.exp(-nu_p*thick), ar0)),
                     np.hstack((ar0, np.exp(-nu_s*thick)))))
    X = np.zeros((4,4,mu.size-1))+0j

    for iv in range(mu.size - 2):
        A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                       np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
        B = np.vstack((np.hstack((E11[:,:,iv],-E12[:,:,iv+1])),
                       np.hstack((E21[:,:,iv], -E22[:,:,iv+1]))))
        L = np.vstack((np.hstack((du[:,:,iv],np.zeros((2,2)))),
                       np.hstack((np.zeros((2,2)),du[:,:,iv+1]))))
        X[:,:,iv] = matlab.mldivide(A,np.matmul(B,L))

    # And the deepest layer (above the halfspace)
    iv = mu.size - 2
    A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                   np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
    B = np.vstack((E11[:,:,iv],E21[:,:,iv]))
    L = du[:,:,iv]
    X[:,:2,-1] = matlab.mldivide(A, np.matmul(B,L))


    #  Calculate the modified R/T coefficients
    Td = np.zeros((2,2,mu.size-1))+0j
    Rd = np.zeros((2,2,mu.size-1))+0j

    Td[:,:,mu.size-2] = X[:2,:2,mu.size-2] # The deepest layer above HS
    Rd[:,:,mu.size-2] = X[2:,:2,mu.size-2]
    for iv in range(mu.size-3,-1,-1):
        Td[:,:,iv] = matlab.mldivide(
                        (np.identity(2) - np.matmul(X[:2,2:,iv],Rd[:,:,iv+1])),
                        X[:2,:2,iv])
        Rd[:,:,iv] = X[2:,:2,iv] + np.matmul(np.matmul(X[2:,2:,iv],Rd[:,:,iv+1]),
                                              Td[:,:,iv])


    # And finally, calculate the absolute value of the secular function
    d = (np.abs(np.linalg.det(E21[:,:,0] +
                      np.matmul(np.matmul(E22[:,:,0],du[:,:,0]), Rd[:,:,0]))
                    /(nu_s[0]*nu_p[0]*mu[0]**2)))



    return d




# =============================================================================
#       Find the misfit (scaled by covariance matrix)
# =============================================================================
def Mahalanobis(rf_obs_all, rf_synth_all, swd_obs,swd_synth, inv_cov) -> float:

    # Make single vector out of all rf_obs
    rf_obs = rf_obs_all[0].amp
    rf_synth = rf_synth_all[0].amp

    for irf in range(1,len(rf_obs_all)):
        rf_obs = np.hstack((rf_obs, rf_obs_all[irf].amp))
        rf_synth = np.hstack((rf_synth, rf_synth_all[irf].amp))

#    g_m = np.concatenate((rf_synth, swd_synth.c))
#    d_obs = np.concatenate((rf_obs, swd_obs.c))
#    misfit = g_m - d_obs

    # More equally weight the surface wave and RF datasets?
    misfit_rf = rf_synth-rf_obs
    misfit_swd = swd_synth.c - swd_obs.c
    if rf_obs_all[0].weight_by == 'even': misfit_rf = misfit_rf * misfit_swd.size/misfit_rf.size
    if type(rf_obs_all[0].weight_by) == int:
        misfit_rf = misfit_rf * rf_obs_all[0].weight_by * misfit_swd.size/misfit_rf.size
    misfit = np.concatenate((misfit_rf, misfit_swd))

    # N.B. with np.matmul, it prepends or appends an additional dimension
    # (depending on the position in matmul) if one of the arguments is a vector
    # so don't need to explicitly transpose below
    return np.matmul(np.matmul(misfit,inv_cov),misfit)

# =============================================================================
#       Choose whether or not to accept the change
# =============================================================================
def AcceptFromLikelihood(misfit_to_obs_m, misfit_to_obs_m0,
                         cov_m, cov_m0, model_change):
    # We have input misfit (smaller misfit == better fit to data!) for
    # the new model (_m), and the old model (_m0).  A much improved fit means
    # misfit_m << misfit_m0 (and a much worse fit means misfit_m0 << misfit_m)
    # We will output alpha, the likelihood of accepting (where high alpha
    # means definitely accept, and low alpha means definitely don't.)

    # Smaller misfit means better fit, so if it is just a lot better,
    # accept it without testing (to save some calculations!)
    if 500 < misfit_to_obs_m0 - misfit_to_obs_m: return (1, 1e300)
    if 500 < misfit_to_obs_m - misfit_to_obs_m0: return (0, 1e-300)
    # Form is dependent on which variable changed
    perturb = model_change.which_change
    if perturb == 'Vs' or perturb == 'Dep':
        alpha_m_m0 = np.exp(-(misfit_to_obs_m - misfit_to_obs_m0)/2)
    elif perturb == 'Birth':
        dv = np.abs(model_change.old_param - model_change.new_param)
        alpha_m_m0 = ((model_change.theta * np.sqrt(2*np.pi) / dv) *
                      np.exp((dv*dv/(2*model_change.theta**2)) -
                             (misfit_to_obs_m - misfit_to_obs_m0)/2))
    elif perturb == 'Death':
        dv = np.abs(model_change.old_param - model_change.new_param)
        alpha_m_m0 = ((dv / (model_change.theta * np.sqrt(2*np.pi))) *
                      np.exp(-(dv*dv/(2*model_change.theta**2)) -
                             (misfit_to_obs_m - misfit_to_obs_m0)/2))
    elif perturb == 'Noise':
        alpha_m_m0 = ((cov_m0.detCovar/cov_m.detCovar) *
                      np.exp(-(misfit_to_obs_m - misfit_to_obs_m0)/2))

#    if alpha_m_m0 == 0:
#        print('Perturb ', perturb, ': ', round(misfit_to_obs_m), ' (new), ',
#               round(misfit_to_obs_m0), ' (old) (', model_change.new_param
#               , ', ', model_change.old_param, ') *********')
#    else:
#        print('Perturb ', perturb, ': ', round(misfit_to_obs_m), ', ',
#               round(misfit_to_obs_m0),  ' (', model_change.new_param
#               , ', ', model_change.old_param, ')' )
    if alpha_m_m0 <= 1e-5:
        return (0, 1e-5)
    keep_yn = random.random() # generate random number between 0-1
    return (keep_yn < alpha_m_m0, alpha_m_m0)


def SaveModel(fullmodel, deps):
    vs = np.zeros_like(deps)
    for k in range(fullmodel.layertops.size):
        vs[deps>=fullmodel.layertops[k]] = fullmodel.vs[k]
    return vs
