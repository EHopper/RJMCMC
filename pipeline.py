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
import matlab


# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class RecvFunc(typing.NamedTuple):
    amp: np.array  # assumed to be processed in same way as synthetics are here
    dt: float      # constant timestep in s
# Receiver function assumed to start at t=0s with a constant timestep, dt
# required to be stretched to constant RP (equivalent to distance of 60 degrees)

class BodyWaveform(typing.NamedTuple):
    amp_R: np.array # horizontal (radial) energy
    amp_Z: np.array # vertical energy
    dt: float
    
class RotBodyWaveform(typing.NamedTuple):
    amp_P: np.array # P energy
    amp_SV: np.array # SV energy
    dt: float

class SurfaceWaveDisp(typing.NamedTuple):
    period: np.array    # dominant period of dispersion measurement, seconds
    c: np.array  # phase velocity in km/s
# Surface wave dispersion measurements are allowed at any series of periods (s)

class Model(typing.NamedTuple):
    vs: np.array
    all_deps: np.array    # all possible depth nodes
    idep: np.array        # indices of used depth nodes
    std_rf: float
    lam_rf: float
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
# Reasonable min and max values for all model parameters define the prior 
# distribution - i.e. uniform probability within some reasonable range of values

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
    ray_param: float


# =============================================================================
# Overall process for the joint inversion
# =============================================================================

def JointInversion(rf_obs: RecvFunc, swd_obs: SurfaceWaveDisp, lims: Limits,
                   max_iter: int, random_seed: int) -> Model:
    
    # N.B.  The variable random_seed ensures that the process is repeatable.
    #       Set this in the master code so can start multiple unique chains.
    #       random_seed must be passed to any subfunction that uses random no.
    
    # =========================================================================
    #      Start by calculating for some (arbitrary) initial model
    # =========================================================================
    model_0 = InitialModel(rf_obs.amp,random_seed)
    if not CheckPrior(model_0, lims):
        return "Starting model does not fit prior distribution"
    
    # Calculate covariance matrix (and inverse, and determinant)
    cov_m0 = CalcCovarianceMatrix(model_0,rf_obs,swd_obs)
    
    # Calculate synthetic data
        # First, need to define the Vp and rho
    fullmodel_0 = MakeFullModel(model_0)
    rf_synth_m0 = SynthesiseRF(fullmodel_0) # forward model the RF data
    swd_synth_m0 = SynthesiseSWD(fullmodel_0, swd_obs.period) # forward model the SWD data
    
    # Calculate fit
    fit_to_obs_m0 = Mahalanobis(
            rf_obs.amp, rf_synth_m0.amp, # RecvFunc
            swd_obs.c, swd_synth_m0.c,  # SurfaceWaveDisp
            cov_m0.invCovar,             # CovarianceMatrix
            )
    
    
# =============================================================================
#           Define parameters for convergence
# =============================================================================
    num_posterior = 200
    burn_in = 2e5 # Number of models to run before bothering to test convergence
    ddeps=np.zeros(model_0.all_deps.size*2-1)
    ddeps[::2] = model_0.all_deps
    ddeps[1::2] = model_0.all_deps[:-1]+np.diff(model_0.all_deps)/2
    conv_models = np.zeros((ddeps.size,num_posterior+1))
    conv_models[:,0] = ddeps
    all_models = np.zeros((ddeps.size,int(max_iter/1)+1))
    all_models[:,0] = ddeps
    all_models[:,1] = SaveModel(fullmodel_0, ddeps)
    all_phi = np.zeros(max_iter) # all misfits
    all_alpha = np.zeros(max_iter-1) # all likelihoods
    all_keep = np.zeros(max_iter-1)
    # Within one chain, we test for convergence by misfit being and remaining
    # low, and likelihood being and remaining high
    
    all_phi[0] = fit_to_obs_m0
    converged = 0  # variable defines # of iterations since convergence
    
    
    # =========================================================================
    #       Iterate by Reverse Jump Markov Chain Monte Carlo
    # =========================================================================    
    for itr in range(1,max_iter):
        if not itr % 20: 
            print("Iteration {}..".format(itr))
        
        # Generate new model by perturbing the old model
        model, changes_model = Mutate(model_0,itr)
        
        if not CheckPrior(model, lims): # check if mutated model compatible with prior distr.
            #print('Failed Prior')
            continue  # if not, continue to next iteration
        
        fullmodel = MakeFullModel(model)
        if changes_model.which_change == 'Noise': 
            # only need to recalculate covariance matrix if changed hyperparameter
            cov_m = CalcCovarianceMatrix(model,rf_obs,swd_obs)
            rf_synth_m = rf_synth_m0
            swd_synth_m = swd_synth_m0
        else:
            cov_m = cov_m0
            rf_synth_m = SynthesiseRF(fullmodel) # forward model the RF data
            swd_synth_m = SynthesiseSWD(fullmodel, swd_obs.period) # forward model the SWD data
        
        # Calculate fit
        fit_to_obs_m = Mahalanobis(
                rf_obs.amp, rf_synth_m.amp, # RecvFunc
                swd_obs.c, swd_synth_m.c,  # SurfaceWaveDisp
                cov_m.invCovar,             # CovarianceMatrix
                )
        all_phi[itr] = fit_to_obs_m
        
        # Calculate probability of keeping mutation
        (keep_yn, all_alpha[itr-1]) = AcceptFromLikelihood(
                fit_to_obs_m, fit_to_obs_m0, # float
                cov_m, cov_m0, # CovarianceMatrix  
                changes_model, # form depends on which change
                )
        all_keep[itr-1] = keep_yn
        #print([changes_model.which_change, all_keep[itr-1], round(fit_to_obs_m,0)])
        if keep_yn: # should be aiming to keep about 40% of perturbations
            model_0 = model
            fullmodel_0 = fullmodel
            cov_m0 = cov_m
            rf_synth_m0 = rf_synth_m
            swd_synth_m0 = swd_synth_m
            fit_to_obs_m0 = fit_to_obs_m
            
            # Test if have converged
            nw = np.min((int(1e5), burn_in))
            if itr == burn_in:
                phi_0_std = all_phi[:nw].std()
                phi_0_mean = all_phi[:nw].mean()
                alpha_0_std = all_alpha[:nw-1].std()
                alpha_0_mean = all_alpha[:nw-1].mean()
            # Test after burn_in iterations every 1000 iterations
            if itr > burn_in and not converged and not (itr-burn_in) % 1e3:
                converged = TestConvergence(all_phi[:itr], all_alpha[:itr-1],
                                            phi_0_std, phi_0_mean, 
                                            alpha_0_std, alpha_0_mean,
                                            itr, nw)
        
        # Save every 500th iteration to see progress
        all_models[:,itr+1] = SaveModel(fullmodel, all_models[:,0])
        
        # Save every 500th iteration after convergence (KL14)
        if converged:
            converged = converged + 1
            if not converged % 500: 
                conv_models[:,(converged/500)+1] = SaveModel(
                        fullmodel, conv_models[:,0])
                
                if converged/500 >= num_posterior:
                    return (conv_models, all_models, all_phi, all_alpha, all_keep,
                            fullmodel)
            
        
    
    return (conv_models, all_models, all_phi, all_alpha, all_keep,
                            fullmodel)


 
    
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
def InitialModel(rf_obs,random_seed) -> Model:
    random.seed(a = random_seed)
    np.random.seed(seed = random_seed+1)
    vs = np.array([round(random.uniform(0.5,5.5),2)])   # arbitrary
    all_deps = np.concatenate((np.arange(0,10,0.2), 
                                np.arange(10,60,1), np.arange(60,201,5)))
    idep = np.array([random.randrange(0,len(all_deps))])  # arbitrary
    std_rf = round(np.std(rf_obs,ddof=1),2) # start off assuming all data is noise (KL14)
    lam_rf = 0.2 # this is as the example given by KL14
    std_swd = 0.15 # arbitrary - the max allowed in Geoff Abers' code
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
        model.vs.size == model.idep.size
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
    
    # Assume waveforms are incident from 60 degrees
    # For P waves, this means horizontal slowness = 0.0618
    # For incident S waves, this means p = 0.1157 (using AK135)
    ray_param = 0.0618 # (horizontal slowness at turning point)
    # phase_vel = 1/horizontal_slowness
    vp = np.round(vp, 3)
    rho = np.round(rho, 3)
    
    return SynthModel(model.vs, vp, rho, thickness, layertops, avdep, ray_param)  


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
    R=np.zeros((rf_obs.amp.size,rf_obs.amp.size))
    for a in range(rf_obs.amp.size):
        for b in range(rf_obs.amp.size):
            R[a,b]=(np.exp(-(model.lam_rf*rf_obs.dt*abs(a-b)))*
             np.cos(model.lam_rf*w0*rf_obs.dt*abs(a-b)))
           
    
    covar=np.zeros((rf_obs.amp.size+swd_obs.period.size,
                    rf_obs.amp.size+swd_obs.period.size))
    covar[:rf_obs.amp.size,:rf_obs.amp.size]=R*model.std_rf**2
    covar[-swd_obs.period.size:,-swd_obs.period.size:]=(
            np.identity(swd_obs.period.size)*model.std_swd**2)
    
    invc=np.linalg.inv(covar)
    #  Note: Need to take the determinant for use in calculating the acceptance
    #        probability if changing a hyperparameter.  However, given that 
    #        this is a 130x130 odd matrix with all values <1, the determinant
    #        is WAY too small.  However, we actually only care about the RATIO
    #        between the new and old determinant, so it is fine to just scale
    #        this up so that it doesn't get lost in the rounding error.
    #  detc=np.linalg.det(covar)
    detc = np.linalg.det(covar*1e4)
    detc = detc / (10**int(np.log10(detc)))

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
    elif itr <= 1000*(model.idep.size-1)*(model.idep.size): # as KL14
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
        unused_idep = [idx for idx,val in enumerate(model.all_deps) 
                if idx not in idep]
        i_d = random.sample(unused_idep, 1)[0]
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
            old = model.std_rf
            new = np.round(np.random.normal(old,theta),5)
            new_model = model._replace(std_rf = new)
        elif hyperparam == 'Lam_RF':
            old = model.lam_rf
            new = np.round(np.random.normal(old,theta),4) 
            new_model = model._replace(lam_rf = new)
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
def SynthesiseRF(fullmodel) -> RecvFunc:
    # Make synthetic waveforms
    wv = _SynthesiseWV(fullmodel)
    # Multiply by rotation matrix
    wv = _RotateToPSV(wv, fullmodel.vp[0], fullmodel.vs[0], fullmodel.ray_param)
    # Prep data (filter, crop, align)
    wv = _PrepWaveform(wv, Ts = [1, 100]) 
    
    # And deconvolve it
    rf = _CalculateRF(wv)
    
    return rf


def _SynthesiseWV(synthmodel) -> BodyWaveform:
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
    # transfer_S_horz = np.zeros(n_fft)
    # transfer_S_vert = np.zeros(n_fft)
    c=1/synthmodel.ray_param
    vp = synthmodel.vp[-1] # Vp in halfspace
    # vs = synthmodel.vs[-1] # Vs in halfspace
    
    for i in range(1,max_loop):
        freq = i/tot_time
        wavenumber = 2*np.pi*(freq/c) # 2*pi/wavelength
        J = _CalcPropagatorMatrix(synthmodel,wavenumber,calc_from_layer = 0)
        D = (J[0][0]-J[1][0])*(J[2][1]-J[3][1])-(J[0][1]-J[1][1])*(J[2][0]-J[3][0])
        # remember when indexing J that Python is zero-indexed!
        transfer_P_horz[i]=2*c/(D*vp)*(J[3][1]-J[2][1])
        transfer_P_vert[i]=2*c/(D*vp)*(J[3][0]-J[2][0])
        # transfer_S_horz[i]=-c/(D*vs)*(J[0][1]-J[1][1])
        # transfer_S_vert[i]=-c/(D*vs)*(J[0][0]-J[1][0])
    
    # apply a Gaussian filter
    transfer_P_horz[1:max_loop] *= np.exp(-dom_freq**2*T[0]/2)
    transfer_P_vert[1:max_loop] *= np.exp(-dom_freq**2*T[0]/2)
    # transfer_S_horz[1:max_loop] *= np.exp(-dom_freq**2*T[0]/2)
    # transfer_S_vert[1:max_loop] *= np.exp(-dom_freq**2*T[0]/2)
    
    # Make symmetric for IFFT
    transfer_P_horz[-2:-max_loop:-1] = transfer_P_horz[1:max_loop-1]
    transfer_P_vert[-2:-max_loop:-1] = transfer_P_vert[1:max_loop-1]
    # transfer_S_horz[-2:-max_loop:-1] = transfer_S_horz[1:max_loop]
    # transfer_S_vert[-2:-max_loop:-1] = transfer_S_vert[1:max_loop]
    
    P_horz = np.real(np.fft.ifft(transfer_P_horz))/dt
    P_vert = np.real(np.fft.ifft(transfer_P_vert))/dt
    # S_horz = np.real(np.fft.ifft(transfer_S_horz))/dt
    # S_vert = np.real(np.fft.ifft(transfer_S_vert))/dt
    
    # Filter and cut to size
    P_horz = matlab.BpFilt(P_horz,T[0],T[1],dt)[:round(tmax/dt)]
    P_vert = matlab.BpFilt(P_vert,T[0],T[1],dt)[:round(tmax/dt)]
    # S_horz = matlab.BpFilt(S_horz,T[0],T[1],dt)[:round(tmax/dt)]
    # S_vert = matlab.BpFilt(S_vert,T[0],T[1],dt)[:round(tmax/dt)]
    
    # Normalise
    P_max = np.max(np.concatenate([P_horz, P_vert]))
    P_horz = P_horz/P_max
    P_vert = P_vert/P_max
    # S_max = np.max(np.concatenate([S_horz, S_vert]))
    # S_horz = S_horz/S_max
    # S_vert = S_vert/S_max
    
    
    #return synth waveforms
    return BodyWaveform(P_horz,P_vert,dt)
    

def _CalcPropagatorMatrix(synthmodel,wavenumber,calc_from_layer):
    vp = synthmodel.vp[calc_from_layer]
    vs = synthmodel.vs[calc_from_layer]
    rho = synthmodel.rho[calc_from_layer]
    c = 1/synthmodel.ray_param # phase velocity
    gamma = 2*(vs/c)**2
    eta_vp = np.sqrt((c/vp)**2-1)
    eta_vs = np.sqrt((c/vs)**2-1)
    
    if calc_from_layer == synthmodel.vs.size - 1:
        # Calculate propagation through the half space
        return np.array([
                [-2*(vs/vp)**2, 0, (rho*vp**2)**-1, 0],
                [0,c**2*(gamma-1)*(vp**2*eta_vp)**-1,0,(rho*vp**2*eta_vp)**-1],
                [(gamma-1)*(gamma*eta_vs)**-1,0,-(rho*c**2*gamma*eta_vs)**-1,0],
                [0,1,0,(rho*c**2*gamma)**-1],
                ])
    
    else:
        # Calculate propagation through each layer 
        thick = synthmodel.thickness[calc_from_layer]
        P = wavenumber*eta_vp*thick
        Q = wavenumber*eta_vs*thick
        
        a_n = np.array([
                [
                    gamma*np.cos(P)-(gamma-1)*np.cos(Q),
                    1j*((gamma-1)*eta_vp**-1*np.sin(P)+gamma*eta_vs*np.sin(Q)),
                    -(rho*c**2)**-1*(np.cos(P)-np.cos(Q)),
                    1j*(rho*c**2)**-1*(eta_vp**-1*np.sin(P)+eta_vs*np.sin(Q)),
                    ],
                [
                    -1j*(gamma*eta_vp*np.sin(P)+(gamma-1)*eta_vs**-1*np.sin(Q)),
                    -(gamma-1)*np.cos(P)+gamma*np.cos(Q),
                    1j*(rho*c**2)**-1*(eta_vp*np.sin(P)+eta_vs**-1*np.sin(Q)),
                    -(rho*c**2)**-1*(np.cos(P)-np.cos(Q)),
                    ],
                [
                    rho*c**2*gamma*(gamma-1)*(np.cos(P)-np.cos(Q)),
                    1j*rho*c**2*((gamma-1)**2*eta_vp**-1*np.sin(P)+gamma**2*eta_vs*np.sin(Q)),
                    -(gamma-1)*np.cos(P)+gamma*np.cos(Q),
                    1j*((gamma-1)*eta_vp**-1*np.sin(P)+gamma*eta_vs*np.sin(Q)),
                    ],
                [
                    1j*rho*c**2*(gamma**2*eta_vp*np.sin(P)+(gamma-1)**2*eta_vs**-1*np.sin(Q)),
                    rho*c**2*gamma*(gamma-1)*(np.cos(P)-np.cos(Q)),
                    -1j*(gamma*eta_vp*np.sin(P)+(gamma-1)*eta_vs**-1*np.sin(Q)),
                    gamma*np.cos(P)-(gamma-1)*np.cos(Q),
                    ],
                ])
        out= np.matmul(
                _CalcPropagatorMatrix(synthmodel,wavenumber,calc_from_layer+1), 
                a_n)
        return out

# This is a free surface transform, used to rotate from RTZ to P-SV-SH
# while surpressing the effects of the free surface
#   Bostock, M.G., 1998. Mantle stratigraphy and evolution of the Slave province.
#       J. Geophys. Res. 103, 21183–21200.
#   Kennett, B.L.N., 1991. The removal of free surface interactions from 
#       three-component seismograms. Geophys. J. Int. 104, 153–163.
def _RotateToPSV(waveform, vp_surface, vs_surface, ray_param) -> RotBodyWaveform:
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
    return RotBodyWaveform(amp_P = amp_P, amp_SV = amp_SV, dt = waveform.dt)

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
    amp_P = waveform.amp_P
    amp_SV = waveform.amp_SV
    dt = waveform.dt    
    
    # Identify the direct arrival (max peak) - INDEX
    tshift = np.argmax(amp_P) # for Ps
    # tshift = np.argmax(rot.amp_SV) # for Sp
       
    # Filter, crop, and align
    #   Set the window length to 100 ish sec, centred on incident phase by 
    #   padding with zeros (actually 2048, so a power of 2 close to 100s 
    #   ASSUMING dt == 0.05 s !!)
    tot_time = 100
    n_fft = 2**(int(tot_time/dt).bit_length()) # next power of 2
    i_arr = int(n_fft/2)
    amp_P = np.concatenate([ # pad the beginning with zeros so incident phase 
            np.zeros(i_arr - tshift), amp_P]) # at tot_time/2
    amp_P = np.concatenate([ # pad the end with zeros
            amp_P, np.zeros(n_fft - amp_P.size)])
    amp_SV = np.concatenate([ # pad the beginning with zeros
            np.zeros(i_arr - tshift), amp_SV])
    amp_SV = np.concatenate([ # pad the end with zeros
            amp_SV, np.zeros(n_fft - amp_SV.size)])
            
    # Bandpass filter
    amp_P = matlab.BpFilt(amp_P, Ts[0], Ts[1], dt)
    amp_SV = matlab.BpFilt(amp_SV, Ts[0], Ts[1], dt)
    
    # Taper the data
    #    This is written so all positions are INDICES not actual time
    taper_width = 5  # Arbitrary (ish), but works for real data processing
    taper_length = 30 # Arbitrary (ish), but works for real data processing
    # Note: this taper cuts off deep multiples - but these are likely to be
    #       really weak in the real data anyway!
    amp_P = matlab.Taper(amp_P, int(taper_width/dt), i_arr - int(taper_length/2/dt), 
                  i_arr + int(taper_length/2/dt))
    amp_SV = matlab.Taper(amp_SV, int(taper_width/dt), i_arr - int(taper_length/2/dt), 
                  i_arr + int(taper_length/2/dt))
    return RotBodyWaveform(amp_P, amp_SV, dt)
   


def _CalculateRF(waveform) -> RecvFunc:
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
    Daughter = np.concatenate([pad, waveform.amp_SV, pad])
    i_starts = np.arange(0,Daughter.size - (n_samp - 1), int(i_shift*n_samp))
    
    i_t0 = int(waveform.amp_P.size/2)
    incident_win = i_t0+int(n_samp/2)*np.array([-1, 1])
    Parent = waveform.amp_P[incident_win[0]:incident_win[1]]
    
    
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

    # Want time window from incident phase to + 30s
    # and resample with larger dt
    n_jump = int(rf_dt / waveform.dt)
    i_t0 = i_t0 + pad.size 
    RF = np.mean(w_RFs, 0)[i_t0 : i_t0 + int(rf_tmax/waveform.dt) : n_jump]
    RF = RF/np.max(np.abs(RF)) * max_RF

    return RecvFunc(amp = RF, dt = rf_dt)

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

def SynthesiseSWD(model, period) -> SurfaceWaveDisp: # fill this out when you know how
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
    n_ksteps = 25 # assume this is finely spaced enough for our purposes
        # was set to 15 when including findmin # Original code had 200, but this should speed things up
    cr = np.zeros(omega.size)
    
    # Look for the wavenumber (i.e. vel) with minimum secular function value 
    k_lims = np.vstack((omega/cr_max, omega/cr_min))
    mu = model.rho * model.vs**2
    for i_om in range(omega.size):
        cr[i_om] = _FindMinValueSecularFunction(omega[i_om], k_lims[:,i_om],
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

def _FindMinValueSecularFunction(omega, k_lims, n_ksteps, thick, rho, vp, vs, mu):
    #tol_s = 0.1 # This is as original code
    
    # Define vector of possible wavenumbers to try
    wavenumbers = np.linspace(k_lims[0], k_lims[1], n_ksteps)
    

    f1 = 1e-10  # arbitrarily low values so f2 < f1 & f2 < f3 never true for 2 rounds
    f2 = 1e-9
    #k1 = 0 # irrelevant, will not be used unless enter IF statement below
    k2 = 0 # and should be replaced with real values before then
    c = 0
    for i_k in range(-1, -wavenumbers.size-1,-1):
        k3 = wavenumbers[i_k]
        
        f3 = _Secular(k3, omega, thick, mu, rho, vp, vs)
        
        if f2 < f1 and f2 < f3: # if f2 has minimum of the 3 values
            # Find minimum more finely
#            kmin, fmin = matlab.findmin(_Secular, brack = (k3, k2, k1),
#                                 args = (omega, thick, mu, rho, vp, vs))
#            if fmin < tol_s:
#                c = omega/kmin
#                break
            # Doing it properly (as above) is REALLY SLOW
            c = omega/k2
            break
        else:
             f1 = f2  # cycle through wavenumber values
             f2 = f3
             #k1 = k2
             k2 = k3
               
    if c == 0 and n_ksteps <= 250:
        #print(n_ksteps)
        c = _FindMinValueSecularFunction(omega, k_lims, n_ksteps+100, thick,
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
def Mahalanobis(rf_obs,rf_synth, swd_obs,swd_synth, inv_cov) -> float:
    g_m = np.concatenate((rf_synth, swd_synth))
    d_obs = np.concatenate((rf_obs, swd_obs))
    misfit = g_m - d_obs
    
    # N.B. with np.matmul, it prepends or appends an additional dimension 
    # (depending on the position in matmul) if one of the arguments is a vector
    # so don't need to explicitly transpose below
    return np.matmul(np.matmul(misfit,inv_cov),misfit)

# =============================================================================
#       Choose whether or not to accept the change
# =============================================================================
def AcceptFromLikelihood(fit_to_obs_m, fit_to_obs_m0, 
                         cov_m, cov_m0, model_change) -> bool:
    # Form is dependent on which variable changed
    perturb = model_change.which_change
    if perturb == 'Vs' or perturb == 'Dep':
        alpha_m_m0 = np.exp(-(fit_to_obs_m - fit_to_obs_m0)/2)
    elif perturb == 'Birth':
        dv = np.abs(model_change.old_param - model_change.new_param)
        alpha_m_m0 = ((model_change.theta * np.sqrt(2*np.pi) / dv) * 
                      np.exp((dv*dv/(2*model_change.theta**2)) - 
                             (fit_to_obs_m - fit_to_obs_m0)/2))
    elif perturb == 'Death':
        dv = np.abs(model_change.old_param - model_change.new_param)
        alpha_m_m0 = ((dv / (model_change.theta * np.sqrt(2*np.pi))) * 
                      np.exp(-(dv*dv/(2*model_change.theta**2)) - 
                             (fit_to_obs_m - fit_to_obs_m0)/2))
    elif perturb == 'Noise':
        alpha_m_m0 = ((cov_m0.detCovar/cov_m.detCovar) *
                      np.exp(-(fit_to_obs_m - fit_to_obs_m0)/2))


    keep_yn = random.random() # generate random number between 0-1
    return (keep_yn < alpha_m_m0, alpha_m_m0)
    
# =============================================================================
#       Test if model has converged or not
# =============================================================================
def TestConvergence(all_phi, all_alpha, phi_0_std, phi_0_mean, 
                    alpha_0_std, alpha_0_mean, itr, nw) -> int:
    # One test of convergence is that likelihood is and stays high (alpha)
    # and misfit is and stays low (phi)
    # Output: = 0 if not converged yet, = 1 if converged
    converged = 1
    
    # Not really a unique way to define this, so...
    # For each of alpha and phi, want to check that values are stable
    # Check standard deviation of new population vs. original population
    if 0.1 * phi_0_std < all_phi[itr-nw:itr].std: return 0
    if 0.1 * alpha_0_std < all_alpha[itr-nw-1:itr-1].std: return 0
    
    # For each of alpha and phi, want to check values are low/high respectively
    if 0.1 * phi_0_mean < all_phi[itr-nw:itr].mean: return 0
    if 0.1 * all_alpha[itr-nw-1:itr-1].mean < alpha_0_mean: return 0
    
    
    return converged
    
    
    

def SaveModel(fullmodel, deps):
    vs = np.zeros_like(deps)
    for k in range(fullmodel.layertops.size):
        vs[deps>=fullmodel.layertops[k]] = fullmodel.vs[k]
    
    return vs
    


