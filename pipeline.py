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
from scipy.signal import butter, detrend
from scipy.optimize import brent as findmin
from spectrum.mtm import dpss 
# install as conda config --add channels conda-forge; conda install spectrum

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
    amp_P: np.array # horizontal (radial) energy
    amp_SV: np.array # vertical energy
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
    change_covar: int
    
class SynthModel(typing.NamedTuple):
    vs: np.array
    vp: np.array
    rho: np.array
    thickness: np.array
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
    model = InitialModel(rf_obs,random_seed)
    if not CheckPrior(model, lims):
        return "Starting model does not fit prior distribution"
    
    # Calculate covariance matrix (and inverse, and determinant)
    cov_m0 = CalcCovarianceMatrix(model,rf_obs,swd_obs)
    
    # Calculate synthetic data
        # First, need to define the Vp and rho
    fullmodel = MakeFullModel(model)
    rf_synth_m0 = SynthesiseRF(fullmodel) # forward model the RF data
    swd_synth_m0 = SynthesiseSWD(fullmodel, swd_obs) # forward model the SWD data
    
    # Calculate fit
    fit_to_obs_m0 = Mahalanobis(
            rf_obs.amp, rf_synth_m0.amp, # RecvFunc
            swd_obs.c, swd_synth_m0.c,  # SurfaceWaveDisp
            cov_m0.invCovar,             # CovarianceMatrix
            )
    
    converged = 0  # variable defines # of iterations since convergence
    
    
    # =========================================================================
    #       Iterate by Reverse Jump Markov Chain Monte Carlo
    # =========================================================================    
    for itr in range(max_iter):
        
        # Generate new model by perturbing the old model
        new_model, changes_model = Mutate(model,itr)
        
        if not CheckPrior(new_model): # check if mutated model compatible with prior distr.
            continue  # if not, continue to next iteration
        
        if changes_model.change_covar: 
            # only need to recalculate covariance matrix if changed hyperparameter
            cov_m = CalcCovarianceMatrix(model,rf_obs,swd_obs)
            rf_synth_m = rf_synth_m0
            swd_synth_m = swd_synth_m0
        else:
            cov_m = cov_m0
            rf_synth_m = SynthesiseRF(new_model) # forward model the RF data
            swd_synth_m = SynthesiseSWD(swd_obs) # forward model the SWD data
        
        # Calculate fit
        fit_to_obs_m = Mahalanobis(
                rf_obs,rf_synth_m, # RecvFunc
                swd_obs,swd_synth_m,  # SurfaceWaveDisp
                cov_m,             # CovarianceMatrix
                )
        
        # Calculate probability of keeping mutation
        keep_yn = AcceptFromLikelihood(
                fit_to_obs_m, fit_to_obs_m0, # float
                cov_m, cov_m0, # CovarianceMatrix               
                )
        if keep_yn: # should be aiming to keep about 40% of perturbations
            model = new_model
            cov_m0 = cov_m
            rf_synth_m0 = rf_synth_m
            swd_synth_m0 = swd_synth_m
            
            # Test if have converged
            if not converged:
                converged = TestConvergence()
        
        # Save every 500th iteration after convergence (KL14)
        if converged:
            converged = converged + 1
            if not converged % 500: 
                SaveModel(model)
            
        
    return model


 
    
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
    vs = np.array(round(random.uniform(0.5,5.5),2))   # arbitrary
    all_deps = np.append(np.arange(0,60,1), np.arange(60,201,5))
    idep = np.array(random.randrange(0,len(all_deps)))  # arbitrary
    std_rf = round(np.std(rf_obs,ddof=1),1) # start off assuming all data is noise (KL14)
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
        _InBounds(model.idep, (0,model.all_deps.size)) and
        _InBounds(model.vs, limits.vs) and
        _InBounds(model.all_deps[model.idep], limits.dep) and
        _InBounds(model.std_rf, limits.std_rf) and
        _InBounds(model.lam_rf, limits.lam_rf) and
        _InBounds(model.std_swd, limits.std_swd) and
        model.vs.size == model.idep.size
    )
    
def _InBounds(value, limit):
    min, max = limit
    if type(value) == np.ndarray:
        return all(np.logical_and(min <= value, value <= max))
    else: 
        return min <= value <= max

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
        rho_scaling = np.zeros_like(model.vs)
        del_rho = np.zeros_like(model.vs)
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
    
    return SynthModel(model.vs, vp, rho, thickness, avdep, ray_param)  


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
    
    detc=np.linalg.det(covar)
    invc=np.linalg.inv(covar)
    
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
    if model.dep.size==1: # and don't kill your only layer...
        perturbs=('Vs','Dep','Birth','Hyperparameter')
    elif itr <= 1000*(model.dep.size-1)*(model.dep.size): # as KL14
        perturbs=('Vs','Dep','Death','Hyperparameter')
    else: 
        perturbs=('Vs','Dep','Birth','Death','Hyperparameter')
    perturb = perturbs[random.randint(0,len(perturbs)-1)]
    
    if perturb=='Vs':          # Change Vs in a layer
        vs=model.vs
        i_vs = random.randint(0,vs.size-1)
        theta =_GetStdForGaussian(perturb)
        vs[i_vs] = np.random.normal(vs[i_vs],theta)
        changes_model = ModelChange(theta,model.vs[i_vs],vs[i_vs],change_covar=0)
        new_model = model._replace(vs=vs)
        
    elif perturb=='Dep':       # Change Depth of one node
        # Remember, we assume possible model points follow the following sequence
        # 1 km spacing from 0-60 km; 5 km spacing from 60-200 km        
        idep = model.idep
        i_id = random.randint(0,idep.size-1)
        # perturb around index in alldeps array
        theta =_GetStdForGaussian(perturb)
        idep[i_id] = round(np.random.normal(idep[i_id],theta))
        changes_model = ModelChange(theta,model.idep[i_id],idep[i_id],change_covar=0)
        new_model = model._replace(idep=idep)
        
    elif perturb=='Birth':     # Layer Birth
        # choose unoccupied position, unused_d (index i_d), with uniform probability
        idep = model.idep
        vs = model.vs
        unused_idep = [idx for idx,val in enumerate(model.all_deps) 
                if idx not in idep]
        unused_d = unused_idep[random.randint(0,len(unused_idep))]
        i_d = np.where(model.all_deps==unused_d)
        theta = _GetStdForGaussian(perturb)
        # identify index of closest nucleus for old Vs value
        i_old = abs(model.all_deps[idep]-unused_d).argmin()
        # Make new array of depth indices, and identify new value index, i_new
        idep = np.sort(np.append(idep,i_d))
        i_new = np.where(idep==i_d)
        vs = np.insert(vs,i_new,np.random.normal(vs[i_old],theta))
        changes_model = ModelChange(theta,model.vs[i_old],vs[i_new],change_covar=0)
        new_model=model._replace(idep=idep,vs=vs)
        
    elif perturb=='Death':     # Layer Death
        # choose occupied position, i_id, with uniform probability
        idep = model.idep
        vs = model.vs
        i_id = random.randint(0,idep.size-1)
        theta = _GetStdForGaussian(perturb)
        idep = np.delete(idep,i_id)
        vs = np.delete(vs,i_id)
        # identify index of closest nucleus for new Vs value
        i_new = abs(model.all_deps[idep]-model.all_deps[model.idep[i_id]])  
        changes_model = ModelChange(theta,model.vs[i_id],vs[i_new],change_covar=0)
        new_model = model._replace(idep=idep,vs=vs)
        
    else: # perturb=='Hyperparameter', Change a hyperparameter
        hyperparams = ['Std_RF','Lam_RF','Std_SW']
        hyperparam = hyperparams[random.randint(0,len(hyperparams)-1)]
        theta = _GetStdForGaussian(hyperparam)
        
        if hyperparam == 'Std_RF':
            old = model.std_rf
            new = np.random.normal(old,theta)
            new_model = model._replace(std_rf = new)
        elif hyperparam == 'Lam_RF':
            old = model.lam_rf
            new = np.random.normal(old,theta) 
            new_model = model._replace(lam_rf = new)
        elif hyperparam == 'Std_SW':
            old = model.std_swd
            new = np.random.normal(old.theta)
            new_model = model._replace(std_swd = new)
            
        changes_model = ModelChange(theta, old, new, change_covar = 1)
            
    return new_model, changes_model

def _GetStdForGaussian(perturb) -> float:
    if perturb=='Vs':          # Change Vs in a layer
        theta=0.05             # B&c12's theta1
    elif perturb=='Dep':       # Change Depth of one node
        theta=2                # B&c12's theta3
    elif perturb=='Birth' or perturb=='Death': # Layer Birth/Death
        theta=1                # B&c12's theta2 for new velocity
    elif perturb=='Std_RF':    # Change hyperparameter - std_rf
        theta=1e-3             # B&c12's theta_h_j
    elif perturb=='Std_SWD':   # Change hyperparameter - std_swd
        theta=1e-2             # B&c12's theta_h_j
    elif perturb=='Lam_RF':    # Change hyperparameter - lam_swd
        theta=1e-2
    
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
    wv = _PrepWaveform(wv, Ts = [1, 50]) 
    
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
    P_horz = BpFilt(P_horz,T[0],T[1],dt)[:round(tmax/dt)]
    P_vert = BpFilt(P_vert,T[0],T[1],dt)[:round(tmax/dt)]
    # S_horz = BpFilt(S_horz,T[0],T[1],dt)[:round(tmax/dt)]
    # S_vert = BpFilt(S_vert,T[0],T[1],dt)[:round(tmax/dt)]
    
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
    amp_P = np.concatenate([ # pad the beginning with zeros
            np.zeros(int(tot_time/2 / dt - tshift)), amp_P])
    amp_P = np.concatenate([ # pad the end with zeros
            amp_P, np.zeros(n_fft - amp_P.size + 1)])
    amp_SV = np.concatenate([ # pad the beginning with zeros
            np.zeros(int(tot_time/2 / dt - tshift)), amp_SV])
    amp_SV = np.concatenate([ # pad the end with zeros
            amp_SV, np.zeros(n_fft - amp_SV.size + 1)])
            
    # Bandpass filter
    amp_P = BpFilt(amp_P, Ts[0], Ts[1], dt)
    amp_SV = BpFilt(amp_SV, Ts[0], Ts[1], dt)
    
    # Taper the data
    #    This is written so all positions are INDICES not actual time
    taper_width = 5  # Arbitrary (ish), but works for real data processing
    taper_length = 30 # Arbitrary (ish), but works for real data processing
    amp_P = Taper(amp_P, int(taper_width/dt), tshift - int(taper_length/2/dt),
                  tshift + int(taper_length/2/dt))
    amp_SV = Taper(amp_SV, int(taper_width/dt), tshift - int(taper_length/2/dt),
                  tshift + int(taper_length/2/dt))
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
    num_windows = 7
    n_samp = 2**(int(win_length / waveform.dt).bit_length()) # for FFT - 1024 samples
    tapers = dpss(n_samp, int((num_windows+1)/2))[0]
    
    # We have already required (in PrepWaveform) that tot_time = 100
        
    # Pad with zeros to give wiggle room on large ETMTM windows
    pad = np.zeros(int(np.ceil(n_samp*i_shift)))
    Daughter = np.concatenate([pad, waveform.amp_SV, pad])
    i_starts = np.arange(0,Daughter.size - (n_samp - 1), i_shift*n_samp)
    
    i_t0 = int(waveform.amp_P.size/2)
    incident_win = i_t0+int(n_samp/2)*np.array([-1, 1])
    Parent = waveform.amp_P[incident_win[0]:incident_win[1]]
    
    
    P_fft = _ETMTMSumFFT(tapers, Parent, Parent, num_windows)
    norm_by = np.max(np.abs(np.real(np.fft.ifft(P_fft/(P_fft + 10)))))
    w_RFs = np.zeros((num_windows, Daughter.size))
    max_RF = 0
    
    for i in range(num_windows):
        # Isolate and taper the daughter window
        Daughter_win = Taper( Daughter[i_starts[i] : i_starts[i] + n_samp], 
                             int(n_samp/5), 0, n_samp)
        w_D_fft = _ETMTMSumFFT(tapers, Parent, Daughter_win, num_windows)
        w_RF = np.real(np.fft.ifft(w_D_fft/(P_fft + 10)))/norm_by # Arbitrary water level = 10
        w_RFs[i, i_starts[i]:i_starts[i]+n_samp] = w_RF
        max_RF = np.max(max_RF, np.max(np.abs(w_RF)))

    # Want time window from incident phase to + 30s
    # and resample with larger dt
    n_jump = int(rf_dt / waveform.dt)
    i_t0 = i_t0 + pad.size 
    RF = np.mean(w_RFs, 0)[i_t0 : i_t0 + rf_tmax/waveform.dt : n_jump]

    return RecvFunc(amp = RF, dt = rf_dt)

#  Extended time multi-taper method of deconvolution:
#   Helffrich, G., 2006. Extended-time multitaper frequency domain 
#       cross-correlation receiver-function estimation. Bull. Seismol. Soc. Am.
#        96, 344–347.
def _ETMTMSumFFT(slepians, data_1, data_2, num_window):
    which_window = num_window-1 # as zero-indexed
    if which_window == 0:
        return (np.conj(np.fft.fft(slepians[:,which_window]*data_1)) * 
                np.fft.fft(slepians[:,which_window]*data_2))
    else:
        return (np.conj(np.fft.fft(slepians[:,which_window]*data_1)) * 
                np.fft.fft(slepians[:,which_window]*data_2) + 
                _ETMTMSumFFT(slepians, data_1, data_2, num_window-1))


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

def SynthesiseSWD(model, swd_obs) -> SurfaceWaveDisp: # fill this out when you know how
    freq = 1/swd_obs.period
    
    if model.vs.size == 1:
        cr = _CalcRaylPhaseVelInHalfSpace(model.vp[0], model.vs[0])
        return np.ones(freq.size)*cr  # no freq. dependence in homogeneous half space
    
    # Set bounds of tested Rayleigh wave phase velocity
    cr_max = np.max(model.vs)
    cr_min = 0.98 * _CalcRaylPhaseVelInHalfSpace(np.min(model.vp), 
                                                 np.min(model.vs))
    omega = 2*np.pi*freq 
    n_ksteps = 25 # Original code had 200, but this should speed things up
    
    # Look for the wavenumber (i.e. vel) with minimum secular function value 
    k_lims = np.vstack((omega/cr_max, omega/cr_min))
    for i_om in range(omega.size):
        cr[i_om] = _FindMinValueSecularFunction(omega[i_om], k_lims[:,i_om],
          n_ksteps, model.thickness, model.rho, model.vp, model.vs)
      
    return SurfaceWaveDisp(period = swd_obs.period, c = cr)

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

def _FindMinValueSecularFunction(omega, k_lims, n_ksteps, thick, rho, vp, vs):
    tol_s = 0.1 # This is as original code
    
    # Define vector of possible wavenumbers to try
    wavenumbers = np.linspace(k_lims[0], k_lims[1], n_ksteps)

    
    # Set up common variables for _Secular
    #   these will either be 1 x vs.length OR wavenumbers.length x vs.length
    wavenumbers = np.reshape(wavenumbers, (wavenumbers.size,1))
    mu = _m3D(rho * vs**2) # 1 x 1 x nl
    all_nu_s = np.sqrt(wavenumbers**2 - omega**2/vs**2) # nk x nl
    all_gamma_s = np.abs(all_nu_s/wavenumbers) # nk x nl
    all_nu_p = np.sqrt(wavenumbers**2 - omega**2/vp**2) # nk x nl
    all_gamma_p = np.abs(all_nu_p/wavenumbers) # nk x nl
    all_chi = 2*wavenumbers - all_nu_s/wavenumbers**2  # nk x nl
    thick = _m3D(thick)
    

    f1 = 1e-10  # arbitrarily low values so f2 < f1 & f2 < f3 never true for 2 rounds
    f2 = 1e-9
    k1 = 0 # irrelevant, will not be used unless enter IF statement below
    k2 = 0 # and should be replaced with real values before then
    for i_k in range(-1, -wavenumbers.size-1,-1):
        nu_s = _m3D(all_nu_s[i_k])
        gamma_s = _m3D(all_gamma_s[i_k])
        nu_p = _m3D(all_nu_p[i_k])
        gamma_p = _m3D(all_gamma_p[i_k])
        chi = _m3D(all_chi[i_k])
        k3 = wavenumbers[i_k]
        
        f3 = _Secular(k3, omega, thick, mu, nu_s, gamma_s, nu_p, gamma_p, chi)
        
        if f2 < f1 and f2 < f3: # if f2 has minimum of the 3 values
            # Find minimum more finely
            fmin, kmin = findmin(_Secular, brack = (k3, k1),
                                 args = (omega, thick, rho, vp, vs),
                                         full_output = True)[0:1]
            if fmin < tol_s:
                c = omega/kmin
                break
        else:
             f1 = f2  # cycle through wavenumber values
             f2 = f3
             k1 = k2
             k2 = k3
               
    if c == 0 and n_ksteps < 250:
        c = _FindMinValueSecularFunction(omega, k_lims, n_ksteps+25, thick, 
                                         thick, mu, nu_s[i_k], gamma_s[i_k], 
                                         nu_p[i_k], gamma_p[i_k], chi[i_k])
        
    return c

def _m3D(x):
    return np.reshape(x,(1,1,x.size))
    
def _Secular(k, om, thick, mu, nu_s, gamma_s, nu_p, gamma_p, chi):
    # Ok.  This is written in kind of a weird way (following the MATLAB code)
    # but that's because it minimises repetition of any multiplication etc - 
    # so it should be faster to run, even if it seems like an odd way of doing 
    # things...
    
    # First, calculate the E and Lambda matrices (up-going and 
    # down-going matrices) for the P-SV case.
    ar1 = np.ones((1,1,mu.size))
    ar0 = np.zeros((1,1,mu.size))
    muchi = mu*chi
    munup = 2*mu*nu_p
    munus = 2*mu*nu_s
    E11 = np.vstack((np.hstack((-ar1,gamma_s)),np.hstack((-gamma_p, ar1))))
    E12 = np.vstack((np.hstack((-ar1, gamma_s)), np.hstack((gamma_p, -ar1))))
    E21 = np.vstack((np.hstack((munup, -muchi)), np.hstack((muchi, -munus))))
    E22 = np.vstack((np.hstack((-munup, muchi)), np.hstack((muchi, -munus))))
    du  = np.vstack((np.hstack((np.exp(-nu_p)*thick, ar0)),
                     np.hstack((ar0, np.exp(-nu_s)*thick))))
    X = np.zeros(4,4,mu.size-1)
    for iv in range(mu.size - 2):
        A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                       np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
        B = np.vstack((np.hstack((E11[:,:,iv],-E12[:,:,iv+1])),
                       np.hstack((E21[:,:,iv], -E22[:,:,iv+1]))))
        L = np.vstack((np.hstack((du[:,:,iv],np.zeros((2,2)))),
                       np.hstack(np.zeros((2,2)),du[:,:,iv+1])))
        X[:,:,iv] = matlab.mldivide(A,np.matmul(B,L))
    
    # And the deepest layer (above the halfspace)
    A = np.vstack((np.hstack((E11[:,:,iv+1],-E12[:,:,iv])),
                   np.hstack((E21[:,:,iv+1], -E22[:,:,iv]))))
    B = np.vstack((E11[:,:,iv],E21[:,:,iv]))
    L = du[:,:,iv]
    X[:,0:1,-1] = matlab.mldivide(A, np.matmul(B,L))
    
    
    #  Calculate the modified R/T coefficients
    Td = np.zeros(2,2,mu.size-1)
    Rd = np.zeros(2,2,mu.size-1)
    
    Td[:,:,mu.size-1] = X[0:1,0:1,mu.size-1] # The deepest layer above HS
    Rd[:,:,mu.size-1] = X[2:3,0:1,mu.size-1]
    for iv in range(mu.size-2,0,-1):
        Td[:,:,iv] = matlab.mldivide(
                np.matmul(
                        np.array([[1-X[2,2], -X[2,3]],[-X[3,2], 1-X[3,3]]]), 
                        Rd[:,:,iv+1]
                        ),
                X[0:1,0:1,iv]
                )
        Rd[:,:,iv] = X[2:3,1:2] + np.matmul(np.matmul(X[2:3,2:3],Rd[:,:,iv+1]),
                                              Td[:,:,iv])
    
    
    # And finally, calculate the absolute value of the secular function
    d = (np.abs(np.det(E21[:,:,0] + 
                      np.matmul(np.matmul(E22[:,:,0],du[:,:,0]), Rd[:,:,0])))
                    /(nu_s[0,0,0]*nu_p[0,0,0]*mu[0,0,0]**2))
    
    

    return d




# =============================================================================
#       Find the misfit (scaled by covariance matrix)
# =============================================================================
def Mahalanobis(rf_obs,rf_synth, swd_obs,swd_synth, inv_cov) -> float:
    g_m = np.concatenate(rf_synth, swd_synth)
    d_obs = np.concatenate(rf_obs, swd_obs)
    misfit = g_m - d_obs
    
    # N.B. with np.matmul, it prepends or appends an additional dimension 
    # (depending on the position in matmul) if one of the arguments is a vector
    # so don't need to explicitly transpose below
    return np.matmul(np.matmul(misfit,inv_cov),misfit)

# =============================================================================
#       Choose whether or not to accept the change
# =============================================================================
def AcceptFromLikelihood() -> bool:
    pass


# =============================================================================
#       Test if model has converged or not
# =============================================================================
def TestConvergence() -> int:
    pass

def SaveModel():
    pass


# =============================================================================
#   Other generally useful functions
# =============================================================================

# Butterworth bandpass filter edited slightly from 
#   http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass

def ButterBandpass(short_T, long_T, dt, order=2):
     nyq = 0.5 / dt
     low = 1 / (nyq*long_T)
     high = 1 / (nyq*short_T)
     b, a = butter(order, [low, high], btype='bandpass')
     return b, a

def BpFilt(data, short_T, long_T, dt, order=2):
     b, a = ButterBandpass(short_T, long_T, dt, order=order)
     data = data - np.mean(data)
     y = matlab.filtfilt(b, a, data) # Note: filtfilt rather than lfilt applies a 
                               # forward and backward filter, so no phase shift
     y = detrend(y)
     return y
 
def Taper(data, i_taper_width, i_taper_start, i_taper_end):
    # This will apply a (modified) Hann window to data (a np.array), such that
    # there is a cosine taper (i_taper_width points long) that is equal to 1 
    # between i_taper_start (index) and i_taper_end (index)
    
    taper = np.concatenate([np.zeros(i_taper_start - i_taper_width),
                            np.insert(np.hanning(2*i_taper_width),i_taper_width,
                                      np.ones(i_taper_end - i_taper_start)),
                            np.zeros(i_taper_end - i_taper_width*2 - i_taper_start)])
    
    return data * taper


