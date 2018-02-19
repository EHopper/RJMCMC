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
from scipy.signal import butter, lfilter

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

class SurfaceWaveDisp(typing.NamedTuple):
    period: np.array    # dominant period of dispersion measurement, seconds
    phase_velocity: np.array  # phase velocity in km/s
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
    rf_synth_m0 = SynthesiseRF(model) # forward model the RF data
    swd_synth_m0 = SynthesiseSWD(swd_obs) # forward model the SWD data
    
    # Calculate fit
    fit_to_obs_m0 = Mahalanobis(
            rf_obs,rf_synth_m0, # RecvFunc
            swd_obs,swd_synth_m0,  # SurfaceWaveDisp
            cov_m0,             # CovarianceMatrix
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
def SynthesiseRF(model) -> RecvFunc:
    # First, need to define the Vp and rho
    synthmodel = _MakeFullModel(model)
    
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
    transfer_P_horz = np.zeros(n_fft)
    transfer_P_vert = np.zeros(n_fft)
    # transfer_S_horz = np.zeros(n_fft)
    # transfer_S_vert = np.zeros(n_fft)
    c=synthmodel.ray_param
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
    transfer_P_horz[1:max_loop] *= np.exp(dom_freq**2*T[0]/2)
    transfer_P_vert[1:max_loop] *= np.exp(dom_freq**2*T[0]/2)
    # transfer_S_horz[1:max_loop] *= np.exp(dom_freq**2*T[0]/2)
    # transfer_S_vert[1:max_loop] *= np.exp(dom_freq**2*T[0]/2)
    
    # Make symmetric for IFFT
    transfer_P_horz[-2:-max_loop:-1] = transfer_P_horz[1:max_loop]
    transfer_P_vert[-2:-max_loop:-1] = transfer_P_vert[1:max_loop]
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
    
    #return synth_rf
    return BodyWaveform(P_horz,P_vert,dt)
    

def _MakeFullModel(model) -> SynthModel:
    # Define Vp, Rho, Thickness
    dep = model.all_deps[model.idep]
    vp = np.zeros_like(model.vs) # in km/s
    rho = np.zeros_like(model.vs) # in Mg/m^3 (or g/cm^3)
    
            
    #  Now calculate the thickness and average depth of the layers
    thickness = np.zeros_like(model.vs)
    dep = np.insert(dep,0,-dep[0])
    thickness[:-1] = (dep[1:-1]-dep[:-2])/2 + (dep[2:]-dep[1:-1])/2
    t1 = np.append(0,thickness)
    t1[-1] = t1[-2]
    avdep = np.cumsum(t1[:-1])+t1[1:]/2
    
    #   1.  Assume that any values of Vs under 4.5 km/s are crustal
    Moho_ind = np.where(model.vs >= 4.5)[0]
    if Moho_ind.size:
        Moho_ind = Moho_ind[0]
    else:
        Moho_ind = np.array([model.vs.size])
    
    crust_inds = np.arange(Moho_ind)
        
    # Equations from Brocher et al., 2005, BSSA (doi: 10.1785/0120050077)
    if crust_inds.size:
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



# =============================================================================
#       Synthesise Surface Wave Dispersion measurements
# =============================================================================
#  
def SynthesiseSWD(swd_obs) -> SurfaceWaveDisp: # fill this out when you know how
    return swd_obs

def Mahalanobis() -> float:
    pass


def AcceptFromLikelihood() -> bool:
    pass

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
     b, a = butter(order, [low, high], btype='band')
     return b, a

def BpFilt(data, short_T, long_T, dt, order=2):
     b, a = ButterBandpass(short_T, long_T, dt, order=order)
     y = lfilter(b, a, data)
     return y


