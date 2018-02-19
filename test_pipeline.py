# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:13:08 2018

@author: emily
"""

import unittest
from parameterized import parameterized
import pipeline
import numpy as np


class PipelineTest(unittest.TestCase):        
    
    # define some more assertions for our classes
    def assertModelEqual(self, actual, expected):
        # np.testing.assert_array_equal() will actually accept floats etc too
        np.testing.assert_array_almost_equal(actual.vs, expected.vs)
        np.testing.assert_array_almost_equal(actual.all_deps, expected.all_deps)
        np.testing.assert_array_equal(actual.idep, expected.idep) # integers
        self.assertAlmostEqual(actual.std_rf, expected.std_rf)
        self.assertAlmostEqual(actual.lam_rf, expected.lam_rf)
        self.assertAlmostEqual(actual.std_swd, expected.std_swd)
    
    def assertCovarianceMatrixEqual(self, actual, expected):
        np.testing.assert_array_almost_equal(actual.R, expected.R, decimal = 6)
        np.testing.assert_array_almost_equal(actual.Covar, expected.Covar, decimal = 6)
        np.testing.assert_array_almost_equal(actual.invCovar, expected.invCovar, decimal = 6)
        self.assertAlmostEqual(actual.detCovar, expected.detCovar, places = 12)
        
    def assertFullVelModelEqual(self, actual, expected):
        np.testing.assert_array_almost_equal(actual.vs, expected.vs, decimal = 3)
        np.testing.assert_array_almost_equal(actual.vp, expected.vp, decimal = 3)
        np.testing.assert_array_almost_equal(actual.rho, expected.rho, decimal = 3)
        np.testing.assert_array_almost_equal(actual.thickness, 
                                             expected.thickness, decimal = 3)
        np.testing.assert_array_almost_equal(actual.avdep, 
                                             expected.avdep, decimal = 3)
        self.assertAlmostEqual(actual.ray_param, expected.ray_param, places = 4)
        

    
    
    
    #  Test initial model 
    deps = np.append(np.arange(0,60,1), np.arange(60,201,5))
    normdist =  np.random.normal(0,0.5,1000) # Gaussian distribution, std = 0.5
    
    @parameterized.expand([
        ("Seed == 1", normdist, 1,
             pipeline.Model(vs = np.array(1.17), all_deps = deps, idep = np.array(8), 
                            std_rf = 0.5, lam_rf = 0.2, std_swd = 0.15)),       
        ("Seed == 10", normdist, 10, 
             pipeline.Model(vs = 3.36, all_deps = deps, idep = 54, 
                            std_rf = 0.5, lam_rf = 0.2, std_swd = 0.15)),
    ])
       
    def test_InitialModel(self, name, normdist, random_seed, expected):
        model= pipeline.InitialModel(normdist,random_seed)
        self.assertModelEqual(model, expected)
        
    del deps, normdist
    
    
    
    # Test checking prior
    deps = np.append(np.arange(0,60,1), np.arange(60,201,5))
    lims = pipeline.Limits(
            vs = (0.5,5), dep = (0,200), std_rf = (0.005,0.5), 
            lam_rf = (0.05,0.5), std_swd = (0.01,0.2),
            )  
    model = pipeline.Model(
            vs = np.array([1.4,4,5]), all_deps = deps, idep = np.array([0,42,64]), 
            std_rf = 0.15, lam_rf = 0.2, std_swd = 0.15,
            )
    
    @parameterized.expand([
            ("Should work", lims, model, True),
            ("Vs too small", lims, model._replace(vs = np.array([0.4,1.4,4])), False),
            ("Vs too big", lims, model._replace(vs = np.array([0.5,1.4,5.6])), False),
            ("Vs lims", lims._replace(vs = (1.5,5)), model, False), 
            ("Dep ind wrong", lims, model._replace(idep = np.array([-1,42,89])), False),
            ("Dep lims", lims._replace(dep = (50, 100)), model, False),
            ("Std_rf", lims, model._replace(std_rf = 0.51),False),
            ("Std_swd", lims._replace(std_swd = (0,0.1)), model, False),
            ("Lam_rf", lims, model._replace(lam_rf = 1), False),
            ])
    
    def test_CheckPrior(self,name,lims,model,expected):
        self.assertEqual(expected,pipeline.CheckPrior(model,lims))
    
    del deps, lims, model
    
    
    
    
    # Test covariance matrix calculation
    model = pipeline.Model(vs = 0, all_deps = 0, idep = 0, # irrelevant
            std_rf = np.sqrt(0.1), lam_rf = 0.2, std_swd = np.sqrt(0.2),
            )
    rf_obs = pipeline.RecvFunc(amp = np.arange(2), 
                               dt = 0.25)  # length and dt matter
    swd_obs = pipeline.SurfaceWaveDisp(period = np.arange(1), # only length matters
                                       phase_velocity = np.arange(0)) # irrelevant 
    
    @parameterized.expand([
            #("1x1 rf", model, rf_obs, swd_obs, pipeline.CovarianceMatrix(1,1,1)),
            ("2x2 rf, 1 sw", model, rf_obs, swd_obs,
                 pipeline.CovarianceMatrix(
                        R = np.array([[1, 0.928302], # calculated in Matlab (code gave
                             [0.928302, 1]]), # ouput as figure 4 from KL14)
                        Covar = np.array([[0.1, 0.0928302, 0],
                                 [0.0928302, 0.1, 0],
                                 [0,0,0.2]]),
                        invCovar = np.array([[72.33026556,-67.144357,0],
                                    [-67.144357,72.33026556,0],
                                    [0,0,5]]), 
                        detCovar = 0.00027650942305,
                               ),
                         ),
            ("3x3 rf, 4 sw", model, rf_obs._replace(amp = np.arange(3)), 
                 swd_obs._replace(period = np.arange(4)),
                 pipeline.CovarianceMatrix(
                        R = np.array([[1, 0.928302, 0.818653],
                             [0.928302, 1, 0.928302],
                             [0.818653, 0.928302, 1]]),
                        Covar = np.array([[0.1, 0.0928302, 0.0818653,0,0,0,0],
                             [0.0928302, 0.1, 0.0928302,0,0,0,0],
                             [0.0818653, 0.0928302, 0.1,0,0,0,0],
                             [0,0,0,0.2,0,0,0],[0,0,0,0,0.2,0,0],
                             [0,0,0,0,0,0.2,0],[0,0,0,0,0,0,0.2]]),
                        invCovar = np.array([[80.1131353,-97.549094,24.970184,0,0,0,0],
                                    [-97.549094,191.11011086,-97.549094,0,0,0,0],
                                    [24.970184,-97.549094,80.1131353,0,0,0,0],
                                    [0,0,0,5,0,0,0],[0,0,0,0,5,0,0],
                                    [0,0,0,0,0,5,0],[0,0,0,0,0,0,5]]),
                        detCovar = 0.00000002761189,
                               ),
                         ),
            ("change std", model._replace(std_rf = np.sqrt(10), std_swd = 2),
                 rf_obs, swd_obs,
                 pipeline.CovarianceMatrix(
                        R = np.array([[1, 0.928302], # calculated in Matlab (code gave
                             [0.928302, 1]]), # ouput as figure 4 from KL14)
                        Covar = np.array([[10, 9.28302369, 0],
                                 [9.28302369, 10, 0],
                                 [0,0,4]]),
                        invCovar = np.array([[0.7233027,-0.671444,0],
                                    [-0.671444,0.7233027,0],
                                    [0,0,0.25]]), 
                        detCovar = 55.3018846104696,
                               ),
                         ),
            ("change lam",model._replace(lam_rf = 1), rf_obs, swd_obs,
                 pipeline.CovarianceMatrix(
                        R = np.array([[1,0.35326101],
                              [0.35326101,1]]),
                       Covar = np.array([[0.1, 0.035326101, 0],
                                 [0.035326101, 0.1, 0],
                                 [0,0,0.2]]),
                        invCovar = np.array([[11.42587289,-4.0363154497,0],
                                    [-4.0363154497,11.42587289,0],
                                    [0,0,5]]), 
                        detCovar = 0.0017504133111,
                             ),
                         )
            ])
    def test_CalcCovarianceMatrix(self, name, model, rf_obs, swd_obs, expected):
        cov = pipeline.CalcCovarianceMatrix(model, rf_obs, swd_obs)
        self.assertCovarianceMatrixEqual(cov, expected)
    del model, rf_obs, swd_obs

        
    
    # test Synthetics for RF in a few steps...
    # 1. Test the velocity model change
    deps = np.concatenate((np.arange(0,10,0.2), np.arange(10,60,1), np.arange(60,201,5)))
    model = pipeline.Model(
            vs = np.array([4,4.5,4.7,5]), all_deps = deps, idep = np.array([60,80,108,120]), 
            std_rf = 0, lam_rf = 0, std_swd = 0, # irrelevant
            )
    @parameterized.expand([
            ("input model",model,pipeline.SynthModel(
                    vs = np.array([4.,4.5, 4.7,5.]), 
                    vp = np.array([6.9357, 7.875, 8.225, 8.75]),
                    thickness = np.array([30, 40, 60, 0]), 
                    avdep = np.array([15, 50, 100, 160]),
                    rho = np.array([2.9496, 3.4268, 3.4390, 3.4367]),
                    ray_param = 0.0618,
                    )
                ),
            ("vary vels", model._replace(vs = np.array([3,4.2,4.6,5.2])),
             pipeline.SynthModel(
                    vs = np.array([3,4.2,4.6,5.2]),
                    vp = np.array([5.0506,7.3307,8.05,9.1]),
                    thickness = np.array([30, 40, 60, 0]), 
                    avdep = np.array([15, 50, 100, 160]),
                    rho = np.array([2.5426, 3.0674, 3.4329, 3.4406]),
                    ray_param = 0.0618,
                    )
             ),
             ("vary deps", model._replace(idep = np.array([10,50,85,125])),
              pipeline.SynthModel(
                    vs = np.array([4.,4.5, 4.7,5.]), 
                    vp = np.array([6.9357, 7.875, 8.225, 8.75]),
                    thickness = np.array([6., 21.5, 87.5, 0.]), 
                    avdep = np.array([3., 16.75, 71.25, 158.75]),
                    rho = np.array([2.9496, 3.4268, 3.4379, 3.4367]),
                    ray_param = 0.0618,
                    )
             ),
             ("vary both", model._replace(idep = np.array([10,50,110,125]),
                                          vs = np.array([4.,4.499999, 4.7,5.])),
              pipeline.SynthModel(
                    vs = np.array([4.,4.5, 4.7,5.]), 
                    vp = np.array([6.9357, 7.9062, 8.225, 8.75]),
                    thickness = np.array([6., 54., 87.5, 0.]), 
                    avdep = np.array([3., 33., 103.75, 191.25]),
                    rho = np.array([2.9496, 3.2579, 3.4391, 3.4381]),
                    ray_param = 0.0618,
                    )
              ),
            ("Crustal vels only", model._replace(vs = np.array([1.7, 2.6, 3.6, 4.4])),
             pipeline.SynthModel(
                     vs = np.array([1.7, 2.6, 3.6, 4.4]),
                     vp = np.array([3.238876, 4.408495,6.1488126, 7.7179102]),
                     thickness = np.array([30., 40., 60., 0]),
                     avdep = np.array([15., 50., 100., 160.]),
                     rho = np.array([2.2723679, 2.4495676, 2.7493737, 3.193219]),
                     ray_param = 0.0618,
                     )
             ),
             ("Mantle vels only", model._replace(vs = np.array([4.5, 4.6, 4.7, 5.])),
              pipeline.SynthModel(
                     vs = np.array([4.5, 4.6, 4.7, 5.]),
                     vp = np.array([7.875, 8.05, 8.225, 8.75]),
                     thickness = np.array([30., 40., 60., 0]),
                     avdep = np.array([15., 50., 100., 160.]),
                     rho = np.array([3.4268, 3.431978, 3.438984, 3.43669]),
                     ray_param = 0.0618,
                      )
              ),
            
            ])
    def test_MakeFullVelModel(self, name, model, expected):
        fullvel = pipeline._MakeFullModel(model)
        self.assertFullVelModelEqual(fullvel, expected)
    
    
    # Test propagator matrix
    model = pipeline.Model(
            vs = np.array([4,4.6]), all_deps = deps, idep = np.array([60,80]), 
            std_rf = 0, lam_rf = 0, std_swd = 0, # irrelevant
            )
    i_loop = 1
    
    
#    out=np.array([[0.1657, 1j*1.5101, -0.00203124, 1j*-0.00091424],
#                  [1j*0.5551, 0.7971, 1j*-0.00535876, -0.00120837],
#                  [-0.1656, 1j*0.8819, -0.0003388096, 1j*0.0086396689],
#                  [1j*-1.343, -0.0489916, 1j*-0.00099158, 0.00102660]])
    @parameterized.expand([
            ("Moho only", model, i_loop,
             np.array([[-0.6205, 1j*0.3189, 0.0044, 1j*-0.001],
                  [1j*0.115, -1.8854, 1j*-0.0014, 0.0025],
                  [-1.3889, 1j*-0.3424, -0.0018, 1j*-0.0038],
                  [1j*0.526, 0.9235, 1j*0.001, 0.0062]])
             ),
            ("Change wavenumber", model, 349.,
             np.array([[0.1657, 1j*1.5101, -0.00203124, 1j*-0.00091424],
                  [1j*0.5551, 0.7971, 1j*-0.00535876, -0.00120837],
                  [-0.1656, 1j*0.8819, -0.0003388096, 1j*0.0086396689],
                  [1j*-1.343, -0.0489916, 1j*-0.00099158, 0.00102660]])
            ),
            ("Change model", model._replace(vs = np.array([3.5, 4.7])), 349,
             np.array([[0.02145, 1j*-0.2944, 0.005172, 1j*0.000808],
                      [1j*-0.1184,	-1.786, 1j*0.001907, 0.003193],
                      [1.443,	1j*0.07356, 0.002563, 1j*0.001433],
                      [1j*-0.1079, 0.008377, 1j*-0.0007377, -0.00808]])
            ),
            ("Change model slow", model._replace(vs = np.array([1.2, 3.5]),
                                               idep = np.array([23, 35])),
            109, np.array([[-0.41367, 1.0027j, -0.0020859, -0.012886j],
                           [0.1545j, 0.30185, -0.02812j, -0.0009798],
                           [-1.4598, -0.1868j, -0.0026856, -0.03802j],
                           [0.44619j, -0.032956, 0.01043j, 0.011559]])
            ),
            ("Add layers", model._replace(vs = np.array([1.2, 3.5, 4.4, 4.7]),
                                          idep = np.array([23, 35, 60, 100])),
             35, np.array([[-0.14738, 0.36886j, 0.0056826, -0.008675j],
                           [-0.05274j, -0.51406, -0.01805j, 0.00478],
                           [0.15073, -0.11593j, 0.006755, -0.0125j],
                           [0.6842j, 0.07583, 0.006988j, -0.01986]])
                    
                    ),
    
    ])  
    
    def test_PropagatorMat(self, name, model, i_loop, expected):
        wavenumber = i_loop/102.4*0.0618*2*np.pi
        fullmodel = pipeline._MakeFullModel(model)
        Jacobian = pipeline._CalcPropagatorMatrix(fullmodel,wavenumber,0)
        np.testing.assert_array_almost_equal(Jacobian, expected, decimal = 3)
    
    # Test whole synthetics process

#    @parameterized.expand([
#            ("Moho only", model,
#             pipeline.RecvFunc()
#                    ),
#            
#            
#            
#            
#            ])


if __name__ == "__main__":
    unittest.main()