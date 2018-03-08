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
    
    def assertModelEquals(self, actual, expected):
        # np.testing.assert_array_equal() will accept floats, ints etc.
        np.testing.assert_array_almost_equal(actual.vs, expected.vs)
        np.testing.assert_array_almost_equal(actual.all_deps, expected.all_deps)
        np.testing.assert_array_almost(actual.idep, expected.idep) # integers
        self.assertAlmostEqual(actual.std_rf, expected.std_rf)
        self.assertAlmostEqual(actual.lam_rf, expected.lam_rf)
        self.assertAlmostEqual(actual.std_swd, expected.std_swd)
    
        
    
    
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
        self.assertModelEquals(model, expected)
        
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
    
    
    


if __name__ == "__main__":
    unittest.main()