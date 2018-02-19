# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:39:58 2018

@author: Emily
"""

import unittest
import turtles
from parameterized import parameterized


#def makeTestCases():
#    cases = []
    
    

class TurtleTest(unittest.TestCase):
    
    @parameterized.expand([
        ("Go 100",0, 100, [turtles.Advance(100)]),
        ("Go 50", 0, 50, [turtles.Advance(50)]),
        ("Deg 1", 1, 50, [turtles.Advance(50/3), turtles.Turn(60),
        turtles.Advance(50/3), turtles.Turn(-120), turtles.Advance(50/3),
        turtles.Turn(60),turtles.Advance(50/3)]),
    ])
    def test_koch(self, name, deg, length, expected):          
        self.assertEqual(expected, turtles.kochCurve(deg, length))

    @parameterized.expand([
        ("Go 10",0, 10, [turtles.Advance(10)]),
        ("Go 50", 0, 5, [turtles.Advance(5)]),
        ("Deg 1", 1, 5, [turtles.Advance(5/3), turtles.Turn(60),
        turtles.Advance(5/3), turtles.Turn(-120), turtles.Advance(5/3),
        turtles.Turn(60),turtles.Advance(5/3)]),
    ])
    def test_koch(self, name, deg, length, expected):          
        self.assertEqual(expected, turtles.kochCurve(deg, length))
#    def test_degree_0(self):
#        expected = [turtles.Advance(100)]
#        self.assertEqual(expected,turtles.kochCurve(0,100))
#        
#    def test_koch_2(self):
#        l = turtles.Turn(60)
#        r = turtles.Turn(-120)
#        s = turtles.Advance(180/9)
#        
#        deg_1 = [s,l,s,r,s,l,s]
#                    
#        
#        expected = (deg_1 + [l] + deg_1 + [r] + deg_1 + [l] + deg_1)
#        self.assertEqual(expected,turtles.kochCurve(2,180))

if __name__ == "__main__":
    unittest.main()