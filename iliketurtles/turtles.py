# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:18:20 2018

@author: Emily
"""

import turtle
import typing

class Turn(typing.NamedTuple):
    degrees: float

class Advance(typing.NamedTuple):
    length: float
    
# Curve: List[Turn|Advance]
    
squareCurve = [Advance(100), Turn(90)] * 4

def drawCurve(curve):
    for segment in curve:
        if type(segment) is Turn:
            turtle.left(segment.degrees)
        if type(segment) is Advance:
            turtle.forward(segment.length)
            
def kochCurve(deg, length):
    if not deg:
        return [Advance(length)]
    
    curve = []
    curve.extend(kochCurve(deg - 1, length / 3))
    curve.append(Turn(60))
    curve.extend(kochCurve(deg - 1, length / 3))
    curve.append(Turn(-120))
    curve.extend(kochCurve(deg - 1, length / 3))
    curve.append(Turn(60))
    curve.extend(kochCurve(deg - 1, length / 3))
    
    return curve