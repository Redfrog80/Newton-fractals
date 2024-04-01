# Evaluates a complex polynomial at the point x, given the roots
from math import prod
from random import random
from sympy import evalf, poly, I, solve, re, im
from sympy.abc import p

# Iterative implementation of newton's method
def newtons_method(x, polynomial, polynomial_deriv, iterations:int, relaxation = 1):
    approx = x
    for iteration in range(iterations):
        deriv = polynomial_deriv(approx)
        if deriv:
            approx = approx - (polynomial(approx)*relaxation)/deriv
        else:
            break
    return complex(approx)

# Returns the squared magnitude of the vector a-b
def magsqr(a, b):
    return (a.real-b.real)**2 + (a.imag-b.imag)**2

# Linear interpolation
def lerp(v0,v1,t):
    return (1-t)*v0+t*v1

# Maps a range of values to another range of values.
def linearMap(a1,a2,b1,b2,x):
    return (b1+(x-a1)*(b2-b1)/(a2-a1))

def randColor():
    return [int(random()*255) for _ in range(3)]