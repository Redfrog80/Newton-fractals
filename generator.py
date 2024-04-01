from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import numpy as np
from sympy import lambdify, I, roots, diff
from sympy.abc import p
from newtons import *
from random import random


# Author Jeremy Lauber
# This code generates a newton's fractal on the cpu for an arbitrary 
# complex polynomial.
# I wrote this in response to an interesting video I found which dives
# into the more interesting properties of Newton's approximation method

# All of the configuration settings are in the beginning of the file
# This is terribly inefficient because it is running on the CPU

# If you want to do this in real time, look into writing gpu shaders.
# There are also some cpu methods which are far better than mine
# This guy does a fantastic job demonstrating the above:
# https://github.com/alordash/newton-fractal

# Define the polynomial here
polynomial = (p-(1+1*I))*(p-(1-I))*(p-(-1+1*I))*(p-(-1-I))
polynomial_roots = [complex(r) for r in roots(polynomial, multiple=1)]
num_roots = len(polynomial_roots)

# Define the colors of the roots
root_colors = np.array([randColor() for _ in range(num_roots)])

polynomial_deriv = lambdify(p,diff(polynomial, p),"numpy")
polynomial = lambdify(p,polynomial,"numpy")

# Resolution of generated image
resolution_w = 500
resolution_h = 500

# Viewport domain
area_r_min = -3
area_r_max = 3
area_i_min = -3
area_i_max = 3

# Desired depth (number of passes through newtons algorithm)
depth = 3
# Relaxation parameter (influences the step size of newton's 
# algorithm)
relaxation = 1

# Making space for my image
img = np.zeros((resolution_w,resolution_h,3),np.uint8)

def calculate_pixel(x,y):
        # Getting my location based on the viewport domain and resolution.
        val = complex(lerp(area_r_min,area_r_max,x/resolution_w), lerp(area_i_min,area_i_max,y/resolution_h))
        # Getting the newtons approximation
        result = newtons_method(val, polynomial,polynomial_deriv, depth, relaxation)

        c = root_colors[0]
        min_distsqr = magsqr(polynomial_roots[0],result)
        for index in range(1,num_roots):
            distsqr = magsqr(polynomial_roots[index],result)
        
            if (distsqr < min_distsqr):
                min_distsqr = distsqr
                c = root_colors[index]
        
        return (x, y, c)
        

if __name__ == "__main__":
    
    locs = []
    for w in range(resolution_w):
        for h in range(resolution_h):
            locs.append((w,h))
    
    print("Lists made...")
    
    pool = ThreadPool()
    print("pooling...")
    for x,y,c in pool.starmap_async(calculate_pixel,locs,chunksize=100).get():
        img[x][y]=c
    print("pooled")
    pool.close()
    
    
    plt.imshow(img)
    plt.show()
    
    # for w in range(resolution_w):
    #     for h in range(resolution_h):
    #         calculate_pixel((w,h))
    
    # Displaying the resulting image

