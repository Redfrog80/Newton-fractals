import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from newtons import *

# Author Jeremy Lauber
# This code generates a newton's fractal on the cpu for an arbitrary 
# complex polynomial.
# I wrote this in response to an interesting video I found which dives
# into the more interesting properties of Newton's approximation method

# All of the configuration settings are in the beginning of main
# This is terribly inefficient because it is running on the CPU (and well, Python)

# If you want to do this in real time, look into writing gpu shaders.
# There are also some cpu methods which are far better than mine
# This guy does a fantastic job demonstrating the above:
# https://github.com/alordash/newton-fractal

def main():
    # Define the polynomial here
    polynomial_roots = [1-1j,4+2j,-2-5j,-1+5j]
    # Define the colors of the roots
    root_colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0)]

    # Resolution of generated image
    resolution_w = 1000
    resolution_h = 1000
    
    # Viewport domain
    area_r_min = -10
    area_r_max = 10
    area_i_min = -10
    area_i_max = 10
    
    # Desired depth (number of passes through newtons algorithm)
    depth = 5
    
    # Making space for my image
    img = np.zeros((resolution_w,resolution_h,3),np.uint8)
    
    # List for my threads
    workers = []
    
    # Function for my worker threads
    # The spawned threads will be doing the heavy lifting
    # This function generates the pixels for every row belonging to the column w.
    def inner_loop_thread(w):
        for h in range(0,resolution_h):
            # Getting my starting location based on the viewport domain and resolution.
            x = complex(lerp(area_r_min,area_r_max,w/resolution_w), lerp(area_i_min,area_i_max,h/resolution_h))
            # Getting the newtons approximation
            result = newtons_method(x, polynomial_roots, depth)
            # Getting the color of the root which the result is closest to.
            closest_root_index = 0
            min_distsqr = magsqr(polynomial_roots[closest_root_index],result)
            for index in range(len(polynomial_roots)):
                distsqr = magsqr(polynomial_roots[index],result)
                if (distsqr < min_distsqr):
                    closest_root_index = index
                    min_distsqr = distsqr
            c = root_colors[closest_root_index]
            # Updating the pixel with the approprate color.
            img[w][h] = c
    
    # Spawns worker threads to process every column
    for w in range(resolution_w):
        worker = Thread(target=inner_loop_thread, args=(w,))
        worker.start()
        workers.append(worker)
    
    # Waiting for the threads to finish their jobs
    for worker in workers:
        worker.join()
    
    # Displaying the resulting image
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
