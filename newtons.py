# Evaluates a complex polynomial at the point x, given the roots
def eval_poly(x:complex, roots:list[complex]):
    return sum([(x-roots[i])**(i+1) for i in range(len(roots))])

# Evaluates the derivative at the point x for a complex polynomial, given the roots
def eval_poly_deriv(x:complex, roots:list[complex]):
    return sum([(i+1)*(x-roots[i])**(i) for i in range(len(roots)) if i])

# Iterative implementation of newton's method
def newtons_method(x:complex, roots:list[complex], iterations:int):
    approx = x
    for iteration in range(iterations):
        approx = complex(approx - eval_poly(approx, roots)/eval_poly_deriv(approx,roots))
    return approx

# Returns the squared magnitude of the vector a-b
def magsqr(a: complex, b: complex):
    return (a.real-b.real)**2 + (a.imag-b.imag)**2

# Linear interpolation
def lerp(v0,v1,t):
    return (1-t)*v0+t*v1

# Maps a range of values to another range of values.
def linearMap(a1,a2,b1,b2,x):
    return (b1+(x-a1)*(b2-b1)/(a2-a1))