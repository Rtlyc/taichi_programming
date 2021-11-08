from taichi import ti 
import math 

## Constants
INFINITY = float('inf')
PI = 3.141592653589793238

## Utility Functions
def degrees_to_radians(degrees):
    return degrees*PI/180.0

@ti.func
def random_in_unit_sphere():
    theta = ti.random()*PI*2.0
    v = ti.random()
    phi = ti.acos(2.0*v-1.0)
    r = ti.random()**(1/3)
    ret = ti.Vector([r*ti.sin(phi)*ti.cos(theta), r*ti.sin(phi)*ti.sin(theta), r*ti.cos(phi)])
    return ret
          
@ti.func
def near_zero(vec):
    s = 1e-8
    return abs(vec.x)<s and abs(vec.y)<s and abs(vec.z)<s 

@ti.func 
def reflect(v,n):
    return v-2*v.dot(n)*n 