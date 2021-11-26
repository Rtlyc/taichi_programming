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
    return v-2.0*v.dot(n)*n 

@ti.func
def refract(uv,n,etai_over_etat):
    cos_theta = min(-uv.dot(n),1.0)
    r_out_perp = etai_over_etat * (uv+cos_theta*n)  
    r_out_parallel = -ti.sqrt(abs(1.0-r_out_perp.norm_sqr()))*n 
    return r_out_perp + r_out_parallel 