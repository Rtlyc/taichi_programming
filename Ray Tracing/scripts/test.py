from math import pi
import taichi as ti
import numpy as np
import sys
from ray import Rays
from hittable import *
from camera import *
from utils import *

####### Initialization ########
# ti.init(ti.cpu, debug=True, cpu_max_num_threads=8, advanced_optimization=False)
ti.init(ti.cpu)
aspect_ratio = 16.0/9.0
WIDTH = 400
HEIGHT = int(400/aspect_ratio)
samples_per_pixel = 100
max_depth = 50
pixels = ti.Vector.field(3, dtype=ti.f32)   
color_pixels = ti.Vector.field(3, dtype=ti.u8)
sample_count = ti.field(dtype=ti.i32)
needs_sample = ti.field(dtype=ti.i32)
ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels, sample_count, needs_sample, color_pixels) 

rays = Rays(WIDTH,HEIGHT)

######## World ##########
world = World()
world.add(Sphere(ti.Vector([0.0,0.0,-1.0]),0.5))
world.add(Sphere(ti.Vector([0.0,-100.5,-1.0]), 100.0))
world.finalize()

###### Camera ##########
viewport_height = 2.0
viewport_width = aspect_ratio * viewport_height
focal_length = 1.0
cam = Camera()
start_attenuation = ti.Vector([1.0, 1.0, 1.0])

origin = ti.Vector([0.0,0.0,0.0])
horizontal = ti.Vector([viewport_width,0,0])
vertical = ti.Vector([0,viewport_height,0])
lower_left_corner = origin-horizontal/2-vertical/2-ti.Vector([0,0,focal_length])

###### Operation #########
  


@ti.func
def hit_sphere(center, radius, origin, direction):
    oc = origin - center
    a = direction.norm_sqr()
    half_b = oc.dot(direction)
    c = oc.norm_sqr() - radius*radius
    discriminant = half_b*half_b - a*c

    result = -1.0
    if discriminant >= 0:
        result = (-half_b-ti.sqrt(discriminant))/a
    return result 

@ti.kernel  
def initialize():
    for x,y in sample_count:
        sample_count[x,y] = 0
        needs_sample[x,y] = 1

@ti.kernel  
def render()->ti.i32:
    num_completed = 0
    for x,y in pixels:
        # color = ti.Vector.zero(float, 3)
        if sample_count[x,y] == samples_per_pixel:
            continue 
        ray_origin, ray_direction, ray_depth, ray_attenuation = ti.Vector.zero(float,3), ti.Vector.zero(float,3), max_depth, start_attenuation
        if needs_sample[x,y] == 1:
            needs_sample[x,y] = 0
            u = (x+ti.random()) / (WIDTH-1)
            v = (y+ti.random()) / (HEIGHT-1)
            ray_origin, ray_direction = cam.get_ray(u,v)
            rays.set_ray(x, y, ray_origin, ray_direction, max_depth, ray_attenuation)
        else:
            ray_origin, ray_direction, ray_depth, ray_attenuation = rays.get_ray(x,y)

        ######## Intersection ########
        # result = ti.Vector.zero(float,3)
        hit_anything, normal, p = world.hit(ray_origin, ray_direction, 0.001, INFINITY)
        ray_depth -= 1
        rays.depths[x,y] = ray_depth 
        if hit_anything:
            target = p+normal+random_in_unit_sphere() 
            new_origin, new_direction = ray_origin, target-p
            rays.set_ray(x,y,new_origin,new_direction,ray_depth,ray_attenuation*0.5)
            ray_direction = new_direction
            # pixels[x,y] += 0.5*(normal+ti.Vector([1.0,1.0,1.0])) 

        if not hit_anything or ray_depth==0:
            sample_count[x,y] += 1
            needs_sample[x,y] = 1
            unit_direction = ray_direction.normalized()
            t = 0.5 * (unit_direction.y+1.0)
            pixels[x,y] += ray_attenuation*((1.0-t)*ti.Vector([1.0,1.0,1.0])+t*ti.Vector([0.5,0.7,1.0]))
            if sample_count[x,y] == samples_per_pixel:
                num_completed += 1
    return num_completed


@ti.kernel
def finalize():
    for x,y in pixels:
        pixels[x,y] = ti.sqrt(pixels[x,y]/samples_per_pixel)



###### Rendering #########
if __name__ == '__main__':
    initialize()
    render()
    pixels_completed = 0
    pixels_all = WIDTH*HEIGHT
    # count = 0
    while pixels_completed<pixels_all:
        num = render()
        pixels_completed += num 
        # count += 1
        # complete = render()
        # print(complete)
        # pixels_completed += complete
        # print(pixels_completed/pixels_all, file=sys.stderr)
    finalize()

    ti.imwrite(pixels.to_numpy(), 'out.png')
    # print(pixels.to_numpy())
    # print(color_pixels.to_numpy())

