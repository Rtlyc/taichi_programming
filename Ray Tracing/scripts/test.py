from math import pi
import taichi as ti
import numpy as np
import sys
from ray import Rays
from hittable import *
from camera import *

####### Initialization ########
# ti.init(ti.cpu, debug=True, cpu_max_num_threads=8, advanced_optimization=False)
ti.init(ti.cpu)
aspect_ratio = 16.0/9.0
WIDTH = 400
HEIGHT = int(400/aspect_ratio)
samples_per_pixel = 100
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

origin = ti.Vector([0.0,0.0,0.0])
horizontal = ti.Vector([viewport_width,0,0])
vertical = ti.Vector([0,viewport_height,0])
lower_left_corner = origin-horizontal/2-vertical/2-ti.Vector([0,0,focal_length])

###### Operation #########
import random 
@ti.func
def random_in_unit_sphere(small, large):
    cur = ti.Vector([1.0,1.0,1.0])
    while True:
        cur = ti.Vector([random.uniform(small, large), random.uniform(small, large), random.uniform(small, large)])
        if cur.norm_sqr()<1.0:
            break 
    return cur 
            


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

@ti.func
def ray_color(ray_origin, ray_direction, world):
    result = ti.Vector([1.0,0.0,0.0])

    hit_anything, normal, p = world.hit(ray_origin, ray_direction, 0, INFINITY)
    # hit_anything, normal, p = True, ti.Vector([1.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])

    if hit_anything:
        # target = p+normal+random_in_unit_sphere(-1.0,1.0) 
        # rays.origins[x,y] = p 
        # rays.origins[x,y] = target-p
        result = 0.5*(normal+ti.Vector([1.0,1.0,1.0])) 
        # result = 0.5*()
    else:
        unit_direction = ray_direction.normalized()
        t = 0.5 * (unit_direction.y+1.0)
        result = ((1.0-t)*ti.Vector([1.0,1.0,1.0])+t*ti.Vector([0.5,0.7,1.0]))
    return result


@ti.kernel  
def initialize():
    for x,y in sample_count:
        sample_count[x,y] = 0
        needs_sample[x,y] = 1

@ti.kernel  
def render():
    for x,y in pixels:
        color = ti.Vector.zero(float, 3)
        for _ in range(samples_per_pixel):
            u = (x+ti.random()) / (WIDTH-1)
            v = (y+ti.random()) / (HEIGHT-1)
            origin,direction = cam.get_ray(u,v)
            color += ray_color(origin,direction,world)
            # color += ti.Vector([0.2,0.1,0.5])
        color /= samples_per_pixel
        pixels[x,y] = color

        # rays.origins[x,y] = origin
        # rays.directions[x,y] = direction
        # pixels[x,y] += ray_color(origin,direction,world)/samples_per_pixel
        # pixels[x,y] += ti.Vector([120,20,50])


@ti.kernel
def finalize():
    for x,y in pixels:
        pixels[x,y] = pixels[x,y]/samples_per_pixel



###### Rendering #########
if __name__ == '__main__':
    initialize()
    render()
    # pixels_completed = 0
    # pixels_all = WIDTH*HEIGHT
    # count = 0
    # while pixels_completed<pixels_all:
    #     num = render()
    #     pixels_completed += num 
    #     count += 1
        # complete = render()
        # print(complete)
        # pixels_completed += complete
        # print(pixels_completed/pixels_all, file=sys.stderr)
    # finalize()

    ti.imwrite(pixels.to_numpy(), 'out.png')
    # print(pixels.to_numpy())
    # print(color_pixels.to_numpy())

