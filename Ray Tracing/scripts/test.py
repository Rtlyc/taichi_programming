from math import pi
import taichi as ti
import numpy as np
import sys
from ray import Rays
from hittable import *
from camera import *
from utils import *
from material import *
from random import random

####### Initialization ########
# ti.init(ti.cpu, debug=True, cpu_max_num_threads=8, advanced_optimization=False)
ti.init(ti.cpu)
aspect_ratio = 3.0/2.0
WIDTH = 1200
HEIGHT = int(WIDTH/aspect_ratio)
samples_per_pixel = 100 #500
max_depth = 50
pixels = ti.Vector.field(3, dtype=ti.f32)   
color_pixels = ti.Vector.field(3, dtype=ti.u8)
sample_count = ti.field(dtype=ti.i32)
needs_sample = ti.field(dtype=ti.i32)
ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels, sample_count, needs_sample, color_pixels) 

rays = Rays(WIDTH,HEIGHT)

######## World ##########
def random_scene():
    world = World()
    material_ground = Lambertian(ti.Vector([0.5,0.5,0.5]))
    world.add(Sphere(ti.Vector([0.0,-1000,0.0]), 1000.0, material_ground))

    for a in range(-8,8):
        for b in range(-8,8):
            choose_mat = random()
            center = ti.Vector([a+0.9*random(),0.2,b+0.9*random()])

            if (center-ti.Vector([4,0.2,0]).norm()>0.9):
                sphere_material = Dielectric(1.5)
                if choose_mat<0.8:
                    albedo = ti.Vector([random(), random(), random()])
                    sphere_material = Lambertian(albedo)
                elif choose_mat<0.95:
                    albedo = ti.Vector([random(), random(), random()])*0.5+0.5
                    fuzz = random()*0.5
                    sphere_material = Metal(albedo, fuzz)
                world.add(Sphere(center, 0.2, sphere_material))
    material1 = Dielectric(1.5)
    world.add(Sphere(ti.Vector([0.0,1.0,0.0]), 1.0, material1))
        
    material2 = Lambertian(ti.Vector([0.4,0.2,0.1]))
    world.add(Sphere(ti.Vector([-4.0,1.0,0.0]), 1.0, material2))

    material3 = Metal(ti.Vector([0.7, 0.6, 0.5]), 0.0)
    world.add(Sphere(ti.Vector([4.0,1.0,0.0]), 1.0, material3))
    return world

world = random_scene()
world.finalize()

###### Camera ##########
viewport_height = 2.0
viewport_width = aspect_ratio * viewport_height
focal_length = 1.0
lookfrom = ti.Vector([13.0,2.0,3.0])
lookat = ti.Vector([0.0,0.0,0.0])
vup = ti.Vector([0.0,1.0,0.0])
dist_to_focus = 10.0
aperture = 0.1
cam = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus)
start_attenuation = ti.Vector([1.0, 1.0, 1.0])

origin = ti.Vector([0.0,0.0,0.0])
horizontal = ti.Vector([viewport_width,0,0])
vertical = ti.Vector([0,viewport_height,0])
lower_left_corner = origin-horizontal/2-vertical/2-ti.Vector([0,0,focal_length])

###### Operation #########
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
        hit_anything, normal, p, ind, front_facing = world.hit(ray_origin, ray_direction, 0.001, INFINITY)
        ray_depth -= 1
        rays.depths[x,y] = ray_depth 
        if hit_anything:
            is_reflected, new_origin, new_direction, attenuation = world.materials.scatter(ind, ray_direction, p, normal, front_facing)
            rays.set_ray(x,y,new_origin,new_direction,ray_depth,ray_attenuation*attenuation)
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

