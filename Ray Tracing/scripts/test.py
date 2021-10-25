import taichi as ti
import numpy as np
import time 
from ray import Rays
from hittable import *

####### Initialization ########
ti.init(ti.cpu)
aspect_ratio = 16.0/9.0
WIDTH = 400
HEIGHT = int(400/aspect_ratio)
pixels = ti.Vector.field(3, dtype=ti.u8)   
ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels) 

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

@ti.func
def ray_color(x,y,world):
    ray_origin = rays.origins[x,y]
    ray_direction = rays.directions[x,y]
    result = ti.Vector([1.0,0.0,0.0])

    # temp_record = HitRecord(0.0)
    hit_anything, normal = world.hit(ray_origin, ray_direction, 0, INFINITY)

    # t = hit_sphere(ti.Vector([0.0,0.0,-1.0]),0.5,origin, direction)
    if hit_anything:
        # n = rays.get_at(x,y,t) - ti.Vector([0.0,0.0,-1.0])
        # n = n.normalized()
        result = 0.5*(normal+ti.Vector([1.0,1.0,1.0])) 
        # result = 0.5*ti.Vector([n.x+1.0,n.y+1.0,n.z+1.0])*255.999
    else:
        unit_direction = ray_direction.normalized()
        # print(unit_direction)
        t = 0.5 * (unit_direction.y+1.0)
        result = ((1.0-t)*ti.Vector([1.0,1.0,1.0])+t*ti.Vector([0.5,0.7,1.0]))
    return result*255.999


@ti.kernel  
def initialize():
    for x,y in pixels:
        u = x / (WIDTH-1)
        v = y / (HEIGHT-1)

        rays.origins[x,y] = origin
        rays.directions[x,y] = lower_left_corner+u*horizontal + v*vertical
        pixels[x,y] = ray_color(x,y,world)
        # print("remaining:", WIDTH-x)



###### Rendering #########
initialize()
# last_t = 0
# for i in range(50000):
#     # render()
#     interval = 10
#     if i % interval == 0 and i > 0:
#         print("{:.2f} samples/s".format(interval / (time.time() - last_t)))
#         last_t = time.time()
#         img = pixels.to_numpy() * (1 / (i + 1))
#         img = img / img.mean() * 0.24
#         # gui.set_image(np.sqrt(img))
#         # gui.show()

ti.imwrite(pixels.to_numpy(), 'out.png')
print(pixels.to_numpy())

