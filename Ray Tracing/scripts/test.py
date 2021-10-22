import taichi as ti
import numpy as np
import time 
from ray import Rays

####### Initialization ########
ti.init(ti.cpu)
aspect_ratio = 16.0/9.0
WIDTH = 400
HEIGHT = int(400/aspect_ratio)
pixels = ti.Vector.field(3, dtype=ti.u8)   
ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels) 

rays = Rays(WIDTH,HEIGHT)
###### Camera ##########
viewport_height = 2.0
viewport_width = aspect_ratio * viewport_height
focal_length = 1.0

origin = ti.Vector([0,0,0])
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
def ray_color(x,y):
    origin = rays.origins[x,y]
    direction = rays.directions[x,y]
    result = ti.Vector([1.0,0.0,0.0])
    t = hit_sphere(ti.Vector([0.0,0.0,-1.0]),0.5,origin, direction)
    if t>0:
        n = rays.get_at(x,y,t) - ti.Vector([0.0,0.0,-1.0])
        n = n.normalized()
        result = 0.5*ti.Vector([n.x+1.0,n.y+1.0,n.z+1.0])*255.999
    else:
        unit_direction = direction.normalized()
        # print(unit_direction)
        t = 0.5 * (unit_direction.y+1.0)
        result = ((1.0-t)*ti.Vector([1.0,1.0,1.0])+t*ti.Vector([0.5,0.7,1.0]))*255.999
    return result


@ti.kernel  
def initialize():
    for x,y in pixels:
        # r = x * 1.0 / (WIDTH-1)
        # g = y * 1.0 / (HEIGHT-1)
        # b = 0.25

        # ir = 255.999 * r
        # ig = 255.999 * g
        # ib = 255.999 * b
        # pixels[x,y] =  ti.Vector([ir, ig, ib])

        u = x / (WIDTH-1)
        v = y / (HEIGHT-1)

        rays.origins[x,y] = origin
        rays.directions[x,y] = lower_left_corner+u*horizontal + v*vertical - origin
        pixels[x,y] = ray_color(x,y)
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

