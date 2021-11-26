import taichi as ti
from utils import *

@ti.data_oriented
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio):
        theta = degrees_to_radians(vfov)
        viewport_height = ti.tan(theta/2)*2
        viewport_width = aspect_ratio*viewport_height

        w = (lookfrom-lookat).normalized()
        u = (vup.cross(w)).normalized()
        v = w.cross(u)

        self.origin = lookfrom
        self.horizontal = viewport_width*u
        self.vertical = viewport_height*v 
        self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - w

    @ti.func 
    def get_ray(self, s, t):
        return self.origin, self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin 

# @ti.func
# def clamp(x, min, max):
#     ans = x
#     if x<min: 
#         ans=min 
#     if x>max: 
#         ans=max 
#     return ans 

# @ti.func
# def write_color(color, samples_per_pixels):
#     r = color.x
#     g = color.y
#     b = color.z

#     scale = 1.0/samples_per_pixels
#     r *= scale
#     g *= scale 
#     b *= scale 
#     return ti.Vector([int(clamp(r,0,256)),int(clamp(g,0,256)),int(clamp(b,0,256))])