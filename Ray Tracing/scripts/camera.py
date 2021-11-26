import taichi as ti
from utils import *

@ti.data_oriented
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = degrees_to_radians(vfov)
        viewport_height = ti.tan(theta/2)*2
        viewport_width = aspect_ratio*viewport_height

        self.w = (lookfrom-lookat).normalized()
        self.u = (vup.cross(self.w)).normalized()
        self.v = self.w.cross(self.u)

        self.origin = lookfrom
        self.horizontal = focus_dist*viewport_width*self.u
        self.vertical = focus_dist*viewport_height*self.v 
        self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - focus_dist*self.w

        self.lens_radius = aperture/2

    @ti.func 
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u*rd[0] + self.v*rd[1]
        return self.origin+offset, self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin - offset  

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