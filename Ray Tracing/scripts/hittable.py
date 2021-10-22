import taichi as ti
from ray import *

## Constants
INFINITY = float('inf')
PI = 3.141592653589793238

## Utility Functions
def degrees_to_radians(degrees):
    return degrees*PI/180.0

class HitRecord:
    def __init__(self, point, normal, t):
        self.point = point
        self.normal = normal
        self.t = t 
        self.front_face = True 
    
    @ti.func
    def set_face_normal(self, direction, outward_normal):
        front_face = direction.dot(outward_normal) < 0
        if front_face:
            self.normal = outward_normal 
        else:
            self.normal = -1.0*outward_normal 

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

@ti.func
def is_front_face(direction, outward_normal):
    return direction.dot(outward_normal) < 0

#TODO: normal = is_front_face?outward_normal:-outward_normal

@ti.func
def hit_sphere(center, radius, origin, direction, t_min, t_max, hit_record):
    oc = origin - center
    a = direction.norm_sqr()
    half_b = oc.dot(direction)
    c = oc.norm_sqr() - radius*radius
    discriminant = half_b*half_b - a*c

    root = -1.0
    hitted = False
    if discriminant >= 0:
        sqrtd = ti.sqrt(discriminant)

        # find the nearest root that lies in the acceptable range
        root = (-half_b-sqrtd)/a
        if not (root<t_min or root>t_max):
            hitted = True
            hit_record.t=root 
            hit_record.point=ray_at(origin, direction, hit_record.t)
            hit_record.normal=(hit_record.point-center)/radius
            outward_normal = (hit_record.point-center)/radius 
            hit_record.set_face_normal(direction, outward_normal)
    return hitted,hit_record

# TODO: take care of hit record, rec.t=root, rec.p=r.at(rec.t), rec.normal=(rec.p-center)/radius 

class World:
    def __init__(self):
        self.spheres = []
    
    def add(self, sphere):
        self.spheres.append(sphere)

    def clear(self):
        self.spheres.clear()

    @ti.func
    def hit(self, origin, direction, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        for sphere in self.spheres:
            hitted,root = hit_sphere(sphere.center,sphere.radius, origin, direction, t_min, t_max)
            if hitted: 
                hit_anything = True
                closest_so_far = root
        return hit_anything, closest_so_far
