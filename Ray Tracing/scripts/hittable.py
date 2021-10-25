import taichi as ti
from ray import *

## Constants
INFINITY = float('inf')
PI = 3.141592653589793238
# normal = ti.Vector([0.0,0.0,0.0])
# ti.root.dense(ti.i, (1)).place(normal)

## Utility Functions
def degrees_to_radians(degrees):
    return degrees*PI/180.0
@ti.data_oriented
class HitRecord:
    def __init__(self, t:ti.f32, point=ti.Vector([0.0,0.0,0.0]), normal=ti.Vector([0.0,0.0,0.0])):
        self.point = point
        self.normal = normal
        self.t = t 
        self.front_face = True 
    
@ti.func
def set_face_normal(direction, outward_normal):
    front_face = direction.dot(outward_normal) < 0
    cur_normal = outward_normal
    if not front_face:
        cur_normal = -1.0*outward_normal 
    return cur_normal

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

@ti.func
def is_front_face(direction, outward_normal):
    return direction.dot(outward_normal) < 0

#TODO: normal = is_front_face?outward_normal:-outward_normal

@ti.func
def hit_sphere(center, radius, ray_origin, ray_direction, t_min, t_max):
    oc = ray_origin - center
    a = ray_direction.norm_sqr()
    half_b = oc.dot(ray_direction)
    c = oc.norm_sqr() - radius*radius
    discriminant = half_b*half_b - a*c
    # hit_record = HitRecord(0.0)
    cur_normal = ti.Vector([0.0,0.0,0.0])

    root = -1.0
    hitted = False
    if discriminant >= 0:
        sqrtd = ti.sqrt(discriminant)

        # find the nearest root that lies in the acceptable range
        root = (-half_b-sqrtd)/a
        if not (root<t_min or root>t_max):
            hitted = True
            # hit_record.t=root 
            point=ray_at(ray_origin, ray_direction, root)
            outward_normal = (point-center)/radius 
            cur_normal = set_face_normal(ray_direction, outward_normal)
        else:
            root=(-half_b+sqrtd)/a
    return hitted,root,cur_normal

# TODO: take care of hit record, rec.t=root, rec.p=r.at(rec.t), rec.normal=(rec.p-center)/radius 

@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []
    
    def add(self, sphere):
        self.spheres.append(sphere)

    def clear(self):
        self.spheres.clear()
    
    def finalize(self):
        self.n = len(self.spheres)
        self.radius = ti.field(ti.f32)
        self.center = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.n).place(self.radius, self.center)
        for i in range(self.n):
            self.radius[i] = self.spheres[i].radius
            self.center[i] = self.spheres[i].center
        

    @ti.func
    def hit(self, ray_origin, ray_direction, t_min, t_max):
        # temp_hitrecord = HitRecord(0.0)
        hit_anything = False
        closest_so_far = t_max
        normal = ti.Vector([0.0,0.0,0.0])
        for i in range(self.n):
            hitted,root,normal = hit_sphere(self.center[i],self.radius[i], ray_origin, ray_direction, t_min, closest_so_far)
            if hitted: 
                hit_anything = True
                closest_so_far = root
        #TODO: if not hit: my solution: 0,1,0, his solution: previous value
        return hit_anything, normal 

@ti.func
def find_normal(ray_origin, ray_direction, root, center, radius):
    point=ray_at(ray_origin, ray_direction, root)
    outward_normal = (point-center)/radius 
    normal = set_face_normal(ray_direction, outward_normal)
    return normal