import taichi as ti
from ray import *
from material import *


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
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

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

    cur_normal = ti.Vector([1.0,0.0,0.0])
    point = ti.Vector([1.0,0.0,0.0])
    root = -1.0
    hitted = False
    if discriminant<0:
        pass 
    else:
        sqrtd = ti.sqrt(discriminant)

        # find the nearest root that lies in the acceptable range
        root = (-half_b-sqrtd)/a
        if (root<t_min or root>t_max):
            root = (-half_b+sqrtd)/a 
            if root<t_min or t_max < root:
                hitted = False 
            else:
                hitted = True  
        else:
            hitted = True 
        # point=ray_at(ray_origin, ray_direction, root)
        # outward_normal = (point-center)/radius 
        # cur_normal = set_face_normal(ray_direction, outward_normal)

    # return hitted,root,cur_normal,point
    return hitted,root

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
        self.materials = Materials(self.n)
        ti.root.dense(ti.i, self.n).place(self.radius, self.center)
        for i in range(self.n):
            self.radius[i] = self.spheres[i].radius
            self.center[i] = self.spheres[i].center
            self.materials.set(i,self.spheres[i].material)
        

    @ti.func
    def hit(self, ray_origin, ray_direction, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        n = ti.Vector([0.0,1.0,0.0])
        p = ti.Vector([0.0,1.0,0.0])
        ind = 0
        for i in range(self.n):
            hitted,root = hit_sphere(self.center[i],self.radius[i], ray_origin, ray_direction, t_min, closest_so_far)
            if hitted: 
                hit_anything = True
                closest_so_far = root
                # normal = cur_normal
                ind = i
        if hit_anything:
            p = ray_at(ray_origin, ray_direction, closest_so_far)
            n = (p-self.center[ind])/self.radius[ind]
            # front_facing = is_front_face(ray_direction, n)
            # n = n if front_facing else -n
            n = set_face_normal(ray_direction, n)
        return hit_anything, n, p, ind

    # @ti.func
    # def scatter(self, ray_direction, p,n, front_facing, index):
    #     return self.materials

@ti.func
def find_normal(ray_origin, ray_direction, root, center, radius):
    point=ray_at(ray_origin, ray_direction, root)
    outward_normal = (point-center)/radius 
    normal = set_face_normal(ray_direction, outward_normal)
    return normal