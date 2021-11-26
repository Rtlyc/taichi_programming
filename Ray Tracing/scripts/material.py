from taichi import ti 
from utils import * 

@ti.data_oriented
class Lambertian:
    def __init__(self, color):
        self.albedo = color 

    @staticmethod 
    @ti.func
    def scatter(ray_point, normal, color):
        scatter_direction = normal + random_in_unit_sphere()
        if near_zero(scatter_direction):
            scatter_direction = normal 
        # scatter = ray(ray_point, scatter_direction)
        # attenuation = self.albedo 
        return True, ray_point, scatter_direction, color

@ti.data_oriented
class Metal:
    def __init__(self, color):
        self.albedo = color 
    
    @staticmethod 
    @ti.func
    def scatter(ray_point, ray_direction, normal, color):
        reflected = reflect(ray_direction.normalized(), normal)
        return reflected.dot(normal)>0, ray_point, reflected, color

@ti.data_oriented
class Materials:
    def __init__(self, n):
        self.colors = ti.Vector.field(3, ti.f32)
        self.types = ti.field(ti.u32)
        ti.root.dense(ti.i, n).place(self.colors, self.types)

    def set(self, ind, mat):
        self.colors[ind] = mat.albedo
        type = 1
        if(isinstance(mat,Lambertian)):
            type = 1
        elif(isinstance(mat,Metal)):
            type = 2
        self.types[ind] = type 


    @ti.func
    def scatter(self, i, ray_direction, p, n):
        is_reflected = True 
        ray_point = ti.Vector([0.0, 0.0, 0.0])
        reflected = ti.Vector([0.0, 0.0, 0.0])
        attenuation = ti.Vector([0.0, 0.0, 0.0])
        if self.types[i]==1:
            is_reflected, ray_point, reflected, attenuation = Lambertian.scatter(p,n,self.colors[i])
        elif self.types[i]==2:
            is_reflected, ray_point, reflected, attenuation = Metal.scatter(p,ray_direction, n, self.colors[i])
        return is_reflected, ray_point, reflected, attenuation 
        