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
    def __init__(self, color, fuzz):
        self.albedo = color 
        self.fuzz = fuzz if fuzz<1.0 else 1.0
    
    @staticmethod 
    @ti.func
    def scatter(ray_point, ray_direction, normal, color, fuzz):
        reflected = reflect(ray_direction.normalized(), normal)
        scattered = reflected + fuzz*random_in_unit_sphere()
        return scattered.dot(normal)>0, ray_point, scattered, color

@ti.data_oriented
class Dielectric:
    def __init__(self, index_of_refraction):
        self.ir = index_of_refraction
        self.albedo = ti.Vector([1.0,1.0,1.0])

    @staticmethod
    @ti.func
    def scatter(ray_point, ray_direction, normal, ir, front_facing):
        attenuation = ti.Vector([1.0,1.0,1.0])
        refraction_ratio = 1.0/ir if front_facing else ir 
        
        unit_direction = ray_direction.normalized()
        cos_theta = min(-unit_direction.dot(normal), 1.0)
        sin_theta = ti.sqrt(1.0-cos_theta*cos_theta)
        cannot_refracted = refraction_ratio*sin_theta>1.0

        direction = refract(unit_direction, normal, refraction_ratio)
        if cannot_refracted:
            direction = reflect(unit_direction, normal)

        
        return True, ray_point, direction, attenuation

@ti.data_oriented
class Materials:
    def __init__(self, n):
        self.colors = ti.Vector.field(3, ti.f32)
        self.types = ti.field(ti.u32)
        self.fuzzes = ti.field(ti.f32)
        self.irs = ti.field(ti.f32)
        ti.root.dense(ti.i, n).place(self.colors, self.types, self.fuzzes, self.irs)

    def set(self, ind, mat):
        self.colors[ind] = mat.albedo
        type = 1
        if(isinstance(mat,Lambertian)):
            type = 1
        elif(isinstance(mat,Metal)):
            type = 2
            self.fuzzes[ind] = mat.fuzz 
        else:
            type = 3
            self.irs[ind] = mat.ir
        self.types[ind] = type 


    @ti.func
    def scatter(self, i, ray_direction, p, n, front_facing):
        is_reflected = True 
        ray_point = ti.Vector([0.0, 0.0, 0.0])
        reflected = ti.Vector([0.0, 0.0, 0.0])
        attenuation = ti.Vector([0.0, 0.0, 0.0])
        if self.types[i]==1:
            is_reflected, ray_point, reflected, attenuation = Lambertian.scatter(p,n,self.colors[i])
        elif self.types[i]==2:
            is_reflected, ray_point, reflected, attenuation = Metal.scatter(p,ray_direction, n, self.colors[i], self.fuzzes[i])
        else:
            is_reflected, ray_point, reflected, attenuation = Dielectric.scatter(p,ray_direction,n,self.irs[i], front_facing)
        return is_reflected, ray_point, reflected, attenuation 
        