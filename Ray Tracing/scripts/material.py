from taichi import ti 
from utils import * 

@ti.data_oriented
class Lambertian:
    def __init__(self, color):
        self.albedo = color 

    @ti.func
    def scatter(self, ray_point, normal, attenuation):
        scatter_direction = normal + random_in_unit_sphere()
        if near_zero(scatter_direction):
            scatter_direction = normal 
        # scatter = ray(ray_point, scatter_direction)
        # attenuation = self.albedo 
        return ray_point, scatter_direction, self.albedo