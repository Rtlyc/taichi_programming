import taichi as ti 

@ti.data_oriented
class Rays:
    def __init__(self, x, y):
        self.origins = ti.Vector.field(3, dtype=ti.f32)  
        self.directions = ti.Vector.field(3, dtype=ti.f32)
        self.lengths = ti.Vector.field(1, dtype=ti.f32)
        self.ats = ti.Vector.field(3, dtype=ti.f32)
        self.attenuations = ti.Vector.field(3, dtype=ti.f32)
        self.depths = ti.field(dtype=ti.i32)
        ti.root.dense(ti.ij, (x, y)).place(self.origins, self.directions, self.lengths, self.ats, self.depths, self.attenuations)
    
    @ti.func
    def get_at(self,x,y,t):
        return self.origins[x,y]+t*self.directions[x,y]
    
    @ti.func
    def set_ray(self, x, y, origin, direction, depth, attenuation):
        self.origins[x,y] = origin 
        self.directions[x,y] = direction
        self.depths[x,y] = depth
        self.attenuations[x,y] = attenuation 
    
    @ti.func
    def get_ray(self, x, y):
        return self.origins[x,y], self.directions[x,y], self.depths[x,y], self.attenuations[x,y]

    # def at(self, length):
    #     return self.origin + length*self.direction

@ti.func
def ray_at(origin, direction, length):
    return origin + length*direction

    
