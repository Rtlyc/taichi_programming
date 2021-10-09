import taichi as ti 

@ti.data_oriented
class Rays:
    def __init__(self, x, y):
        self.origins = ti.Vector.field(3, dtype=ti.f32)  
        self.directions = ti.Vector.field(3, dtype=ti.f32)
        self.lengths = ti.Vector.field(1, dtype=ti.f32)
        ti.root.dense(ti.ij, (x, y)).place(self.origins, self.directions, self.lengths) 

    # def at(self, length):
    #     return self.origin + length*self.direction

    
