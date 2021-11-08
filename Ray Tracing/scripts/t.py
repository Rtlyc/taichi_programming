import taichi as ti

ti.init()

pixels = ti.Vector.field(3, ti.u8)
ti.root.dense(ti.ij, (512, 512)).place(pixels)
# , shape=(512, 512)

@ti.kernel
def set_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([60,80,20])

set_pixels()
filename = f'imwrite_export.png'
ti.imwrite(pixels.to_numpy(), filename)
print(f'The image has been saved to {filename}')