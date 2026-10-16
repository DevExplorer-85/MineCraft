import numpy as np
import moderngl as mgl
from terrain_gen import fbm

CLOUD_Y = 108          # peaks 88 tak jaate hain, toh upar
CLOUD_THICK = 5
CELL = 16              # ek cloud block ka size
AREA_MIN, AREA_MAX = -300, 1260   # world (0..960) se bahar tak, taaki kinare khaali na dikhein
DRIFT_SPEED = 2.0      # blocks/second

VERT = """
#version 330 core
layout(location=0) in vec3 in_position;
layout(location=1) in float in_shade;
uniform mat4 m_proj;
uniform mat4 m_view;
uniform vec3 u_offset;
out float shade;
void main() {
    shade = in_shade;
    gl_Position = m_proj * m_view * vec4(in_position + u_offset, 1.0);
}
"""

FRAG = """
#version 330 core
in float shade;
layout(location=0) out vec4 fragColor;
void main() {
    fragColor = vec4(vec3(shade), 0.88);
}
"""


class Clouds:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        data = self.build_mesh()
        self.vbo = self.ctx.buffer(data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, '3f 1f', 'in_position', 'in_shade')]
        )

    def build_mesh(self):
        n = (AREA_MAX - AREA_MIN) // CELL
        grid = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                wx = AREA_MIN + i * CELL
                wz = AREA_MIN + j * CELL
                v = 0.5 + 0.5 * fbm(wx + 5000.0, wz - 3000.0, 4, 0.004)
                grid[i, j] = v > 0.55      # cloud kam/zyada: 0.50 se 0.62 ke beech tune karo

        verts = []

        def quad(p0, p1, p2, p3, s):
            for p in (p0, p1, p2, p0, p2, p3):
                verts.append((*p, s))

        y0, y1 = CLOUD_Y, CLOUD_Y + CLOUD_THICK
        for i in range(n):
            for j in range(n):
                if not grid[i, j]:
                    continue
                x0 = AREA_MIN + i * CELL
                x1 = x0 + CELL
                z0 = AREA_MIN + j * CELL
                z1 = z0 + CELL

                quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), 1.0)   # top
                quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), 0.7)   # bottom
                if i == 0 or not grid[i - 1, j]:
                    quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), 0.85)
                if i == n - 1 or not grid[i + 1, j]:
                    quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), 0.85)
                if j == 0 or not grid[i, j - 1]:
                    quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), 0.8)
                if j == n - 1 or not grid[i, j + 1]:
                    quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), 0.8)

        return np.array(verts, dtype='f4')

    def render(self):
        t = getattr(self.app, 'time', 0.0)
        self.program['m_proj'].write(self.app.player.m_proj)
        self.program['m_view'].write(self.app.player.m_view)
        self.program['u_offset'].value = (t * DRIFT_SPEED, 0.0, 0.0)

        self.vao.render()