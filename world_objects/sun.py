import numpy as np
import glm
import moderngl as mgl

SUN_DIR = glm.normalize(glm.vec3(0.55, 0.60, 0.35))   # sun kis taraf hai
SUN_DIST = 1000.0                                     # FAR (1500) se andar rakho
SUN_HALF = 260.0                                      # quad ka size (glow ke saath)

VERT = """
#version 330 core
layout(location=0) in vec2 in_corner;
uniform mat4 m_proj;
uniform mat4 m_view;
uniform vec3 u_center;
uniform vec3 u_right;
uniform vec3 u_up;
out vec2 v_uv;
void main() {
    v_uv = in_corner;
    vec3 p = u_center + u_right * in_corner.x + u_up * in_corner.y;
    gl_Position = m_proj * m_view * vec4(p, 1.0);
}
"""

FRAG = """
#version 330 core
in vec2 v_uv;
layout(location=0) out vec4 fragColor;
void main() {
    float r = length(v_uv);

    // gol disc, kinare thode naram
    float core = 1.0 - smoothstep(0.20, 0.23, r);

    // disc ke bahar naram chamak (halo)
    float glow = pow(clamp(1.0 - r, 0.0, 1.0), 3.0);

    vec3 col = mix(vec3(1.0, 0.85, 0.55), vec3(1.0, 0.98, 0.85), core);
    float a = clamp(core + glow * 0.6, 0.0, 1.0);

    if (a < 0.003) discard;
    fragColor = vec4(col, a);
}
"""


class Sun:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.vbo = self.ctx.buffer(corners.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f', 'in_corner')])

        right = glm.normalize(glm.cross(SUN_DIR, glm.vec3(0, 1, 0)))
        up = glm.cross(right, SUN_DIR)
        self.right = right * SUN_HALF
        self.up = up * SUN_HALF

    def render(self):
        pos = self.app.player.position
        center = pos + SUN_DIR * SUN_DIST      # camera ke saath chalta hai, isliye hamesha door

        self.program['m_proj'].write(self.app.player.m_proj)
        self.program['m_view'].write(self.app.player.m_view)
        self.program['u_center'].value = tuple(center)
        self.program['u_right'].value = tuple(self.right)
        self.program['u_up'].value = tuple(self.up)
        self.vao.render(mgl.TRIANGLE_STRIP)