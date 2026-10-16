import numpy as np
import pygame as pg
import moderngl as mgl
from setting import WIN_RES
from world_objects.inventory import ITEMS, RECIPES

VERT = """
#version 330 core
layout(location=0) in vec2 p;
out vec2 uv;
void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }
"""
FRAG = """
#version 330 core
in vec2 uv;
uniform sampler2D u_tex;
layout(location=0) out vec4 c;
void main() { c = texture(u_texture_dummy, uv); }
""".replace('u_texture_dummy', 'u_tex')


class HUD:
    def __init__(self, app):
        self.ctx = app.ctx
        self.w, self.h = int(WIN_RES.x), int(WIN_RES.y)
        self.surf = pg.Surface((self.w, self.h), pg.SRCALPHA)
        self.font = pg.font.Font(None, 28)
        self.small = pg.font.Font(None, 21)
        self.big = pg.font.Font(None, 36)

        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f', 'p')])
        self.tex = self.ctx.texture((self.w, self.h), 4)
        self.tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self.program['u_tex'].value = 6
        self.sig = None

    def text(self, font, s, color, pos, center=False):
        img = font.render(s, True, color)
        r = img.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        self.surf.blit(img, r)

    def draw(self, inv, sel, start, menu, progress, msgs):
        s = self.surf
        s.fill((0, 0, 0, 0))
        cx, cy = self.w // 2, self.h // 2

        # crosshair
        for w, col in ((5, (0, 0, 0, 200)), (2, (255, 255, 255, 235))):
            L = 11 if w == 5 else 10
            pg.draw.line(s, col, (cx - L, cy), (cx + L, cy), w)
            pg.draw.line(s, col, (cx, cy - L), (cx, cy + L), w)

        if progress > 0:
            pg.draw.rect(s, (0, 0, 0, 170), (cx - 42, cy + 26, 84, 10))
            pg.draw.rect(s, (255, 230, 120, 255), (cx - 40, cy + 28, int(80 * min(progress, 1.0)), 6))

        # hotbar (9 slot ka window)
        items = inv.items()
        size, gap = 62, 6
        x0 = (self.w - (9 * size + 8 * gap)) // 2
        y0 = self.h - 84
        for i in range(9):
            x = x0 + i * (size + gap)
            idx = start + i
            on = idx == sel and idx < len(items)
            pg.draw.rect(s, (20, 20, 20, 170), (x, y0, size, size), border_radius=6)
            pg.draw.rect(s, (255, 255, 255, 255) if on else (90, 90, 90, 255), (x, y0, size, size),
                         3 if on else 2, border_radius=6)
            self.text(self.small, str(i + 1), (200, 200, 200), (x + 5, y0 + 3))
            if idx < len(items):
                name, n = items[idx]
                label, col, _ = ITEMS[name]
                pg.draw.rect(s, col, (x + 17, y0 + 14, 28, 28), border_radius=4)
                pg.draw.rect(s, (0, 0, 0, 255), (x + 17, y0 + 14, 28, 28), 2, border_radius=4)
                self.text(self.font, str(n), (255, 255, 255), (x + size - 8, y0 + size - 16), center=True)
                if on:
                    self.text(self.big, label, (255, 255, 255), (self.w // 2, y0 - 22), center=True)

        # messages
        for i, m in enumerate(msgs):
            self.text(self.font, m, (255, 255, 200), (20, 20 + i * 28))

        self.text(self.small, 'LMB: hit / mine   RMB: place   C: crafting   Z/X: select   1-9: slot',
                  (230, 230, 230), (20, self.h - 26))

        # crafting panel
        if menu:
            pw, ph = 620, 90 + len(RECIPES) * 40
            px, py = (self.w - pw) // 2, (self.h - ph) // 2
            pg.draw.rect(s, (15, 15, 20, 220), (px, py, pw, ph), border_radius=10)
            pg.draw.rect(s, (255, 255, 255, 255), (px, py, pw, ph), 2, border_radius=10)
            self.text(self.big, 'CRAFTING  (press 1-9, C to close)', (255, 255, 255), (px + 20, py + 16))
            for i, (label, need, (out, n)) in enumerate(RECIPES):
                ok = inv.has(need)
                col = (120, 255, 120) if ok else (150, 150, 150)
                needs = ', '.join(f"{v} {ITEMS[k][0]}" for k, v in need.items())
                self.text(self.font, f"[{i + 1}] {label}", col, (px + 20, py + 66 + i * 40))
                self.text(self.small, needs, col, (px + 280, py + 70 + i * 40))

        data = pg.image.tobytes(s, 'RGBA', True)
        self.tex.write(data)

    def render(self):
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self.tex.use(location=6)
        self.vao.render(mgl.TRIANGLE_STRIP)
        self.ctx.enable(mgl.DEPTH_TEST)