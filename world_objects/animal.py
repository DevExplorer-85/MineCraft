import math
import random
import numpy as np
import moderngl as mgl
from setting import WORLD_W, WORLD_D, WORLD_H, WORLD_AREA, CHUNK_SIZE, CHUNK_AREA, WATER_LVL
from terrain_gen import get_height

VERT = """
#version 330 core
in vec3 in_v;
in vec3 in_n;
in vec3 in_pos;
in float in_yaw;
in vec3 in_off;
in vec3 in_size;
in vec3 in_col;
in float in_swing;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform vec3 u_cam_pos;

out vec3 v_col;
out float v_shade;
out float v_dist;
out vec3 v_pp;
out float v_seed;

void main() {
    float hy = in_size.y * 0.5;
    vec3 local = in_v * in_size - vec3(0.0, hy, 0.0);      // pivot = part ka upar ka hissa

    float c = cos(in_swing), s = sin(in_swing);
    local = vec3(local.x, local.y * c - local.z * s, local.y * s + local.z * c);
    local += in_off + vec3(0.0, hy, 0.0);

    float cy = cos(in_yaw), sy = sin(in_yaw);
    vec3 w = vec3(local.x * cy + local.z * sy, local.y, -local.x * sy + local.z * cy) + in_pos;
    vec3 n = vec3(in_n.x * cy + in_n.z * sy, in_n.y, -in_n.x * sy + in_n.z * cy);

    vec3 L = normalize(vec3(0.55, 0.60, 0.35));
    v_shade = 0.55 + 0.45 * max(dot(n, L), 0.0);
    v_col = in_col;
    v_pp = in_v * in_size * 16.0 - in_n * 0.02;            // part-space pixels (texture noise ke liye)
    v_seed = dot(in_off, vec3(12.9898, 78.233, 37.719));
    v_dist = length(w - u_cam_pos);
    gl_Position = m_proj * m_view * vec4(w, 1.0);
}
"""

FRAG = """
#version 330 core
in vec3 v_col;
in float v_shade;
in float v_dist;
in vec3 v_pp;
in float v_seed;
layout(location=0) out vec4 fragColor;

float h3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }

void main() {
    float n = h3(floor(v_pp) + vec3(v_seed, v_seed * 0.37, v_seed * 0.71));
    vec3 col = pow(v_col, vec3(2.2)) * v_shade * (0.88 + 0.24 * n);
    col = pow(col, vec3(1.0 / 2.2));

    float fog = smoothstep(120.0, 520.0, v_dist);
    fog = fog * fog * 0.95;
    col = mix(col, vec3(0.58, 0.78, 0.95), fog);
    fragColor = vec4(col, 1.0);
}
"""


# ---------------- models (Minecraft pixels: 16 px = 1 block, forward = +Z) ----------------
def P(c, s, col, sw=0.0):
    return (tuple(v / 16.0 for v in c), tuple(v / 16.0 for v in s), col, sw)


def legs4(lx, lz, h, w, col):
    # top at y=h. Diagonal pairs ulti swing karte hain
    def leg(x, z, sw):
        return P((x, h / 2, z), (w, h, w), col, sw)
    return [leg(lx, lz, 1.0), leg(-lx, -lz, 1.0), leg(-lx, lz, -1.0), leg(lx, -lz, -1.0)]


DARK = (0.08, 0.06, 0.06)

PINK, PINK_D = (0.96, 0.68, 0.67), (0.90, 0.58, 0.58)
PIG = legs4(3, 5.5, 6, 4, PINK_D) + [
    P((0, 10, 0), (10, 8, 16), PINK),
    P((0, 11, 12), (8, 8, 8), PINK),
    P((0, 9.5, 16.5), (4, 3, 1), (0.88, 0.52, 0.54)),
    P((-1, 9.5, 17.1), (0.8, 0.8, 0.4), DARK), P((1, 9.5, 17.1), (0.8, 0.8, 0.4), DARK),
    P((-2.5, 12.5, 16.1), (1.4, 1.4, 0.4), DARK), P((2.5, 12.5, 16.1), (1.4, 1.4, 0.4), DARK),
    P((-3, 15.3, 10.5), (2, 1.5, 2), PINK_D), P((3, 15.3, 10.5), (2, 1.5, 2), PINK_D),
]

WHITE, BLK = (0.92, 0.90, 0.87), (0.20, 0.16, 0.14)
COW = legs4(4, 7, 12, 4, (0.80, 0.79, 0.77)) + [
    P((0, 17, 0), (12, 10, 18), WHITE),
    P((3, 17, 3), (6.4, 10.4, 6), BLK), P((-3, 17, -4), (6.4, 10.4, 5), BLK),
    P((0, 20, 12), (8, 8, 6), (0.30, 0.24, 0.20)),
    P((0, 17.8, 15.5), (4, 3, 1), (0.86, 0.72, 0.66)),
    P((-4.5, 25.2, 11.5), (1, 2.4, 1), (0.90, 0.88, 0.80)), P((4.5, 25.2, 11.5), (1, 2.4, 1), (0.90, 0.88, 0.80)),
    P((-3, 21, 15.1), (1.4, 1.4, 0.4), (0.95, 0.95, 0.95)), P((3, 21, 15.1), (1.4, 1.4, 0.4), (0.95, 0.95, 0.95)),
]

WOOL, SKIN = (0.93, 0.93, 0.91), (0.76, 0.66, 0.56)
SHEEP = legs4(3, 6.5, 12, 4, (0.62, 0.52, 0.45)) + [
    P((0, 17.5, 0), (10, 11, 16), WOOL),
    P((0, 20.5, 11.5), (6, 6, 8), SKIN),
    P((0, 24.2, 10), (6.8, 1.6, 5), WOOL),
    P((-4, 22, 11), (2.4, 1, 2.4), (0.66, 0.55, 0.47)), P((4, 22, 11), (2.4, 1, 2.4), (0.66, 0.55, 0.47)),
    P((-2, 21.5, 15.6), (1, 1, 0.4), (0.12, 0.10, 0.10)), P((2, 21.5, 15.6), (1, 1, 0.4), (0.12, 0.10, 0.10)),
]

DB, DD, WH, ANT = (0.52, 0.33, 0.18), (0.36, 0.22, 0.12), (0.94, 0.90, 0.84), (0.82, 0.74, 0.58)
DEER = legs4(3, 7.5, 15, 3, DD) + [
    P((0, 19.5, 0), (8, 9, 18), DB),
    P((0, 15.2, 0), (6, 0.9, 14), WH),
    P((0, 27, 9), (4, 10, 4), DB),
    P((0, 31.5, 13), (5, 5, 8), DB),
    P((0, 30.5, 17.4), (3, 2.6, 0.8), DARK),
    P((-2.8, 32.5, 14.5), (0.6, 1, 1), DARK), P((2.8, 32.5, 14.5), (0.6, 1, 1), DARK),
    P((-2, 35.5, 10.5), (1, 4, 2), DB), P((2, 35.5, 10.5), (1, 4, 2), DB),
    P((-1.4, 38.5, 12), (1.2, 9, 1.2), ANT), P((1.4, 38.5, 12), (1.2, 9, 1.2), ANT),
    P((-1.4, 40.5, 13.6), (1.2, 1.2, 3.6), ANT), P((1.4, 40.5, 13.6), (1.2, 1.2, 3.6), ANT),
    P((0, 22, -10), (3, 5, 2.5), WH),
]

# hp = kitni chot, r/h = hitbox, drop = (item, min, max)
SPECIES = [
    dict(name='pig',   parts=PIG,   count=70, group=(2, 4), hmin=WATER_LVL + 3, hmax=34, speed=1.2, step=1,
         hp=10, r=0.7, h=1.0, drop=('raw_pork', 1, 3)),
    dict(name='cow',   parts=COW,   count=60, group=(2, 4), hmin=22, hmax=44, speed=1.0, step=1,
         hp=10, r=0.8, h=1.5, drop=('raw_beef', 1, 3)),
    dict(name='sheep', parts=SHEEP, count=60, group=(3, 6), hmin=30, hmax=58, speed=1.0, step=1,
         hp=8, r=0.7, h=1.4, drop=('raw_mutton', 1, 2)),
    dict(name='deer',  parts=DEER,  count=50, group=(2, 4), hmin=40, hmax=75, speed=2.0, step=1,
         hp=14, r=0.7, h=2.2, drop=('raw_venison', 2, 3)),
]

WORLD_SIZE_X = WORLD_W * CHUNK_SIZE
WORLD_SIZE_Z = WORLD_D * CHUNK_SIZE
MARGIN = 6
FAR2 = 160.0 * 160.0      # isse door ke animals update nahi hote (FPS bachane ke liye)


def make_cube():
    verts = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3)
            n[ax] = sgn
            t = np.zeros(3)
            b = np.zeros(3)
            t[(ax + 1) % 3] = 1
            b[(ax + 2) % 3] = 1
            c = [n * 0.5 - t * 0.5 - b * 0.5, n * 0.5 + t * 0.5 - b * 0.5,
                 n * 0.5 + t * 0.5 + b * 0.5, n * 0.5 - t * 0.5 + b * 0.5]
            for i in (0, 1, 2, 0, 2, 3):
                verts.append(np.concatenate([c[i], n]))
    return np.array(verts, dtype='f4')


class Animal:
    def __init__(self, sp, x, z, start, parts, rng):
        self.sp = sp
        self.x, self.z = x, z
        self.y = float(get_height(int(x), int(z)))
        self.yaw = rng.uniform(0, 2 * math.pi)
        self.target_yaw = self.yaw
        self.moving = False
        self.timer = rng.uniform(0, 4)
        self.phase = rng.uniform(0, 6.28)
        self.amp = 0.0
        self.s, self.e = start, start + len(parts)
        self.hp = sp['hp']
        self.hurt = 0.0
        self.flee = 0.0
        self.flash = False
        self.dead = False
        self.respawn = 0.0
        self.swing_sign = np.array([p[3] for p in parts], dtype='f4')
        self.sizes = np.array([p[1] for p in parts], dtype='f4')
        self.colors = np.array([p[2] for p in parts], dtype='f4')


class Animals:
    def __init__(self, app, world):
        self.app = app
        self.world = world
        self.ctx = app.ctx
        self.rng = random.Random(1234)

        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.cube_vbo = self.ctx.buffer(make_cube().tobytes())

        self.animals = []
        self.spawn()

        n = max(1, self.total)
        self.inst_vbo = self.ctx.buffer(reserve=n * 14 * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(self.program, [
            (self.cube_vbo, '3f 3f', 'in_v', 'in_n'),
            (self.inst_vbo, '3f 1f 3f 3f 3f 1f/i',
             'in_pos', 'in_yaw', 'in_off', 'in_size', 'in_col', 'in_swing'),
        ])
        self.write_instances()

    # ---------- world queries ----------
    def solid_at(self, x, y, z):
        if not (0 <= x < WORLD_SIZE_X and 0 <= z < WORLD_SIZE_Z and 0 <= y < WORLD_H * CHUNK_SIZE):
            return True
        ci = x // CHUNK_SIZE + WORLD_W * (z // CHUNK_SIZE) + WORLD_AREA * (y // CHUNK_SIZE)
        vi = x % CHUNK_SIZE + (z % CHUNK_SIZE) * CHUNK_SIZE + (y % CHUNK_SIZE) * CHUNK_AREA
        return self.world.voxels[ci][vi] != 0

    def can_stand(self, sp, x, z, cur_h=None):
        if not (MARGIN <= x < WORLD_SIZE_X - MARGIN and MARGIN <= z < WORLD_SIZE_Z - MARGIN):
            return False
        ix, iz = int(x), int(z)
        h = get_height(ix, iz)
        if h < sp['hmin'] or h > sp['hmax']:
            return False
        if cur_h is not None and abs(h - cur_h) > sp['step']:
            return False
        if self.solid_at(ix, h, iz) or self.solid_at(ix, h + 1, iz):
            return False
        return True

    # ---------- spawning ----------
    def spawn(self):
        rng = self.rng
        pending = []
        for sp in SPECIES:
            placed, tries = 0, 0
            while placed < sp['count'] and tries < 6000:
                tries += 1
                cx = rng.uniform(MARGIN, WORLD_SIZE_X - MARGIN)
                cz = rng.uniform(MARGIN, WORLD_SIZE_Z - MARGIN)
                if not self.can_stand(sp, cx, cz):
                    continue
                n = min(rng.randint(*sp['group']), sp['count'] - placed)
                for _ in range(n):
                    x = cx + rng.uniform(-4, 4)
                    z = cz + rng.uniform(-4, 4)
                    if self.can_stand(sp, x, z):
                        pending.append((sp, x, z))
                        placed += 1

        self.total = sum(len(sp['parts']) for sp, _, _ in pending)
        self.inst = np.zeros((max(1, self.total), 14), dtype='f4')

        start = 0
        for sp, x, z in pending:
            parts = sp['parts']
            a = Animal(sp, x, z, start, parts, rng)
            for i, (off, size, col, sw) in enumerate(parts):
                r = start + i
                self.inst[r, 4:7] = off
                self.inst[r, 7:10] = size
                self.inst[r, 10:13] = col
            self.animals.append(a)
            start += len(parts)

    def try_respawn(self, a):
        sp = a.sp
        pp = self.app.player.position
        for _ in range(200):
            x = self.rng.uniform(MARGIN, WORLD_SIZE_X - MARGIN)
            z = self.rng.uniform(MARGIN, WORLD_SIZE_Z - MARGIN)
            if (x - pp.x) ** 2 + (z - pp.z) ** 2 < 40 * 40:
                continue
            if self.can_stand(sp, x, z):
                a.x, a.z = x, z
                a.y = float(get_height(int(x), int(z)))
                a.hp, a.hurt, a.flee, a.dead = sp['hp'], 0.0, 0.0, False
                sl = slice(a.s, a.e)
                self.inst[sl, 0], self.inst[sl, 1], self.inst[sl, 2] = a.x, a.y, a.z
                self.inst[sl, 7:10] = a.sizes
                self.inst[sl, 10:13] = a.colors
                return
        a.respawn = 10.0

    # ---------- combat ----------
    def raycast(self, o, d, max_dist):
        # o, d = (x, y, z) tuples. Return: (animal, distance) ya (None, max_dist)
        best, bt = None, max_dist
        for a in self.animals:
            if a.dead:
                continue
            dx, dz = a.x - o[0], a.z - o[2]
            if dx * dx + dz * dz > (max_dist + 2.0) ** 2:
                continue
            r, h = a.sp['r'], a.sp['h']
            mn = (a.x - r, a.y, a.z - r)
            mx = (a.x + r, a.y + h, a.z + r)
            t0, t1, ok = 0.0, bt, True
            for i in range(3):
                if abs(d[i]) < 1e-8:
                    if o[i] < mn[i] or o[i] > mx[i]:
                        ok = False
                        break
                else:
                    ta, tb = (mn[i] - o[i]) / d[i], (mx[i] - o[i]) / d[i]
                    if ta > tb:
                        ta, tb = tb, ta
                    t0, t1 = max(t0, ta), min(t1, tb)
                    if t0 > t1:
                        ok = False
                        break
            if ok and t0 < bt:
                bt, best = t0, a
        return best, bt

    def hit(self, a, dmg, src):
        # return: drops list [(item, n)] agar mar gaya, warna []
        a.hp -= dmg
        a.hurt = 0.35
        a.flee = 3.0
        dx, dz = a.x - src[0], a.z - src[2]
        l = math.hypot(dx, dz) or 1.0
        a.target_yaw = math.atan2(dx, dz)
        nx, nz = a.x + dx / l * 0.7, a.z + dz / l * 0.7
        if self.can_stand(a.sp, nx, nz, get_height(int(a.x), int(a.z))):
            a.x, a.z = nx, nz

        if a.hp > 0:
            return []
        a.dead = True
        a.respawn = self.rng.uniform(60, 120)
        self.inst[a.s:a.e, 7:10] = 0.0          # gayab
        item, lo, hi = a.sp['drop']
        return [(item, self.rng.randint(lo, hi))]

    # ---------- per frame ----------
    def update(self):
        dt = min(self.app.delta_time * 0.001, 0.1)
        if dt <= 0:
            return
        rng = self.rng
        pp = self.app.player.position

        for a in self.animals:
            sp = a.sp
            sl = slice(a.s, a.e)

            if a.dead:
                a.respawn -= dt
                if a.respawn <= 0:
                    self.try_respawn(a)
                continue

            ddx, ddz = a.x - pp.x, a.z - pp.z
            if ddx * ddx + ddz * ddz > FAR2:
                continue

            a.timer -= dt
            if a.flee > 0:
                a.flee -= dt
                a.moving = True
                if a.flee <= 0:
                    a.moving = False
                    a.timer = 1.0
            elif a.timer <= 0:
                if a.moving:
                    a.moving = False
                    a.timer = rng.uniform(2.0, 6.0)
                else:
                    a.moving = True
                    a.timer = rng.uniform(3.0, 8.0)
                    a.target_yaw = a.yaw + rng.uniform(-1.5, 1.5)

            turn = 5.0 if a.flee > 0 else 2.5
            diff = (a.target_yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
            a.yaw += max(-turn * dt, min(turn * dt, diff))

            speed = sp['speed'] * (2.4 if a.flee > 0 else 1.0)
            cur_h = get_height(int(a.x), int(a.z))

            if a.moving and abs(diff) < 0.6:
                nx = a.x + math.sin(a.yaw) * speed * dt
                nz = a.z + math.cos(a.yaw) * speed * dt
                if self.can_stand(sp, nx, nz, cur_h):
                    a.x, a.z = nx, nz
                else:
                    a.target_yaw = a.yaw + rng.uniform(1.8, 3.8)
                    if a.flee <= 0:
                        a.moving = False
                        a.timer = 1.2

            ty = float(get_height(int(a.x), int(a.z)))
            a.y += (ty - a.y) * min(1.0, 10.0 * dt)

            a.amp += ((1.0 if a.moving else 0.0) - a.amp) * min(1.0, 8.0 * dt)
            if a.moving:
                a.phase += speed * dt * 5.0
            angle = 0.65 * a.amp * math.sin(a.phase)

            self.inst[sl, 0] = a.x
            self.inst[sl, 1] = a.y
            self.inst[sl, 2] = a.z
            self.inst[sl, 3] = a.yaw
            self.inst[sl, 13] = a.swing_sign * angle

            # chot lagne par laal flash
            if a.hurt > 0:
                a.hurt -= dt
                t = max(a.hurt, 0.0) / 0.35 * 0.65
                self.inst[sl, 10:13] = a.colors * (1 - t) + np.array([1.0, 0.15, 0.15], dtype='f4') * t
                a.flash = True
            elif a.flash:
                self.inst[sl, 10:13] = a.colors
                a.flash = False

        self.write_instances()

    def write_instances(self):
        self.inst_vbo.write(self.inst.tobytes())

    def render(self):
        if not self.animals:
            return
        self.program['m_proj'].write(self.app.player.m_proj)
        self.program['m_view'].write(self.app.player.m_view)
        self.program['u_cam_pos'].value = tuple(self.app.player.position)
        self.vao.render(mgl.TRIANGLES, instances=len(self.inst))