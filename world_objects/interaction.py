import math
import random
import glm
import pygame as pg
from setting import WORLD_W, WORLD_D, WORLD_H, WORLD_AREA, CHUNK_SIZE, CHUNK_AREA
from world_objects.inventory import Inventory, ITEMS, RECIPES
from world_objects.hud import HUD

WX, WZ, WY = WORLD_W * CHUNK_SIZE, WORLD_D * CHUNK_SIZE, WORLD_H * CHUNK_SIZE
WATER = 8
BREAK_DROP = {1: 'sand', 2: 'dirt', 3: 'dirt', 4: 'stone', 7: 'log', 11: 'planks'}
LEAVES = (6, 9, 10)
DIGITS = [pg.K_1 + i for i in range(9)]
KEYS = [pg.K_c, pg.K_z, pg.K_x] + DIGITS


class Interaction:
    def __init__(self, app, world, animals):
        self.app = app
        self.world = world
        self.animals = animals
        self.inv = Inventory()
        self.hud = HUD(app)
        self.sel = 0
        self.menu = False
        self.prev_keys = set()
        self.prev_right = False
        self.mine_pos = None
        self.progress = 0.0
        self.attack_cd = 0.0
        self.msgs = []
        self.sig = None
        self.rng = random.Random(7)

    # ---------- voxel helpers ----------
    @staticmethod
    def index(x, y, z):
        cs = CHUNK_SIZE
        return (x // cs + WORLD_W * (z // cs) + WORLD_AREA * (y // cs),
                x % cs + (z % cs) * cs + (y % cs) * CHUNK_AREA)

    def get_voxel(self, x, y, z):
        if not (0 <= x < WX and 0 <= z < WZ and 0 <= y < WY):
            return 0
        ci, vi = self.index(x, y, z)
        return int(self.world.voxels[ci][vi])

    def set_voxel(self, x, y, z, vid):
        cs = CHUNK_SIZE
        ci, vi = self.index(x, y, z)
        self.world.voxels[ci][vi] = vid
        cx, cy, cz = x // cs, y // cs, z // cs
        todo = {(cx, cy, cz)}
        for c, l, lim, ax in ((cx, x % cs, WORLD_W, 0), (cy, y % cs, WORLD_H, 1), (cz, z % cs, WORLD_D, 2)):
            for dlt, edge in ((-1, 0), (1, cs - 1)):
                if l == edge and 0 <= c + dlt < lim:
                    t = [cx, cy, cz]
                    t[ax] = c + dlt
                    todo.add(tuple(t))
        for a, b, c in todo:
            ch = self.world.chunks[a + WORLD_W * c + WORLD_AREA * b]
            ch.is_empty = False
            ch.mesh.rebuild()

    def raycast_block(self, o, d, max_dist=5.0):
        # return: (hit_pos, prev_pos, dist) ya None. Paani ko ignore karta hai
        pos = [int(math.floor(c)) for c in o]
        step, tmax, tdelta = [], [], []
        for oi, di in zip(o, d):
            if abs(di) < 1e-9:
                step.append(1)
                tmax.append(1e30)
                tdelta.append(1e30)
            else:
                step.append(1 if di > 0 else -1)
                nxt = math.floor(oi) + 1 if di > 0 else math.floor(oi)
                tmax.append((nxt - oi) / di)
                tdelta.append(abs(1.0 / di))
        prev, t = None, 0.0
        for _ in range(64):
            vid = self.get_voxel(*pos)
            if vid and vid != WATER:
                return tuple(pos), prev, t
            i = tmax.index(min(tmax))
            t = tmax[i]
            if t > max_dist:
                return None
            prev = tuple(pos)
            pos[i] += step[i]
            tmax[i] += tdelta[i]
        return None

    # ---------- gameplay ----------
    def msg(self, text):
        self.msgs.append([text, self.app.time + 2.5])

    def damage(self):
        inv = self.inv
        if inv.count('wood_sword'):
            return 6
        if inv.count('wood_axe'):
            return 4
        if inv.count('wood_pickaxe'):
            return 3
        return 2

    def hardness(self, vid):
        inv = self.inv
        if vid in (7, 11):
            if inv.count('wood_axe'):
                return 0.75
            return 3.0 if vid == 7 else 2.0
        if vid == 4:
            return 1.0 if inv.count('wood_pickaxe') else 5.0
        if vid in LEAVES:
            return 0.25
        if vid == 5:
            return 0.3
        return 0.6

    def give(self, item, n=1):
        self.inv.add(item, n)
        self.msg(f"+{n} {ITEMS[item][0]}")

    def break_block(self, pos, vid):
        x, y, z = pos
        self.set_voxel(x, y, z, 0)
        if vid in LEAVES:
            if self.rng.random() < 0.2:
                self.give('stick', 1)
        elif vid in BREAK_DROP:
            self.give(BREAK_DROP[vid], 1)

    def place(self, block):
        items = self.inv.items()
        if not block or not items:
            return
        name = items[self.sel][0]
        bid = ITEMS[name][2]
        pos, prev, _ = block
        if not bid or prev is None:
            return
        x, y, z = prev
        if not (0 <= x < WX and 0 <= z < WZ and 0 <= y < WY):
            return
        ex = self.get_voxel(x, y, z)
        if ex and ex != WATER:
            return
        p = self.app.player.position
        if (int(math.floor(p.x)), int(math.floor(p.y)), int(math.floor(p.z))) == prev:
            return
        self.set_voxel(x, y, z, bid)
        self.inv.remove(name)
        self.sel = min(self.sel, max(0, len(self.inv.order) - 1))

    def start(self):
        n = len(self.inv.order)
        return max(0, min(self.sel - 4, n - 9))

    def handle_keys(self):
        keys = pg.key.get_pressed()
        down = {k for k in KEYS if keys[k]}
        edge = lambda k: k in down and k not in self.prev_keys
        n = len(self.inv.order)

        if edge(pg.K_c):
            self.menu = not self.menu
        if n:
            if edge(pg.K_z):
                self.sel = (self.sel - 1) % n
            if edge(pg.K_x):
                self.sel = (self.sel + 1) % n
        for i, k in enumerate(DIGITS):
            if not edge(k):
                continue
            if self.menu:
                if i < len(RECIPES):
                    r = RECIPES[i]
                    if self.inv.craft(r):
                        self.msg(f"Crafted {r[0]}")
                    else:
                        self.msg("Material kam hai")
            else:
                idx = self.start() + i
                if idx < n:
                    self.sel = idx
        self.sel = min(self.sel, max(0, len(self.inv.order) - 1))
        self.prev_keys = down

    def update(self):
        dt = min(self.app.delta_time * 0.001, 0.1)
        now = self.app.time
        self.msgs = [m for m in self.msgs if m[1] > now]
        self.attack_cd = max(0.0, self.attack_cd - dt)
        self.handle_keys()

        pl = self.app.player
        fw = getattr(pl, 'forward', None)
        if fw is None:
            fw = glm.vec3(math.cos(pl.yaw) * math.cos(pl.pitch), math.sin(pl.pitch),
                          math.sin(pl.yaw) * math.cos(pl.pitch))
        o = (pl.position.x, pl.position.y, pl.position.z)
        d = (fw.x, fw.y, fw.z)

        left, _, right = pg.mouse.get_pressed()
        block = self.raycast_block(o, d, 5.0)
        animal, at = self.animals.raycast(o, d, 4.0)
        bt = block[2] if block else 99.0

        if left and not self.menu:
            if animal is not None and at <= bt:
                self.mine_pos, self.progress = None, 0.0
                if self.attack_cd <= 0:
                    self.attack_cd = 0.45
                    for item, n in self.animals.hit(animal, self.damage(), o):
                        self.give(item, n)
            elif block and block[0][1] > 0:
                pos = block[0]
                if pos != self.mine_pos:
                    self.mine_pos, self.progress = pos, 0.0
                vid = self.get_voxel(*pos)
                self.progress += dt / self.hardness(vid)
                if self.progress >= 1.0:
                    self.break_block(pos, vid)
                    self.mine_pos, self.progress = None, 0.0
            else:
                self.mine_pos, self.progress = None, 0.0
        else:
            self.mine_pos, self.progress = None, 0.0

        if right and not self.prev_right and not self.menu:
            self.place(block)
        self.prev_right = right

        sig = (self.inv.version, self.sel, self.menu, int(self.progress * 40), tuple(m[0] for m in self.msgs))
        if sig != self.sig:
            self.sig = sig
            self.hud.draw(self.inv, self.sel, self.start(), self.menu, self.progress,
                          [m[0] for m in self.msgs])

    def render(self):
        self.hud.render()