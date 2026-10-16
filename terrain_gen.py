# terrain_gen.py
import math
from numba import njit
from opensimplex.internals import _noise2, _noise3, _init
from setting import CENTER_XZ, CHUNK_SIZE, WORLD_H, SEED as WORLD_SEED

# Existing vertical world capacity: 3 chunks x 32 = 96 blocks
MAX_H = CHUNK_SIZE * WORLD_H

# Keep noise deterministic for a consistent world.
SEED = WORLD_SEED
_perm, _perm_grad = _init(seed=SEED)


@njit(cache=True)
def noise2(x, z):
    return _noise2(x, z, _perm)


@njit(cache=True)
def noise3(x, y, z):
    return _noise3(x, y, z, _perm, _perm_grad)


@njit(cache=True)
def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def fbm(x, z, octaves, freq):
    total = 0.0
    amp = 1.0
    norm = 0.0

    for _ in range(octaves):
        total += noise2(x * freq, z * freq) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0

    return total / norm


@njit(cache=True)
def ridged(x, z, octaves, freq):
    total = 0.0
    amp = 1.0
    norm = 0.0
    weight = 1.0

    for _ in range(octaves):
        n = 1.0 - abs(noise2(x * freq, z * freq))
        n = n * n * weight
        weight = min(max(n * 2.0, 0.0), 1.0)

        total += n * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0

    return total / norm if norm > 0.0 else 0.0


@njit(cache=True)
def get_layers(x, z):
    # 1. Domain warp
    warp_x = fbm(x + 1200.0, z + 300.0, 3, 0.004)
    warp_z = fbm(x - 400.0, z + 900.0, 3, 0.004)
    wx = x + 70.0 * warp_x
    wz = z + 70.0 * warp_z

    # 2. Base terrain
    continental = 0.5 + 0.5 * fbm(wx, wz, 4, 0.0025)
    medium = fbm(wx + 250.0, wz - 150.0, 4, 0.009)
    detail = fbm(wx - 600.0, wz + 500.0, 3, 0.05)

    base = 24.0 + 6.0 * continental + 5.0 * medium

    # 3. Basin
    dx = (wx - (CENTER_XZ + 25.0)) / 300.0
    dz = (wz - (CENTER_XZ - 15.0)) / 230.0
    d = math.sqrt(dx * dx + dz * dz)
    d += 0.35 * fbm(wx + 300.0, wz - 700.0, 3, 0.006)
    basin = 1.0 - smoothstep(0.35, 0.95, d)

    # 4. Mountain masses
    mass_noise = 0.5 + 0.5 * fbm(wx - 900.0, wz + 450.0, 4, 0.003)
    mass = smoothstep(0.44, 0.64, mass_noise)
    mass *= 1.0 - 0.9 * basin

    # 5. Massif + ridges
    ridge_noise = ridged(wx, wz, 5, 0.007)
    peaks = ridge_noise ** 1.5
    mountains = mass * (14.0 + 22.0 * mass + 38.0 * peaks)

    # 6. Foothills
    foothill_zone = smoothstep(0.30, 0.55, mass_noise) * (1.0 - basin)
    foothills = foothill_zone * (4.0 + 8.0 * (0.5 + 0.5 * medium))

    # 7. Final: soft compression
    height = base - 22.0 * basin + foothills + mountains + 3.0 * detail

    limit = MAX_H - 8.0
    knee = limit - 14.0
    if height > knee:
        over = height - knee
        height = knee + 14.0 * (1.0 - math.exp(-over / 14.0))
    height = max(5.0, height)

    final = max(1, min(int(height), MAX_H - 1))
    return 1.0, continental, mass, ridge_noise, height, final


@njit(cache=True)
def get_height(x, z):
    return int(get_layers(x, z)[5])


@njit(cache=True)
def is_cave(x, y, z, surface_h):
    depth = surface_h - y

    # zameen ke bilkul neeche aur world floor par cave nahi
    if y < 4 or depth < 5:
        return False

    # surface ke paas kam caves
    fade = min(1.0, (depth - 5) / 10.0)

    # 1. Tunnels: do noise fields ke zero ke paas ka hissa
    a = noise3(x * 0.02, y * 0.035, z * 0.02)
    b = noise3(x * 0.02 + 500.0, y * 0.035 + 200.0, z * 0.02 - 300.0)
    width = 0.14 * fade
    if abs(a) < width and abs(b) < width:
        return True

    # 2. Caverns: bade khule hall
    c = noise3(x * 0.035 - 800.0, y * 0.05, z * 0.035 + 400.0)
    if c > 0.55 + 0.25 * (1.0 - fade):
        return True

    return False

TREE_CELL = 8


@njit(cache=True)
def hash2(x, z, s):
    h = (x * 374761393 + z * 668265263 + s * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


@njit(cache=True)
def tree_in_cell(gx, gz):
    # return: ox, oz, base_height, trunk_height  (base_height = -1 matlab tree nahi)
    h = hash2(gx, gz, 11)
    ox = gx * TREE_CELL + 2 + h % 6
    oz = gz * TREE_CELL + 2 + (h >> 8) % 6

    # groves: kahin zyada, kahin kam
    grove = 0.5 + 0.5 * fbm(ox, oz, 2, 0.008)
    chance = 12.0 + 55.0 * smoothstep(0.4, 0.7, grove)      # percent
    if (h >> 16) % 100 >= chance:
        return 0, 0, -1, 0

    oh = get_height(ox, oz)

    # sirf ghaas wali, kam height par
    if oh <= 17 or oh > 62:
        return 0, 0, -1, 0

    # sirf lagbhag samatal zameen par
    if (abs(oh - get_height(ox + 1, oz)) > 2 or abs(oh - get_height(ox - 1, oz)) > 2 or
            abs(oh - get_height(ox, oz + 1)) > 2 or abs(oh - get_height(ox, oz - 1)) > 2):
        return 0, 0, -1, 0

    th = 4 + (h >> 24) % 2      # 4 ya 5      # 3 ya 4 block (pehle 6-8)
    return ox, oz, oh, th

@njit(cache=True)
def tree_kind(ox, oz, oh):
    # 0 = cherry (pink), 1 = oak (green), 2 = spruce (pine)
    n = fbm(ox, oz, 2, 0.01)             # border ko tedha-medha karta hai
    if oh >= 46 + 5.0 * n:
        return 2
    if oh >= 32 + 4.0 * n:
        return 1
    # neeche wale ground par kabhi-kabhi oak bhi (~15%)
    if hash2(ox, oz, 31) % 100 < 15:
        return 1
    return 0

@njit(cache=True)
def tree_voxel(dx, dy, dz, th, seed, ox, oz, kind):
    # return: 7 = wood, 6 = pink leaves, 9 = oak leaves, 10 = spruce leaves, 0 = khali
    adx = abs(dx)
    adz = abs(dz)

    # ---------- SPRUCE: lamba tana + cone ----------
    if kind == 2:
        top = th + 2
        if dx == 0 and dz == 0 and dy < top:
            return 7
        if dy < 2 or dy > top:
            return 0

        rel = top - dy                      # 0 = sabse upar nok
        r = min(3, (rel + 1) // 2)
        if rel >= 2 and rel % 2 == 0:       # tiers: har doosri layer thodi andar
            r = max(r - 1, 1)

        if r == 0:
            if dx == 0 and dz == 0:
                return 10
            return 0

        d2 = adx * adx + adz * adz
        if d2 > r * r + 1:
            return 0
        if d2 > r * r - 1 and hash2(ox + dx * 131 + dy * 7919, oz + dz, 9) % 4 == 0:
            return 0
        return 10

    # ---------- CHERRY (6) aur OAK (9): gol layered canopy ----------
    leaf = 6
    if kind == 1:
        leaf = 9

    if dx == 0 and dz == 0 and dy <= th:
        return 7

    if dy == th - 2 or dy == th - 1:
        if adx > 2 or adz > 2:
            return 0
        if adx == 2 and adz == 2:
            if hash2(ox + dx * 131 + dy * 7919, oz + dz, 9) % 2 == 0:
                return 0
        return leaf

    if dy == th:
        if adx > 1 or adz > 1:
            return 0
        if adx == 1 and adz == 1:
            if hash2(ox + dx * 131 + dy * 7919, oz + dz, 9) % 4 == 0:
                return 0
        return leaf

    if dy == th + 1:
        if adx + adz <= 1:
            return leaf
        return 0

    return 0