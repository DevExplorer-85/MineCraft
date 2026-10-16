import numpy as np
from numba import njit
from setting import WORLD_W, WORLD_D, CHUNK_SIZE
from terrain_gen import get_height


@njit(cache=True)
def _build(w, d):
    out = np.zeros((d, w), dtype=np.uint8)
    for z in range(d):
        for x in range(w):
            out[z, x] = get_height(x, z)
    return out


def build_heightmap():
    return _build(WORLD_W * CHUNK_SIZE, WORLD_D * CHUNK_SIZE)