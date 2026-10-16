from setting import *
from meshes.chunk_mesh import ChunkMesh
from terrain_gen import *


class Chunk:
    def __init__(self, world, position):
        self.app = world.app
        self.world = world
        self.position = position
        self.m_model = self.get_model_matrix()
        self.voxels: np.array = self.build_voxels()
        self.mesh: ChunkMesh = None
        self.is_empty = not np.any(self.voxels)
        self.build_mesh()
        self.center = (glm.vec3(self.position) + 0.5) * CHUNK_SIZE
        self.is_on_frustum = self.app.player.frustum.is_on_frustum

    def get_model_matrix(self):
        return glm.translate(
            glm.mat4(),
            glm.vec3(self.position) * CHUNK_SIZE
        )

    def set_uniform(self):
        self.mesh.program['m_model'].write(self.m_model)

    def build_mesh(self):
        self.mesh = ChunkMesh(self)

    def render(self):
        if not self.is_empty and self.is_on_frustum(self):
            self.set_uniform()
            self.mesh.render()

    def build_voxels(self):
        voxels = np.zeros(CHUNK_VOL, dtype=np.uint8)

        cx, cy, cz = glm.ivec3(self.position) * CHUNK_SIZE
        self.generate_terrain(voxels, cx, cy, cz)
        self.generate_trees(voxels, cx, cy, cz)

        return voxels

    @staticmethod
    @njit
    def generate_terrain(voxels, cx, cy, cz):
        for x in range(CHUNK_SIZE):
            wx = x + cx

            for z in range(CHUNK_SIZE):
                wz = z + cz
                world_height = get_height(wx, wz)

                # --- water: zameen ke upar WATER_LVL tak ---
                if world_height < WATER_LVL:
                    for wy in range(world_height, WATER_LVL):
                        yy = wy - cy
                        if 0 <= yy < CHUNK_SIZE:
                            voxels[x + CHUNK_SIZE * z + CHUNK_AREA * yy] = 8

                local_height = max(0, min(world_height - cy, CHUNK_SIZE))
                if local_height == 0:
                    continue

                h_left = get_height(wx - 1, wz)
                h_right = get_height(wx + 1, wz)
                h_back = get_height(wx, wz - 1)
                h_front = get_height(wx, wz + 1)
                slope = max(
                    abs(world_height - h_left),
                    abs(world_height - h_right),
                    abs(world_height - h_back),
                    abs(world_height - h_front)
                )

                snow_line = SNOW_LVL + 14.0 + 7.0 * fbm(wx, wz, 3, 0.03)
                patch = 0.5 + 0.5 * fbm(wx + 900.0, wz - 400.0, 2, 0.12)

                is_bed = world_height <= WATER_LVL + 1

                if is_bed:
                    surface = 1                       # Sand (bed + beach)
                elif world_height >= snow_line + 6.0:
                    surface = 4 if slope >= 5 else 5
                elif world_height >= snow_line - 6.0:
                    t = (world_height - (snow_line - 6.0)) / 12.0
                    if slope >= 4:
                        surface = 4
                    elif patch < t:
                        surface = 5
                    else:
                        surface = 4 if slope >= 3 else 2
                elif slope >= 3:
                    surface = 4                       # Stone
                else:
                    surface = 2                       # Grass

                for y in range(local_height):
                    wy = y + cy
                    depth = world_height - 1 - wy

                    if depth == 0:
                        voxel_id = surface
                    elif depth < 4:
                        if is_bed and depth < 3:
                            voxel_id = 1              # sand bed
                        else:
                            voxel_id = 3              # Dirt
                    else:
                        voxel_id = 4                  # Stone

                    # cave: voxel khali chhod do
                    if depth >= 4 and is_cave(wx, wy, wz, world_height):
                        continue

                    voxels[x + CHUNK_SIZE * z + CHUNK_AREA * y] = voxel_id

    @staticmethod
    @njit
    def generate_trees(voxels, cx, cy, cz):
        R = 6
        gx0 = (cx - R) // TREE_CELL
        gx1 = (cx + CHUNK_SIZE + R) // TREE_CELL
        gz0 = (cz - R) // TREE_CELL
        gz1 = (cz + CHUNK_SIZE + R) // TREE_CELL

        for gx in range(gx0, gx1 + 1):
            for gz in range(gz0, gz1 + 1):
                ox, oz, oh, th = tree_in_cell(gx, gz)
                if oh < 0:
                    continue

                kind = tree_kind(ox, oz, oh)
                if kind == 2:
                    th += 3                      # spruce zyada lamba

                seed = hash2(ox, oz, 77)

                for dx in range(-R, R + 1):
                    x = ox + dx - cx
                    if x < 0 or x >= CHUNK_SIZE:
                        continue
                    for dz in range(-R, R + 1):
                        z = oz + dz - cz
                        if z < 0 or z >= CHUNK_SIZE:
                            continue
                        for dy in range(0, th + 4):
                            y = oh + dy - cy
                            if y < 0 or y >= CHUNK_SIZE:
                                continue

                            v = tree_voxel(dx, dy, dz, th, seed, ox, oz, kind)
                            if v == 0:
                                continue

                            idx = x + CHUNK_SIZE * z + CHUNK_AREA * y
                            if v == 7 or voxels[idx] == 0:
                                voxels[idx] = v