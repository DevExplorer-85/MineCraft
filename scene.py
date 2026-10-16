import moderngl as mgl
from setting import *
from world import World
from world_objects.voxel_marker import VoxelMarker
from world_objects.cloud import Clouds
from heightmap import build_heightmap
from world_objects.sun import Sun
from world_objects.animal import Animals
from world_objects.animal import Animals
from world_objects.interaction import Interaction

HEIGHTMAP_UNIT = 3
UNDERWATER_FOG = (0.05, 0.25, 0.45)


class Scene:
    def __init__(self, app):
        self.app = app
        self.world = World(self.app)
        self.sun = Sun(app)
        self.animals = Animals(app, self.world)
        self.voxel_marker = VoxelMarker(self.world.voxel_handler)
        self.clouds = Clouds(app)
        self.animals = Animals(app, self.world)
        self.interaction = Interaction(app, self.world, self.animals)
        self.bg_color = BG_COLOR
        self.underwater = False

        hm = build_heightmap()
        self.heightmap = app.ctx.texture((hm.shape[1], hm.shape[0]), 1, hm.tobytes(), alignment=1)
        self.heightmap.filter = (mgl.NEAREST, mgl.NEAREST)
        app.shader_program.chunk['u_heightmap'].value = HEIGHTMAP_UNIT

    def is_underwater(self):
        p = self.app.player.position
        x, y, z = int(p.x), int(p.y), int(p.z)
        if not (0 <= x < WORLD_W * CHUNK_SIZE and 0 <= z < WORLD_D * CHUNK_SIZE
                and 0 <= y < WORLD_H * CHUNK_SIZE):
            return False
        ci = x // CHUNK_SIZE + WORLD_W * (z // CHUNK_SIZE) + WORLD_AREA * (y // CHUNK_SIZE)
        vi = x % CHUNK_SIZE + (z % CHUNK_SIZE) * CHUNK_SIZE + (y % CHUNK_SIZE) * CHUNK_AREA
        return self.world.voxels[ci][vi] == WATER

    def update(self):
        self.world.update()
        self.voxel_marker.update()
        self.underwater = self.is_underwater()
        self.bg_color = glm.vec3(*UNDERWATER_FOG) if self.underwater else BG_COLOR
        
        self.animals.update()
        self.interaction.update()

    def render(self):
        prog = self.app.shader_program.chunk
        prog['u_underwater'].value = 1.0 if self.underwater else 0.0
        prog['u_cam_pos'].value = tuple(self.app.player.position)
        prog['u_time'].value = float(getattr(self.app, 'time', 0.0))

        self.heightmap.use(location=HEIGHTMAP_UNIT)

        # pass 0: zameen, trees, sab kuch (paani ke bina)
        prog['u_pass'].value = 0
        self.world.render()

        # pass 1: sirf paani, transparent, depth write off
        prog['u_pass'].value = 1
        self.app.ctx.fbo.depth_mask = False
        self.world.render()
        self.app.ctx.fbo.depth_mask = True

        self.world.render()

        if not self.underwater:
            self.animals.render()
            self.sun.render()
            self.clouds.render()

        self.interaction.render()

        if not self.underwater:
            self.clouds.render()

        if not self.underwater:
            self.sun.render()
            self.clouds.render()