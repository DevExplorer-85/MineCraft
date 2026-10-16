from setting import*
from meshes.base_mesh import BaseMesh

class QuadMesh(BaseMesh):
    def __init__(self,app):
        super().__init__()

        self.app=app
        self.ctx=app.ctx
        self.program=app.shader_program.quad
        self.vbo_format = '3f 3f'
        self.attrs = ('in_position', 'in_color')
        self.vao = self.get_vao()
        print("QuadMesh VAO =", self.vao)

    def get_vertex_data(self):
        vertices = np.array([
            (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0), (-0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0), (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0)
        ], dtype=np.float32)

        colors = np.array([
            (0, 1, 0), (1, 0, 0), (1, 1, 0),
            (0, 1, 0), (1, 1, 0), (0, 0, 1)
        ], dtype=np.float32)

        vertex_data = np.hstack((vertices, colors))

        print(vertex_data.shape)
        print(vertex_data)

        return vertex_data