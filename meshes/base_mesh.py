import numpy as np


class BaseMesh:
    def __init__(self):
        self.ctx = None
        self.program = None
        self.vbo_format = None
        self.attrs: tuple[str, ...] = None
        self.vao = None

    def get_vertex_data(self) -> np.array: ...

    def get_vao(self):
        vertex_data = self.get_vertex_data()

        # agar is chunk mein koi visible face nahi hai, VAO skip kar do
        if vertex_data is None or len(vertex_data) == 0:
            return None

        vbo = self.ctx.buffer(vertex_data)

        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, self.vbo_format, *self.attrs)]
        )

        return vao

    def render(self):
        if self.vao is None:
            return   # kuch render karne layak nahi, skip
        self.vao.render()