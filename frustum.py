from setting import *

class Frustum:
    def __init__(self,camera):
        self.cam = camera

    def is_on_frustum(self,chunk):
        return True