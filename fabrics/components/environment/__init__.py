from typing import Dict

NumberPlanes = int
NumberSpheres = Dict[str, int]
NumberCuboids = Dict[str, int]

class Environment:
    _number_spheres: NumberSpheres
    _number_planes: NumberPlanes
    _number_cuboids: NumberCuboids

    def __init__(self, 
                 number_spheres: NumberSpheres,
                 number_planes: NumberPlanes,
                 number_cuboids: NumberCuboids):
        self._number_planes = number_planes
        self._number_spheres = number_spheres
        self._number_cuboids = number_cuboids

    @property
    def number_planes(self) -> NumberPlanes:
        return self._number_planes

    @property
    def number_spheres(self) -> NumberSpheres:
        return self._number_spheres

    @property
    def number_cuboids(self) -> NumberCuboids:
        return self._number_cuboids
