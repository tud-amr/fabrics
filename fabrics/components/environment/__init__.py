from typing import Dict, List
import casadi as ca

from fabrics.helpers.geometric_primitives import Cuboid, GeometricPrimitive, Plane, Sphere

NumberPlanes = int
NumberSpheres = Dict[str, int]
NumberCuboids = Dict[str, int]

class Environment:
    _number_spheres: NumberSpheres
    _number_planes: NumberPlanes
    _number_cuboids: NumberCuboids
    _obstacles: List[GeometricPrimitive]

    def __init__(self, 
                 number_spheres: NumberSpheres,
                 number_planes: NumberPlanes,
                 number_cuboids: NumberCuboids):
        self._number_planes = number_planes
        self._number_spheres = number_spheres
        self._number_cuboids = number_cuboids
        self.generate_obstacles()

    def generate_obstacles(self):
        self._obstacles = []
        i = 0
        for sphere_type, number_spheres in self.number_spheres.items():
            for _ in range(number_spheres):
                if sphere_type == 'static':
                    obstacle = Sphere(f'obst_{i}')
                    obstacle.set_position(ca.SX.sym(f'x_{obstacle.name}', 3), free=True)
                    self._obstacles.append(obstacle)
                    i += 1
                elif sphere_type == 'dynamic':
                    obstacle = Sphere(f'dynamic_obst_{i}')
                    obstacle.set_position(ca.SX.sym(f'x_{obstacle.name}', 3), free=True)
                    self._obstacles.append(obstacle)
                    i += 1
        for cuboid_type, number_cuboids in self.number_cuboids.items():
            for _ in range(number_cuboids):
                if cuboid_type == 'static':
                    obstacle = Cuboid(f'obst_{i}')
                    obstacle.set_position(ca.SX.sym(f'x_{obstacle.name}', 3), free=True)
                    self._obstacles.append(obstacle)
                    i += 1
                elif cuboid_type == 'dynamic':
                    obstacle = Cuboid(f'dynamic_obst_{i}')
                    obstacle.set_position(ca.SX.sym(f'x_{obstacle.name}', 3), free=True)
                    self._obstacles.append(obstacle)
                    i += 1
        for j in range(self.number_planes):
            obstacle = Plane(f"constraint_{j}")
            obstacle.set_position(ca.SX.sym(f'{obstacle.name}', 3), free=True)
            self._obstacles.append(obstacle)

    @property
    def obstacles(self) -> List[GeometricPrimitive]:
        return self._obstacles

    @property
    def number_planes(self) -> NumberPlanes:
        return self._number_planes

    @property
    def number_spheres(self) -> NumberSpheres:
        return self._number_spheres

    @property
    def number_cuboids(self) -> NumberCuboids:
        return self._number_cuboids
