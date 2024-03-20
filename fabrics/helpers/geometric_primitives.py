from typing import List, Tuple, Dict, Union
import casadi as ca
import numpy as np
from fabrics.helpers.distances import capsule_to_plane, capsule_to_sphere, cuboid_to_capsule, sphere_to_plane, sphere_to_sphere, cuboid_to_sphere

class DistanceNotImplementedError(Exception):
    def __init__(self, primitive_1: "GeometricPrimitive", primitive_2: "GeometricPrimitive"):
        message=f"Distance between {primitive_1} and {primitive_2} not implemented"
        super().__init__(message)



class GeometricPrimitive:
    _name: str
    _origin: ca.SX
    _parameters: Dict[str, Union[np.ndarray, float]]
    _sym_parameters: Dict[str, ca.SX]

    def __init__(self, name: str):
        self._name = name
        self._position = ca.SX()
        self._sym_parameters = {}
        self._parameters = {}
        self._origin = ca.SX(np.identity(4))

    def __str__(self) -> str:
        return self.__class__.__name__ + ": " + self._name

    @property
    def position(self) -> ca.SX:
        return self._origin[0:3, 3]

    def set_position(self, position: ca.SX, free: bool = False) -> None:
        self._origin[0:3,3] = position
        if free:
            self._sym_parameters[position[0].name()[:-2]] = position

    @property
    def origin(self) -> ca.SX:
        return self._origin

    def set_origin(self, origin: ca.SX, free: bool = False) -> None:
        self._origin = origin
        if free:
            self._sym_parameters[origin[0][0].name()[:-2]] = origin

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> List[float]:
        return []

    @property
    def sym_parameters(self) -> Dict[str, ca.SX]:
        self._sym_parameters.update(self.sym_size)
        return self._sym_parameters

    @property
    def parameters(self) -> Dict[str, Union[float, np.ndarray]]:
        return self._parameters

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {}

    def distance(self, primitive: "GeometricPrimitive") -> ca.SX:
        pass

class Capsule(GeometricPrimitive):
    _radius: float
    _length: float
    _sym_radius: ca.SX
    _sym_length: ca.SX

    def __init__(self, name: str, radius: float = 0, length: float = 0):
        super().__init__(name)
        self._radius = radius
        self._length = length
        self._sym_radius = ca.SX.sym(f"radius_{self.name}", 1)
        self._sym_length = ca.SX.sym(f"length_{self.name}", 1)
        self._parameters[f'radius_{self.name}'] = self._radius
        self._parameters[f'length_{self.name}'] = self._length

    @property
    def size(self) -> List[float]:
        return [self.radius, self.length]

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {
            self.sym_radius.name(): self.sym_radius,
            self.sym_length.name(): self.sym_length,
        }

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def sym_radius(self) -> ca.SX:
        return self._sym_radius

    @property
    def sym_length(self) -> ca.SX:
        return self._sym_length

    @property
    def length(self) -> float:
        return self._length

    @property
    def centers(self) -> List[ca.SX]:
        tf_origin_center_0 = ca.SX(np.identity(4))
        tf_origin_center_0[2, 3] = self.sym_length / 2
        tf_center_0 = ca.mtimes(self.origin, tf_origin_center_0)
        tf_origin_center_1 = ca.SX(np.identity(4))
        tf_origin_center_1[2, 3] = - self.sym_length / 2
        tf_center_1 = ca.mtimes(self.origin, tf_origin_center_1)
        return [tf_center_0[0:3, 3], tf_center_1[0:3, 3]]


    def distance(self, primitive: GeometricPrimitive) -> ca.SX:
        if isinstance(primitive, Sphere):
            return capsule_to_sphere(
                    self.centers,
                    primitive.position,
                    self.sym_radius,
                    primitive.sym_radius
            )
        elif isinstance(primitive, Cuboid):
            return cuboid_to_capsule(
                    primitive.position,
                    self.centers,
                    primitive.sym_sizes,
                    self.sym_radius
            )
        elif isinstance(primitive, Plane):
            return capsule_to_plane(
                    self.centers,
                    primitive.sym_plane_equation,
                    self.sym_radius
            )
        raise DistanceNotImplementedError(self, primitive)


class Sphere(GeometricPrimitive):
    _radius: float
    _sym_radius: ca.SX

    def __init__(self, name: str, radius: float = 0):
        super().__init__(name)
        self._radius = radius
        self._sym_radius = ca.SX.sym(f"radius_{self.name}", 1)
        self._parameters[f'radius_{self.name}'] = self._radius

    @property
    def size(self) -> List[float]:
        return [self.radius]

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {
            self.sym_radius.name(): self.sym_radius,
        }

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def sym_radius(self) -> ca.SX:
        return self._sym_radius

    def distance(self, primitive: GeometricPrimitive) -> ca.SX:
        if isinstance(primitive, Sphere):
            return sphere_to_sphere(
                    self.position,
                    primitive.position,
                    self.sym_radius,
                    primitive.sym_radius
            )
        elif isinstance(primitive, Plane):
            return sphere_to_plane(
                    self.position,
                    primitive.sym_plane_equation,
                    self.sym_radius
            )
        elif isinstance(primitive, Cuboid):
            return cuboid_to_sphere(
                    primitive.position,
                    self.position,
                    primitive.sym_sizes,
                    self.sym_radius,
            )
        raise DistanceNotImplementedError(self, primitive)

class Cuboid(GeometricPrimitive):
    _sizes: List[float]
    _sym_sizes: ca.SX

    def __init__(self, name: str, sizes: List[float] = [0.0, 0.0, 0.0]):
        super().__init__(name)
        self._sizes = sizes
        self._sym_sizes = ca.SX.sym(f"sizes_{self.name}", 3)
        self._parameters[f'sizes_{self.name}'] = self._sizes

    @property
    def size(self) -> List[float]:
        return self._sizes

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {self._sym_sizes[0].name()[:-2]: self.sym_sizes}

    @property
    def sizes(self) -> List[float]:
        return self._sizes

    @property
    def sym_sizes(self) -> ca.SX:
        return self._sym_sizes

    def distance(self, primitive: GeometricPrimitive) -> ca.SX:
        raise DistanceNotImplementedError(self, primitive)

class Plane(GeometricPrimitive):
    _plane_equation: List[float]
    _sym_plane_equation: ca.SX

    def __init__(self, name: str, plane_equation: List[float] = [0, 0, 0, 1]):
        super().__init__(name)
        self._plane_equation = plane_equation
        self._sym_plane_equation = ca.SX.sym(f"{self.name}", 4)
        self._parameters[f'{self.name}'] = self._plane_equation

    @property
    def size(self) -> List[float]:
        return self.plane_equation

    @property
    def sym_size(self) -> List[float]:
        return {f"{self.name}": self.sym_plane_equation}

    @property
    def plane_equation(self) -> List[float]:
        return self._plane_equation

    @property
    def sym_plane_equation(self) -> ca.SX:
        return self._sym_plane_equation

    def distance(self, primitive: GeometricPrimitive) -> ca.SX:
        raise DistanceNotImplementedError(self, primitive)

