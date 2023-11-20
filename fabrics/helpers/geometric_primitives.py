from typing import List, Tuple, Dict
import casadi as ca
from urdfenvs.urdf_common.urdf_env import Plane
from fabrics.helpers.distances import sphere_to_plane, sphere_to_sphere, cuboid_to_sphere

class DistanceNotImplementedError(Exception):
    def __init__(self, primitive_1: "GeometricPrimitive", primitive_2: "GeometricPrimitive"):
        message=f"Distance between {type(primitive_1)} and {type(primitive_2)} not implemented"
        super().__init__(message)


class GeometricPrimitive:
    _name: str
    _position: ca.SX
    _parameters: Dict[str, ca.SX]

    def __init__(self, name: str):
        self._name = name
        self._position = ca.SX()
        self._parameters = {}


    @property
    def position(self) -> ca.SX:
        return self._position

    def set_position(self, position: ca.SX, free: bool = False) -> None:
        self._position = position
        if free:
            self._parameters[position[0].name()[:-2]] = position

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> List[float]:
        return []

    @property
    def parameters(self) -> Dict[str, ca.SX]:
        self._parameters.update(self.sym_size)
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

    def distance(self, primitive: GeometricPrimitive) -> ca.SX:
        raise DistanceNotImplementedError(self, primitive)


class Sphere(GeometricPrimitive):
    _radius: float
    _sym_radius: ca.SX

    def __init__(self, name: str, radius: float = 0):
        super().__init__(name)
        self._radius = radius
        self._sym_radius = ca.SX.sym(f"radius_{self.name}", 1)

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
        if isinstance(primitive, Plane):
            return sphere_to_plane(
                    self.position,
                    primitive.sym_plane_equation,
                    self.sym_radius
            )
        if isinstance(primitive, Cuboid):
            return cuboid_to_sphere(
                    primitive.position,
                    self.position,
                    primitive.sym_sizes,
                    self.sym_radius,
            )
        else:
            raise DistanceNotImplementedError(self, primitive)

class Cuboid(GeometricPrimitive):
    _sizes: List[float]
    _sym_sizes: ca.SX

    def __init__(self, name: str, sizes: List[float] = [0.0, 0.0, 0.0]):
        super().__init__(name)
        self._sizes = sizes
        self._sym_sizes = ca.SX.sym(f"sizes_{self.name}", 3)

    @property
    def size(self) -> List[float]:
        return self._sizes

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {self._sym_sizes[0].name()[:-2]: self.sym_sizes}

    @property
    def sizes(self) -> float:
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

