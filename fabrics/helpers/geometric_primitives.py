from typing import List, Tuple, Dict
import casadi as ca
from fabrics.helpers.distances import sphere_to_sphere


class GeometricPrimitive:
    _name: str

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> List[float]:
        return []

    @property
    def sym_size(self) -> Dict[str, ca.SX]:
        return {}

class Capsule(GeometricPrimitive):
    _radius: float
    _length: float
    _sym_radius: ca.SX
    _sym_length: ca.SX

    def __init__(self, name: str, radius: float, length: float):
        super().__init__(name)
        self._radius = radius
        self._length = length
        self._sym_radius = ca.SX.sym(f"{self._name}_radius", 1)
        self._sym_length = ca.SX.sym(f"{self._name}_length", 1)

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


class Sphere(GeometricPrimitive):
    _radius: float
    _sym_radius: ca.SX

    def __init__(self, name: str, radius: float):
        super().__init__(name)
        self._radius = radius
        self._sym_radius = ca.SX.sym(f"{self._name}_radius", 1)

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

