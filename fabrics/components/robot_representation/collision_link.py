class CollisionLink:
    pass

class Capsule(CollisionLink):
    _radius: float
    _length: float
    def __init__(self, radius: float, length: float):
        self._radius = radius
        self._length = length

    @property
    def radius(self) -> float:
         return self._radius

    @property
    def length(self) -> float:
         return self._length

class Sphere(CollisionLink):
    _radius: float
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self) -> float:
         return self._radius

