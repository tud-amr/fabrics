from typing import Dict, List, Union
from fabrics.helpers.geometric_primitives import GeometricPrimitive

class CollisionLinkDoesNotExistError(Exception):
    def __init__(self, collision_link_name: str):
        message = f"Collision link with name {collision_link_name} does not exist."
        super().__init__(message)

class CollisionLinkUndefinedError(Exception):
    def __init__(self, collision_link_name: str):
        message = f"Collision link with name {collision_link_name} does not exist but is used in the self_collision_pairs."
        super().__init__(message)

SelfCollisionPairsType = Dict[str, List[str]]
CollisionLinksType = Dict[str, GeometricPrimitive]

class RobotRepresentation:
    _collision_links: CollisionLinksType
    _self_collision_pairs: SelfCollisionPairsType

    def __init__(self,
                 collision_links: Union[CollisionLinksType, None],
                 self_collision_pairs: Union[SelfCollisionPairsType, None]):
        if not collision_links:
            self._collision_links = {}
        else:
            self._collision_links = collision_links
        if not self_collision_pairs:
            self._self_collision_pairs = {}
        else:
            self._self_collision_pairs = self_collision_pairs
        self.check_self_collision_pairs()

    def check_self_collision_pairs(self):
        for link_name, paired_links_names in self._self_collision_pairs.items():
            if link_name not in self._collision_links:
                raise CollisionLinkUndefinedError(link_name)
            for paired_link_name in paired_links_names:
                if paired_link_name not in self._collision_links:
                    raise CollisionLinkUndefinedError(paired_link_name)

    @property
    def collision_links(self) -> CollisionLinksType:
        return self._collision_links

    def collision_link(self, name: str) -> GeometricPrimitive:
        if name in self._collision_links:
            return self._collision_links[name]
        raise CollisionLinkDoesNotExistError(name)

    @property
    def self_collision_pairs(self) -> SelfCollisionPairsType:
        return self._self_collision_pairs

    def self_collision_pair(self, name: str) -> List[str]:
        if name in self._self_collision_pairs:
            return self._self_collision_pairs[name]
        raise CollisionLinkDoesNotExistError(name)

