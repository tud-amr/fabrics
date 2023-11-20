from typing import Dict
import pytest
from fabrics.components.robot_representation import (
    CollisionLinkDoesNotExistError,
    CollisionLinkUndefinedError,
    RobotRepresentation,
)
from fabrics.components.robot_representation.collision_link import (
    CollisionLink,
    Sphere,
    Capsule,
)


def test_robot_representation_errors():
    collision_links = {}
    self_collision_pairs = {"link1": ["link3", "link4"]}
    with pytest.raises(CollisionLinkUndefinedError):
        robot_representation = RobotRepresentation(
            collision_links=collision_links,
            self_collision_pairs=self_collision_pairs,
        )

    collision_links: Dict[str, CollisionLink] = {
        "link1": Sphere(radius=0.1),
        "link2": Sphere(radius=0.1),
        "link3": Sphere(radius=0.1),
        "link4": Sphere(radius=0.1),
    }
    robot_representation = RobotRepresentation(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
    )
    with pytest.raises(CollisionLinkDoesNotExistError):
        robot_representation.collision_link("link7")
    with pytest.raises(CollisionLinkDoesNotExistError):
        robot_representation.self_collision_pair("link2")


def test_robot_representation():
    self_collision_pairs = {"link1": ["link3", "link4"]}
    collision_links = {
        "link1": Sphere(radius=0.1),
        "link2": Capsule(radius=0.1, length=1.0),
        "link3": Sphere(radius=0.1),
        "link4": Sphere(radius=0.1),
    }
    robot_representation = RobotRepresentation(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
    )
    assert robot_representation.collision_link("link1").size[0] == 0.1
    assert robot_representation.collision_link("link2").size[0] == 0.1
    assert robot_representation.collision_link("link2").size[1] == 1.0
