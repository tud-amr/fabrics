from fabrics.components.environment import Environment


def test_environment():
    environment_configuration = {
        'number_spheres':{'static': 3, 'dynamic': 3},
        'number_planes':2,
        'number_cuboids':{'static': 1, 'dynamic': 0},
    }
    environment = Environment(**environment_configuration)
    assert environment.number_planes == 2
