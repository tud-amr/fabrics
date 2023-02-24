from fabrics.planner.serialized_planner import SerializedFabricPlanner

from panda_ring import run_panda_ring_example


def set_planner(file_name):
    """
    Initializes the fabric planner for the panda robot from the file.

    Params
    ---------
    file_name: str
        File name to which the planner has been serialized.

    """

    planner = SerializedFabricPlanner(
        file_name,
    )
    return planner


def run_panda_ring_serialized_example(n_steps=5000, render=True):
    planner = set_planner("serialized_10.pbz2")
    return run_panda_ring_example(n_steps=n_steps, render=render, planner=planner)


if __name__ == "__main__":
    res = run_panda_ring_serialized_example(n_steps=5000)
