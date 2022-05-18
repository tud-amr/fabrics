from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper_deserialized


class SerializedFabricPlanner(ParameterizedFabricPlanner):
    def __init__(self, file_name: str):
        self._funs = CasadiFunctionWrapper_deserialized(file_name)
        self._isload = True

    #Disable all functions to compose the tree of fabrics.

    def initialize_joint_variables(self):
        print("Deserialized planner cannot be changed.")
        pass

    def set_base_geometry(self):
        print("Deserialized planner cannot be changed.")
        pass

    def add_geometry(self):
        print("Deserialized planner cannot be changed.")
        pass

    def add_weighted_geometry(self):
        print("Deserialized planner cannot be changed.")
        pass

    def add_leaf(self):
        print("Deserialized planner cannot be changed.")
        pass

    def add_forcing_geometry(self):
        print("Deserialized planner cannot be changed.")
        pass

    def set_execution_energy(self):
        print("Deserialized planner cannot be changed.")
        pass

    def set_speed_control(self):
        print("Deserialized planner cannot be changed.")
        pass

    def set_components(self):
        print("Deserialized planner cannot be changed.")
        pass

    def concretize(self):
        print("Deserialized planner cannot be changed.")
        pass
