import carla
from typing import NamedTuple, List

is_vehicle = lambda actor: "vehicle" in actor.type_id
is_pedestrian = lambda actor: "walker" in actor.type_id
is_traffic_light = lambda actor: "traffic_light" in actor.type_id


CONSTRUCTION_TYPES = {
    "static.prop.container",
    "static.prop.warningaccident",
    "static.prop.warningconstruction",
    "static.prop.constructioncone",
    "static.prop.trafficwarning",
    "static.prop.streetbarrier",
    "static.prop.warningaccident"
    # "static.prop.mesh"
}

EMERGENCY_TYPES = {
    "police",
    "ambulance",
    "firetruck"
}

BIKE_TYPES = {
    "crossbike",
    "diamondback",
    "gazelle"
}

def is_construction(actor):
    for type in CONSTRUCTION_TYPES:
        if type in actor.type_id:
            return True
    return False

def is_emergency(actor):
    for type in EMERGENCY_TYPES:
        if type in actor.type_id:
            return True
    return False

def is_bike(actor):
    for type in BIKE_TYPES:
        if type in actor.type_id:
            return True
    return False


class SegregatedActors(NamedTuple):
    vehicles: List[carla.Actor]
    pedestrians: List[carla.Actor]
    traffic_lights: List[carla.Actor]
    constructions: List[carla.Actor]
    emergencys: List[carla.Actor]
    bikes: List[carla.Actor]


def segregate_by_type(actors: List[carla.Actor]) -> SegregatedActors:
    vehicles = []
    pedestrians = []
    traffic_lights = []
    constructions = []
    emergencys = []
    bikes = []
    for actor in actors:
        if is_emergency(actor):
            emergencys.append(actor)
        elif is_construction(actor):
            constructions.append(actor)
        elif is_pedestrian(actor):
            pedestrians.append(actor)
        elif is_traffic_light(actor):
            traffic_lights.append(actor)
        elif is_bike(actor):
            bikes.append(actor)
        elif is_vehicle(actor):
            vehicles.append(actor)
    
        # if "static.vegetation" in actor.type_id:
        # print(actor.type_id)

    return SegregatedActors(vehicles, pedestrians, traffic_lights, constructions, emergencys, bikes)


def query_all(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    all_actors = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            all_actors.append(actor)
    return all_actors