def get_offset_into_lane(road_network, lane_id, position):
    lane = road_network.lane_by_id(lane_id)
    return road_network.offset_into_lane(lane, position[:2])


def get_position_from_lane_offset(road_network, lane_id, offset):
    lane = road_network.lane_by_id(lane_id)
    return road_network.world_coord_from_offset(lane, offset)


def get_lane_vector_at_offset(road_network, lane_id, offset):
    lane = road_network.lane_by_id(lane_id)
    return road_network.lane_vector_at_offset(lane, offset)

def is_further_along_route(road_network, f_lane_id, f_pos, o_lane_id, o_pos):
    o_lane = road_network.lane_by_id(o_lane_id)
    f_lane = road_network.lane_by_id(f_lane_id)
    
    o_off = get_offset_into_lane(road_network, o_lane_id, o_pos[:2])
    f_off = get_offset_into_lane(road_network, f_lane_id, f_pos[:2])

    outgoing = o_lane.getOutgoing()

    if f_lane in outgoing:
        return True
        
    if f_lane_id == o_lane_id:
        return f_off > o_off


    return False