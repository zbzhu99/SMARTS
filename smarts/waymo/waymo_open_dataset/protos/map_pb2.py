# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: smarts/waymo/waymo_open_dataset/protos/map.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n0smarts/waymo/waymo_open_dataset/protos/map.proto\x12\x12waymo.open_dataset\"u\n\x03Map\x12\x34\n\x0cmap_features\x18\x01 \x03(\x0b\x32\x1e.waymo.open_dataset.MapFeature\x12\x38\n\x0e\x64ynamic_states\x18\x02 \x03(\x0b\x32 .waymo.open_dataset.DynamicState\"j\n\x0c\x44ynamicState\x12\x19\n\x11timestamp_seconds\x18\x01 \x01(\x01\x12?\n\x0blane_states\x18\x02 \x03(\x0b\x32*.waymo.open_dataset.TrafficSignalLaneState\"\x8c\x03\n\x16TrafficSignalLaneState\x12\x0c\n\x04lane\x18\x01 \x01(\x03\x12?\n\x05state\x18\x02 \x01(\x0e\x32\x30.waymo.open_dataset.TrafficSignalLaneState.State\x12\x30\n\nstop_point\x18\x03 \x01(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"\xf0\x01\n\x05State\x12\x16\n\x12LANE_STATE_UNKNOWN\x10\x00\x12\x19\n\x15LANE_STATE_ARROW_STOP\x10\x01\x12\x1c\n\x18LANE_STATE_ARROW_CAUTION\x10\x02\x12\x17\n\x13LANE_STATE_ARROW_GO\x10\x03\x12\x13\n\x0fLANE_STATE_STOP\x10\x04\x12\x16\n\x12LANE_STATE_CAUTION\x10\x05\x12\x11\n\rLANE_STATE_GO\x10\x06\x12\x1c\n\x18LANE_STATE_FLASHING_STOP\x10\x07\x12\x1f\n\x1bLANE_STATE_FLASHING_CAUTION\x10\x08\"\xda\x02\n\nMapFeature\x12\n\n\x02id\x18\x01 \x01(\x03\x12.\n\x04lane\x18\x03 \x01(\x0b\x32\x1e.waymo.open_dataset.LaneCenterH\x00\x12\x31\n\troad_line\x18\x04 \x01(\x0b\x32\x1c.waymo.open_dataset.RoadLineH\x00\x12\x31\n\troad_edge\x18\x05 \x01(\x0b\x32\x1c.waymo.open_dataset.RoadEdgeH\x00\x12\x31\n\tstop_sign\x18\x07 \x01(\x0b\x32\x1c.waymo.open_dataset.StopSignH\x00\x12\x32\n\tcrosswalk\x18\x08 \x01(\x0b\x32\x1d.waymo.open_dataset.CrosswalkH\x00\x12\x33\n\nspeed_bump\x18\t \x01(\x0b\x32\x1d.waymo.open_dataset.SpeedBumpH\x00\x42\x0e\n\x0c\x66\x65\x61ture_data\"+\n\x08MapPoint\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"\xa2\x01\n\x0f\x42oundarySegment\x12\x18\n\x10lane_start_index\x18\x01 \x01(\x05\x12\x16\n\x0elane_end_index\x18\x02 \x01(\x05\x12\x1b\n\x13\x62oundary_feature_id\x18\x03 \x01(\x03\x12@\n\rboundary_type\x18\x04 \x01(\x0e\x32).waymo.open_dataset.RoadLine.RoadLineType\"\xc7\x01\n\x0cLaneNeighbor\x12\x12\n\nfeature_id\x18\x01 \x01(\x03\x12\x18\n\x10self_start_index\x18\x02 \x01(\x05\x12\x16\n\x0eself_end_index\x18\x03 \x01(\x05\x12\x1c\n\x14neighbor_start_index\x18\x04 \x01(\x05\x12\x1a\n\x12neighbor_end_index\x18\x05 \x01(\x05\x12\x37\n\nboundaries\x18\x06 \x03(\x0b\x32#.waymo.open_dataset.BoundarySegment\"\xa5\x04\n\nLaneCenter\x12\x17\n\x0fspeed_limit_mph\x18\x01 \x01(\x01\x12\x35\n\x04type\x18\x02 \x01(\x0e\x32\'.waymo.open_dataset.LaneCenter.LaneType\x12\x15\n\rinterpolating\x18\x03 \x01(\x08\x12.\n\x08polyline\x18\x08 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\x12\x17\n\x0b\x65ntry_lanes\x18\t \x03(\x03\x42\x02\x10\x01\x12\x16\n\nexit_lanes\x18\n \x03(\x03\x42\x02\x10\x01\x12<\n\x0fleft_boundaries\x18\r \x03(\x0b\x32#.waymo.open_dataset.BoundarySegment\x12=\n\x10right_boundaries\x18\x0e \x03(\x0b\x32#.waymo.open_dataset.BoundarySegment\x12\x38\n\x0eleft_neighbors\x18\x0b \x03(\x0b\x32 .waymo.open_dataset.LaneNeighbor\x12\x39\n\x0fright_neighbors\x18\x0c \x03(\x0b\x32 .waymo.open_dataset.LaneNeighbor\"]\n\x08LaneType\x12\x12\n\x0eTYPE_UNDEFINED\x10\x00\x12\x10\n\x0cTYPE_FREEWAY\x10\x01\x12\x17\n\x13TYPE_SURFACE_STREET\x10\x02\x12\x12\n\x0eTYPE_BIKE_LANE\x10\x03\"\xcd\x01\n\x08RoadEdge\x12\x37\n\x04type\x18\x01 \x01(\x0e\x32).waymo.open_dataset.RoadEdge.RoadEdgeType\x12.\n\x08polyline\x18\x02 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"X\n\x0cRoadEdgeType\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x1b\n\x17TYPE_ROAD_EDGE_BOUNDARY\x10\x01\x12\x19\n\x15TYPE_ROAD_EDGE_MEDIAN\x10\x02\"\x88\x03\n\x08RoadLine\x12\x37\n\x04type\x18\x01 \x01(\x0e\x32).waymo.open_dataset.RoadLine.RoadLineType\x12.\n\x08polyline\x18\x02 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"\x92\x02\n\x0cRoadLineType\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x1c\n\x18TYPE_BROKEN_SINGLE_WHITE\x10\x01\x12\x1b\n\x17TYPE_SOLID_SINGLE_WHITE\x10\x02\x12\x1b\n\x17TYPE_SOLID_DOUBLE_WHITE\x10\x03\x12\x1d\n\x19TYPE_BROKEN_SINGLE_YELLOW\x10\x04\x12\x1d\n\x19TYPE_BROKEN_DOUBLE_YELLOW\x10\x05\x12\x1c\n\x18TYPE_SOLID_SINGLE_YELLOW\x10\x06\x12\x1c\n\x18TYPE_SOLID_DOUBLE_YELLOW\x10\x07\x12\x1e\n\x1aTYPE_PASSING_DOUBLE_YELLOW\x10\x08\"H\n\x08StopSign\x12\x0c\n\x04lane\x18\x01 \x03(\x03\x12.\n\x08position\x18\x02 \x01(\x0b\x32\x1c.waymo.open_dataset.MapPoint\":\n\tCrosswalk\x12-\n\x07polygon\x18\x01 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\":\n\tSpeedBump\x12-\n\x07polygon\x18\x01 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint')



_MAP = DESCRIPTOR.message_types_by_name['Map']
_DYNAMICSTATE = DESCRIPTOR.message_types_by_name['DynamicState']
_TRAFFICSIGNALLANESTATE = DESCRIPTOR.message_types_by_name['TrafficSignalLaneState']
_MAPFEATURE = DESCRIPTOR.message_types_by_name['MapFeature']
_MAPPOINT = DESCRIPTOR.message_types_by_name['MapPoint']
_BOUNDARYSEGMENT = DESCRIPTOR.message_types_by_name['BoundarySegment']
_LANENEIGHBOR = DESCRIPTOR.message_types_by_name['LaneNeighbor']
_LANECENTER = DESCRIPTOR.message_types_by_name['LaneCenter']
_ROADEDGE = DESCRIPTOR.message_types_by_name['RoadEdge']
_ROADLINE = DESCRIPTOR.message_types_by_name['RoadLine']
_STOPSIGN = DESCRIPTOR.message_types_by_name['StopSign']
_CROSSWALK = DESCRIPTOR.message_types_by_name['Crosswalk']
_SPEEDBUMP = DESCRIPTOR.message_types_by_name['SpeedBump']
_TRAFFICSIGNALLANESTATE_STATE = _TRAFFICSIGNALLANESTATE.enum_types_by_name['State']
_LANECENTER_LANETYPE = _LANECENTER.enum_types_by_name['LaneType']
_ROADEDGE_ROADEDGETYPE = _ROADEDGE.enum_types_by_name['RoadEdgeType']
_ROADLINE_ROADLINETYPE = _ROADLINE.enum_types_by_name['RoadLineType']
Map = _reflection.GeneratedProtocolMessageType('Map', (_message.Message,), {
  'DESCRIPTOR' : _MAP,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Map)
  })
_sym_db.RegisterMessage(Map)

DynamicState = _reflection.GeneratedProtocolMessageType('DynamicState', (_message.Message,), {
  'DESCRIPTOR' : _DYNAMICSTATE,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.DynamicState)
  })
_sym_db.RegisterMessage(DynamicState)

TrafficSignalLaneState = _reflection.GeneratedProtocolMessageType('TrafficSignalLaneState', (_message.Message,), {
  'DESCRIPTOR' : _TRAFFICSIGNALLANESTATE,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.TrafficSignalLaneState)
  })
_sym_db.RegisterMessage(TrafficSignalLaneState)

MapFeature = _reflection.GeneratedProtocolMessageType('MapFeature', (_message.Message,), {
  'DESCRIPTOR' : _MAPFEATURE,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.MapFeature)
  })
_sym_db.RegisterMessage(MapFeature)

MapPoint = _reflection.GeneratedProtocolMessageType('MapPoint', (_message.Message,), {
  'DESCRIPTOR' : _MAPPOINT,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.MapPoint)
  })
_sym_db.RegisterMessage(MapPoint)

BoundarySegment = _reflection.GeneratedProtocolMessageType('BoundarySegment', (_message.Message,), {
  'DESCRIPTOR' : _BOUNDARYSEGMENT,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.BoundarySegment)
  })
_sym_db.RegisterMessage(BoundarySegment)

LaneNeighbor = _reflection.GeneratedProtocolMessageType('LaneNeighbor', (_message.Message,), {
  'DESCRIPTOR' : _LANENEIGHBOR,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.LaneNeighbor)
  })
_sym_db.RegisterMessage(LaneNeighbor)

LaneCenter = _reflection.GeneratedProtocolMessageType('LaneCenter', (_message.Message,), {
  'DESCRIPTOR' : _LANECENTER,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.LaneCenter)
  })
_sym_db.RegisterMessage(LaneCenter)

RoadEdge = _reflection.GeneratedProtocolMessageType('RoadEdge', (_message.Message,), {
  'DESCRIPTOR' : _ROADEDGE,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.RoadEdge)
  })
_sym_db.RegisterMessage(RoadEdge)

RoadLine = _reflection.GeneratedProtocolMessageType('RoadLine', (_message.Message,), {
  'DESCRIPTOR' : _ROADLINE,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.RoadLine)
  })
_sym_db.RegisterMessage(RoadLine)

StopSign = _reflection.GeneratedProtocolMessageType('StopSign', (_message.Message,), {
  'DESCRIPTOR' : _STOPSIGN,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.StopSign)
  })
_sym_db.RegisterMessage(StopSign)

Crosswalk = _reflection.GeneratedProtocolMessageType('Crosswalk', (_message.Message,), {
  'DESCRIPTOR' : _CROSSWALK,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Crosswalk)
  })
_sym_db.RegisterMessage(Crosswalk)

SpeedBump = _reflection.GeneratedProtocolMessageType('SpeedBump', (_message.Message,), {
  'DESCRIPTOR' : _SPEEDBUMP,
  '__module__' : 'smarts.waymo.waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.SpeedBump)
  })
_sym_db.RegisterMessage(SpeedBump)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _LANECENTER.fields_by_name['entry_lanes']._options = None
  _LANECENTER.fields_by_name['entry_lanes']._serialized_options = b'\020\001'
  _LANECENTER.fields_by_name['exit_lanes']._options = None
  _LANECENTER.fields_by_name['exit_lanes']._serialized_options = b'\020\001'
  _MAP._serialized_start=72
  _MAP._serialized_end=189
  _DYNAMICSTATE._serialized_start=191
  _DYNAMICSTATE._serialized_end=297
  _TRAFFICSIGNALLANESTATE._serialized_start=300
  _TRAFFICSIGNALLANESTATE._serialized_end=696
  _TRAFFICSIGNALLANESTATE_STATE._serialized_start=456
  _TRAFFICSIGNALLANESTATE_STATE._serialized_end=696
  _MAPFEATURE._serialized_start=699
  _MAPFEATURE._serialized_end=1045
  _MAPPOINT._serialized_start=1047
  _MAPPOINT._serialized_end=1090
  _BOUNDARYSEGMENT._serialized_start=1093
  _BOUNDARYSEGMENT._serialized_end=1255
  _LANENEIGHBOR._serialized_start=1258
  _LANENEIGHBOR._serialized_end=1457
  _LANECENTER._serialized_start=1460
  _LANECENTER._serialized_end=2009
  _LANECENTER_LANETYPE._serialized_start=1916
  _LANECENTER_LANETYPE._serialized_end=2009
  _ROADEDGE._serialized_start=2012
  _ROADEDGE._serialized_end=2217
  _ROADEDGE_ROADEDGETYPE._serialized_start=2129
  _ROADEDGE_ROADEDGETYPE._serialized_end=2217
  _ROADLINE._serialized_start=2220
  _ROADLINE._serialized_end=2612
  _ROADLINE_ROADLINETYPE._serialized_start=2338
  _ROADLINE_ROADLINETYPE._serialized_end=2612
  _STOPSIGN._serialized_start=2614
  _STOPSIGN._serialized_end=2686
  _CROSSWALK._serialized_start=2688
  _CROSSWALK._serialized_end=2746
  _SPEEDBUMP._serialized_start=2748
  _SPEEDBUMP._serialized_end=2806
# @@protoc_insertion_point(module_scope)
