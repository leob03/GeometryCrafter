import bpy, numpy as np
from mathutils import Vector

# ---------------------------------------------------------------------
# 0. parameters – adjust to your naming scheme
PREFIX      = "geometry_0"          # each frame-mesh starts with this
TRACKER_NAME= "Tracker"        # Empty that will receive keyframes
SCENE       = bpy.context.scene

# ---------------------------------------------------------------------
# 1. ensure tracker Empty exists
if TRACKER_NAME in bpy.data.objects:
    tracker = bpy.data.objects[TRACKER_NAME]
else:
    tracker = bpy.data.objects.new(TRACKER_NAME, None)
    SCENE.collection.objects.link(tracker)
tracker.empty_display_size = 0.05
tracker.empty_display_type = 'ARROWS'

# ---------------------------------------------------------------------
# 2. build a dict  frame_index → mesh_object
frame_mesh = {}
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(PREFIX):
        # extract the 4-digit index:  mesh_0038.001 → 38
        idx = int(obj.name.split('_')[1][:4])
        frame_mesh[idx] = obj

if not frame_mesh:
    raise ValueError(f"No objects whose name starts with “{PREFIX}” found!")

# make sure we key only within the timeline range
frame_start = min(frame_mesh)
frame_end   = max(frame_mesh)
SCENE.frame_start = frame_start
SCENE.frame_end   = frame_end

# ---------------------------------------------------------------------
# 3. main loop – centroid per frame
for f in range(frame_start, frame_end + 1):
    SCENE.frame_set(f)

    mesh_obj = frame_mesh.get(f)
    if not mesh_obj or mesh_obj.hide_viewport:
        # no visible geometry this frame – skip keyframes
        continue

    # vertices → Nx3 numpy array in world space ----------------------
    verts_world = np.array([mesh_obj.matrix_world @ v.co
                            for v in mesh_obj.data.vertices],
                           dtype=np.float32)
    centroid = Vector(verts_world.mean(axis=0))

    # keyframe tracker location --------------------------------------
    tracker.location = centroid
    tracker.keyframe_insert(data_path="location", frame=f)

print("✓  Tracker Empty animated with centroid path.")
