import bpy, os, re

folder   = "/home/innovation/dev-lbringer/video_to_4d/GeometryCrafter/workspace/examples_output"
pattern  = re.compile(r"mesh_(\d{4})\.glb")

files = sorted(
    (f for f in os.listdir(folder) if pattern.match(f)),
    key=lambda x: int(pattern.match(x).group(1))
)
n_frames = len(files)
print(f"Importing {n_frames} meshes …")

frame_objects = []

for file in files:
    frame_idx = int(pattern.match(file).group(1))
    path = os.path.join(folder, file)

    bpy.ops.import_scene.gltf(filepath=path)
    imported = list(bpy.context.selected_objects)
    frame_objects.append(imported)

    # make them hidden by default & keyframe **at frame 0**
    for obj in imported:
        obj.hide_viewport = True
        obj.hide_render   = True
        obj.keyframe_insert(data_path="hide_viewport", frame=0)
        obj.keyframe_insert(data_path="hide_render",   frame=0)

# ------------------------------------------------------------
# toggle each set ON at its own frame, OFF at the next
for f, objs in enumerate(frame_objects):
    for obj in objs:
        obj.hide_viewport = False
        obj.hide_render   = False
        obj.keyframe_insert(data_path="hide_viewport", frame=f)
        obj.keyframe_insert(data_path="hide_render",   frame=f)

        obj.hide_viewport = True
        obj.hide_render   = True
        obj.keyframe_insert(data_path="hide_viewport", frame=f+1)
        obj.keyframe_insert(data_path="hide_render",   frame=f+1)

# timeline range
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end   = n_frames-1

print("✓  One-mesh-per-frame visibility animation set.")
