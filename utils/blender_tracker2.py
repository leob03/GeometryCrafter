import bpy, bmesh, mathutils, os, re
import numpy as np

folder   = "/home/innovation/dev-lbringer/video_to_4d/GeometryCrafter/workspace/examples_output"
pattern = re.compile(r"mesh_(\d{4})\.glb")       # keep same as before

# --------------------------------------------------------------------------
def pca_orientation(verts_np: np.ndarray) -> mathutils.Quaternion:
    """Return the quaternion whose axes are the principal components."""
    # Subtract mean
    P = verts_np - verts_np.mean(0, keepdims=True)
    # 3×3 covariance
    cov = (P.T @ P) / len(P)
    # eigenvectors sorted by eigenvalue
    eigval, eigvec = np.linalg.eigh(cov)
    # need right-handed basis: largest → X, middle → Z, cross → Y
    idx = np.argsort(eigval)[::-1]
    x, z = eigvec[:, idx[0]], eigvec[:, idx[1]]
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return mathutils.Matrix(R).to_quaternion()

# --------------------------------------------------------------------------
# collect file list
files = sorted(
    (f for f in os.listdir(folder) if pattern.match(f)),
    key=lambda f: int(pattern.match(f).group(1))
)
n_frames = len(files)
print(f"Analysing {n_frames} meshes")

# create the driver Empty
track = bpy.data.objects.new("TrackedMotion", None)
bpy.context.collection.objects.link(track)

for f, file in enumerate(files):
    path = os.path.join(folder, file)
    bpy.ops.import_scene.gltf(filepath=path)
    # keep only the MESH children of the newly-imported hierarchy
    mesh_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not mesh_objs:
        print(f"⚠  No mesh found in {file}, skipped.")
        continue

    obj = mesh_objs[0] 

    # -- grab vertices in world space --------------------------------------
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj  = obj.evaluated_get(deps)
    mesh_tmp  = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=deps)

    verts = np.array([eval_obj.matrix_world @ v.co for v in mesh_tmp.vertices])

    centre = verts.mean(0)
    orient = pca_orientation(verts)

    # ---------------------------------------------------------------------
    # write keyframes on the Empty
    track.location = centre
    track.rotation_mode = 'QUATERNION'
    track.rotation_quaternion = orient
    track.keyframe_insert("location", frame=f)
    track.keyframe_insert("rotation_quaternion", frame=f)

    # clean up -- free memory and hide the GLB again
    eval_obj.to_mesh_clear()           # frees mesh_tmp; nothing else needed
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    # purge orphans so VRAM doesn’t spiral
    bpy.ops.outliner.orphans_purge(do_recursive=True)

print("✓  Motion baked to object  ➜  TrackedMotion")

# OPTIONAL: set timeline
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end   = n_frames-1
