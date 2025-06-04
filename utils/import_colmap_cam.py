# ──────────────────────────────────────────────────────────────────────────
#  COLMAP ► Blender  |  One-file loader for PINHOLE / SIMPLE_PINHOLE cams
#  author: you :)
# ──────────────────────────────────────────────────────────────────────────
import os, math, bpy, mathutils
import numpy as np

# ───────────────────────────────── CONFIG ─────────────────────────────────
# SFM_DIR   = "/absolute/path/to/your/colmap/project"       # <-- change me
SFM_DIR   = "/home/innovation/dev-lbringer/video_to_4d/GeometryCrafter/workspace/sfm/dancer"
PATH_ID   = 0                                             # sub-folder id
SENSOR_W  = 36.0                                          # mm; pick real value if known
SCENE_FPS = 24                                            # timeline FPS

# ───────────────────────── COLMAP text parsers (unchanged) ───────────────
def read_intrinsics_text(path):
    cams = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.split()
            cid, model, w, h = int(elems[0]), elems[1], int(elems[2]), int(elems[3])
            cams[cid] = {"model": model,
                         "width": w,
                         "height": h,
                         "params": np.array(list(map(float, elems[4:])))}
    return cams

def read_extrinsics_text(path):
    imgs = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('#') or not line.strip():
                continue
            elems = line.split()
            iid = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            cid  = int(elems[8])
            name = elems[9]
            imgs[iid] = {"qvec": qvec, "tvec": tvec, "cid": cid, "name": name}
            _ = f.readline()      # skip 2nd line with key-points
    return imgs

def qvec2rotmat(q):
    return np.array([
        [1 - 2*q[2]**2 - 2*q[3]**2,   2*q[1]*q[2] - 2*q[0]*q[3],   2*q[1]*q[3] + 2*q[0]*q[2]],
        [2*q[1]*q[2] + 2*q[0]*q[3],   1 - 2*q[1]**2 - 2*q[3]**2,   2*q[2]*q[3] - 2*q[0]*q[1]],
        [2*q[1]*q[3] - 2*q[0]*q[2],   2*q[2]*q[3] + 2*q[0]*q[1],   1 - 2*q[1]**2 - 2*q[2]**2],
    ])

def focal2fov(f, pixels):        # helper
    return 2*math.atan(pixels / (2*f))

# ────────────────────────────── Load COLMAP ──────────────────────────────
intr_file = os.path.join(SFM_DIR, "sparse", str(PATH_ID), "cameras.txt")
extr_file = os.path.join(SFM_DIR, "sparse", str(PATH_ID), "images.txt")
cams  = read_intrinsics_text(intr_file)
imgs  = read_extrinsics_text(extr_file)

# sort by image name for temporal coherence
frames = sorted(imgs.values(), key=lambda d: d["name"])

# ─────────────────────────── Blender scene prep ──────────────────────────
scene = bpy.context.scene
scene.render.fps = SCENE_FPS
scene.frame_start = 1
scene.frame_end   = len(frames)

# create / reuse a camera object
cam_data = bpy.data.cameras.new("ColmapCam") if "ColmapCam" not in bpy.data.cameras \
           else bpy.data.cameras["ColmapCam"]
cam_obj  = bpy.data.objects.new("ColmapCam", cam_data) if "ColmapCam" not in bpy.data.objects \
           else bpy.data.objects["ColmapCam"]
if cam_obj.name not in scene.collection.objects:
    scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_data.sensor_width = SENSOR_W
cam_obj.rotation_mode = 'QUATERNION'

# openCV ➜ Blender axis conversion  (x,y,z)_cv  →  (x,-y,-z)_bl
CV2BLENDER = mathutils.Matrix(((1,0,0,0),
                               (0,-1,0,0),
                               (0,0,-1,0),
                               (0,0,0,1)))
AXIS_FIX = mathutils.Matrix.Rotation(math.radians(-90), 4, 'X')


# ───────────────────────────── Key-frame loop ────────────────────────────
for f_idx, img in enumerate(frames, start=1):
    cam_intr = cams[img["cid"]]
    assert cam_intr["model"] in ("PINHOLE", "SIMPLE_PINHOLE"), \
           f"Unsupported model {cam_intr['model']}"

    # 1. Poses  (world ➜ camera given by COLMAP)
    R_wc = qvec2rotmat(img["qvec"]).T                 # camera ➜ world
    t_wc = -R_wc @ img["tvec"]

    # SCALE = 0.01   # example: metres ➜ centimetres
    # t_wc *= SCALE

    #########################################################
    M_cv2bl = mathutils.Matrix(((1,  0,  0),
                            (0, -1,  0),
                            (0,  0, -1)))
    
    R_wc_bl = mathutils.Matrix(R_wc)             # numpy → mathutils
    R_bl    = R_wc_bl @ M_cv2bl                  # <-- post-multiply, not pre

    # 2. build a 4×4 matrix
    T = mathutils.Matrix((
        (R_bl[0][0], R_bl[0][1], R_bl[0][2], t_wc[0]),
        (R_bl[1][0], R_bl[1][1], R_bl[1][2], t_wc[1]),
        (R_bl[2][0], R_bl[2][1], R_bl[2][2], t_wc[2]),
        (0,          0,          0,          1       )
    ))

    cam_obj.matrix_world = AXIS_FIX @ T
    #########################################################

    # T = mathutils.Matrix((
    #     (R_wc[0][0], R_wc[0][1], R_wc[0][2], t_wc[0]),
    #     (R_wc[1][0], R_wc[1][1], R_wc[1][2], t_wc[1]),
    #     (R_wc[2][0], R_wc[2][1], R_wc[2][2], t_wc[2]),
    #     (0,0,0,1)
    # ))
    # T = CV2BLENDER @ T              # convert axis conventions

    # cam_obj.matrix_world = T

    # 2. Intrinsics  (per-frame if needed)
    if cam_intr["model"] == "PINHOLE":
        fx, fy = cam_intr["params"][:2]
    else:                           # SIMPLE_PINHOLE
        fx = fy = cam_intr["params"][0]

    cam_data.lens_unit = 'MILLIMETERS'
    cam_data.angle_x = focal2fov(fx, cam_intr["width"])
    cam_data.angle_y = focal2fov(fy, cam_intr["height"])

    # 3. Insert key-frames
    cam_obj.keyframe_insert(data_path="location",               frame=f_idx)
    cam_obj.keyframe_insert(data_path="rotation_quaternion",    frame=f_idx)
    # cam_data.keyframe_insert(data_path="angle_x",               frame=f_idx)
    # cam_data.keyframe_insert(data_path="angle_y",               frame=f_idx)
    cam_data.lens_unit = 'FOV'
    fov = focal2fov(fx, cam_intr["width"])      # horizontal FOV
    # cam_data.angle = fov
    # cam_data.keyframe_insert(data_path="angle", frame=f_idx)

    lens_mm = SENSOR_W * fx / cam_intr["width"]
    cam_data.lens = lens_mm
    cam_data.keyframe_insert(data_path="lens", frame=f_idx)

print(f"Imported {len(frames)} frames ➜ Timeline 1-{len(frames)}.")
