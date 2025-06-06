from pathlib import Path
import torch
from typing import Union
from decord import VideoReader, cpu
from diffusers.training_utils import set_seed
from fire import Fire
import os
import numpy as np
import torch.nn.functional as F
import utils3d
import cv2
import matplotlib
import trimesh, tempfile

from third_party import MoGe
from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)

from scipy.ndimage import binary_dilation

def canny_depth_edges(depth_np: np.ndarray,
                      low=0.4, high=0.5) -> np.ndarray:
    """depth_np: H×W float32, NaNs OK."""
    d = np.nan_to_num(depth_np, nan=0.0).astype(np.float32)
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(d_norm.astype(np.uint8),
                      int(low*255), int(high*255))
    return edges.astype(bool)

def dilate_mask(mask_np: np.ndarray, radius: int = 3) -> np.ndarray:
    if radius <= 0:
        return mask_np
    kernel = np.ones((radius, radius), bool)
    return binary_dilation(mask_np, kernel)

def resize_image_like_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    h_pts, w_pts = points.shape[:2]
    if (h_img, w_img) != (h_pts, w_pts):
        from PIL import Image
        return np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((w_pts, h_pts), resample=Image.BILINEAR)) / 255.0
    return image


def main(
    video_path: str,
    save_folder: str = "workspace/output/",
    cache_dir: str = "workspace/cache", 
    height: int = None,
    width: int = None,
    downsample_ratio: float = 1.0,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    decode_chunk_size: int = 8,
    overlap: int = 25,
    process_length: int = -1,
    process_stride: int = 1,
    seed: int = 42,
    model_type: str = 'diff', # 'determ'
    force_projection: bool = True,
    force_fixed_focal: bool = True,
    use_extract_interp: bool = False,
    track_time: bool = False,
    low_memory_usage: bool = False
):
    assert model_type in ['diff', 'determ']
    set_seed(seed)
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float16)
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float32)
    prior_model = MoGe(
        cache_dir=cache_dir,
    ).requires_grad_(False).to('cuda', dtype=torch.float32)
    if model_type == 'diff':
        pipe = GeometryCrafterDiffPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    else:
        pipe = GeometryCrafterDetermPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")

    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(e)
        print("Xformers is not enabled")
    # bugs at https://github.com/continue-revolution/sd-webui-animatediff/issues/101
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    
    video_base_name = os.path.basename(video_path).split('.')[0]
    vid = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    if height is None or width is None:
        height = original_height
        width = original_width
    
    assert height % 64 == 0
    assert width % 64 == 0

    frames_idx = list(range(0, len(vid), process_stride))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
    if process_length > 0:
        process_length = min(process_length, len(frames))
        frames = frames[:process_length]
    else:
        process_length = len(frames)
    window_size = min(window_size, process_length)
    if window_size == process_length: 
        overlap = 0
    frames_tensor = torch.tensor(frames.astype("float32"), device='cuda').float().permute(0, 3, 1, 2)
    # t,3,h,w

    # Extract the first frame and assign it to the 'image' variable
    image = frames[0]  # First frame in the original format

    if downsample_ratio > 1.0:
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        frames_tensor = F.interpolate(frames_tensor, (round(frames_tensor.shape[-2]/downsample_ratio), round(frames_tensor.shape[-1]/downsample_ratio)), mode='bicubic', antialias=True).clamp(0, 1)

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        rec_point_map, rec_valid_mask = pipe(
            frames_tensor,
            point_map_vae,
            prior_model,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            decode_chunk_size=decode_chunk_size,
            overlap=overlap,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            track_time=track_time,
            low_memory_usage=low_memory_usage
        )

        if downsample_ratio > 1.0:
            rec_point_map = F.interpolate(rec_point_map.permute(0,3,1,2), (original_height, original_width), mode='bilinear').permute(0, 2, 3, 1)
            rec_valid_mask = F.interpolate(rec_valid_mask.float().unsqueeze(1), (original_height, original_width), mode='bilinear').squeeze(1) > 0.5

        ## saving the pointcloud in a mesh format (similar to MoGe)
        n_frames = rec_point_map.shape[0]
        all_meshes = []                     # collect for the final OBJ

        for t in range(n_frames):
            points = rec_point_map[t].cpu().numpy().astype(np.float32)
            mask_t = rec_valid_mask[t].cpu().numpy().astype(np.bool_)
            image_t = frames[t]
            depth_t = points[:, :, 2]

            h_pts, w_pts = points.shape[:2]
            image_t_resized = resize_image_like_points(image_t, points)
            assert image_t_resized.shape[:2] == (h_pts, w_pts), \
                f"RGB {image_t_resized.shape[:2]} vs points {points.shape[:2]}"
            
            # edge strip
            edge = canny_depth_edges(depth_t, 0.4, 0.5)
            edge = dilate_mask(edge, radius=3)
            mesh_mask = mask_t & ~edge

            print(f"[frame {t:03d}] points {points.shape[:2]} | "
                f"RGB {image_t_resized.shape[:2]} | mask {mesh_mask.shape[:2]}")

            h_pts, w_pts = points.shape[:2]

            faces, verts, vcols, uvs = utils3d.numpy.image_mesh(
                points,
                image_t_resized.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=w_pts, height=h_pts),
                mask=mesh_mask,
                tri=True
            )
            verts, uvs = verts * [1, -1, -1], uvs * [1, -1] + [0, 1]

            # save per-frame GLB
            glb_path = save_path / f"mesh_{t:04d}.glb"
            save_glb(glb_path, verts, faces, uvs, image_t)

            all_meshes.append(
                trimesh.Trimesh(verts, faces, visual=trimesh.visual.texture.TextureVisuals(
                    uv=uvs, image=image_t.astype(np.uint8)), process=False)
            )

        np.savez(
            str(save_path / f"{video_base_name}.npz"), 
            point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
            mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_))

        for i, m in enumerate(all_meshes):
            node = trimesh.SceneNode(name=f"f{i:04d}")
            node.transform = trimesh.transformations.translation_matrix([0, 0, 0])
            scene.add_geometry(m, node_name=node.name)
        
        scene.export("animated_sequence.glb")

        

def save_glb(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
):
    import numpy as np, trimesh
    from PIL import Image

    if texture.dtype != np.uint8:
        texture_u8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
    else:
        texture_u8 = texture

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture_u8, mode="RGB"),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(save_path)


def save_ply(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_colors: np.ndarray,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(save_path)

    
if __name__ == "__main__":
    Fire(main)
