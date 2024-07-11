"""Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.

(MPG) is holder of all proprietary rights on this computer program. You can
only use this computer program if you have closed a license agreement with MPG
or you get the right to use the computer program from someone who is authorized
to grant you that right. Any use of the computer program without a valid
license is prohibited and liable to prosecution. Copyright 2019 Max-Planck-
Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of
its Max Planck Institute for Intelligent Systems and the Max Planck Institute
for Biological Cybernetics. All rights reserved. More information about VOCA is
available at http://voca.is.tue.mpg.de. For comments or questions, please email
us at voca@tue.mpg.de
"""

import cv2
import numpy as np
import os
import subprocess
import torch
from tqdm import tqdm
from dataset.dataset_config import dataset_config

def render_meshes(mesh_vertices: torch.Tensor,
                  faces: torch.Tensor,
                  img_size: tuple = (256, 256),
                  aa_factor: int = 1,
                  vis_bs: int = 100):
    assert len(mesh_vertices) == len(faces) or len(faces) == 1
    if len(faces) == 1:
        faces = faces.repeat(len(mesh_vertices), 1, 1)

    x_max = y_max = mesh_vertices[..., 0:2].abs().max()
    from pytorch3d.renderer import (
        DirectionalLights,
        FoVOrthographicCameras,
        HardPhongShader,
        Materials,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.renderer.blending import BlendParams
    from pytorch3d.renderer.materials import Materials
    from pytorch3d.structures import Meshes
    aa_factor = int(aa_factor)
    device = mesh_vertices.device
    verts_batch_lst = torch.split(mesh_vertices, vis_bs)
    faces_batch_lst = torch.split(faces, vis_bs)
    R = torch.eye(3).to(mesh_vertices.device)
    R[0, 0] = -1
    R[2, 2] = -1
    T = torch.zeros(3).to(mesh_vertices.device)
    T[2] = 10
    R, T = R[None], T[None]
    W, H = img_size
    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=0.01,
        zfar=3,
        max_x=x_max * 1.2,
        min_x=-x_max * 1.2,
        max_y=y_max * 1.2,
        min_y=-y_max * 1.2,
    )
    # cameras = FoVPerspectiveCameras(
    # 	device=device,
    # 	R=R,
    # 	T=T,
    # 	znear=0.01,
    # 	zfar=3,
    # 	aspect_ratio=1,
    # 	fov=0.3,
    # 	degrees=False
    # 	)

    raster_settings = RasterizationSettings(
        image_size=(W * aa_factor, H * aa_factor),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    raster = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )
    blend_params = BlendParams(
        sigma=0.0, gamma=0.0, background_color=(0.0, 0.0, 0.0))
    lights = DirectionalLights(
        device=device,
        direction=((0, 0, 1), ),
        ambient_color=((0.3, 0.3, 0.3), ),
        diffuse_color=((0.6, 0.6, 0.6), ),
        specular_color=((0.1, 0.1, 0.1), ))
    materias = Materials(
        ambient_color=((1, 1, 1), ),
        diffuse_color=((1, 1, 1), ),
        specular_color=((1, 1, 1), ),
        shininess=15,
        device=device)
    shader = HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materias,
        blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=raster,
        shader=shader,
    )
    rendered_imgs = []
    for verts_batch, faces_batch in tqdm(
            zip(verts_batch_lst, faces_batch_lst),
            total=(len(verts_batch_lst))):
        textures = TexturesVertex(verts_features=torch.ones_like(verts_batch))
        meshes = Meshes(
            verts=verts_batch, faces=faces_batch, textures=textures)
        with torch.no_grad():
            imgs = renderer(meshes)
        rendered_imgs.append(imgs.cpu())
    rendered_imgs = torch.cat(rendered_imgs, dim=0)
    if aa_factor > 1:
        rendered_imgs = rendered_imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        rendered_imgs = torch.nn.functional.interpolate(
            rendered_imgs, scale_factor=1 / aa_factor, mode='bicubic')
        rendered_imgs = rendered_imgs.permute(0, 2, 3, 1)  # NCHW -> NHWC
    return rendered_imgs


def read_obj(in_path):
    with open(in_path, 'r') as obj_file:
        # Read the lines of the OBJ file
        lines = obj_file.readlines()

    # Initialize empty lists for vertices and faces
    verts = []
    faces = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        elements = line.split()  # Split the line into elements

        if len(elements) == 0:
            continue  # Skip empty lines

        # Check the type of line (vertex or face)
        if elements[0] == 'v':
            # Vertex line
            x, y, z = map(float,
                          elements[1:4])  # Extract the vertex coordinates
            verts.append((x, y, z))  # Add the vertex to the list
        elif elements[0] == 'f':
            # Face line
            face_indices = [
                int(index.split('/')[0]) for index in elements[1:]
            ]  # Extract the vertex indices
            faces.append(face_indices)  # Add the face to the list
    return np.array(verts), np.array(faces)


def pad_for_libx264(image_array):
    """Pad zeros if width or height of image_array is not divisible by 2.
    Otherwise you will get.

    \"[libx264 @ 0x1b1d560] width not divisible by 2 \"

    Args:
            image_array (np.ndarray):
                    Image or images load by cv2.imread().
                    Possible shapes:
                    1. [height, width]
                    2. [height, width, channels]
                    3. [images, height, width]
                    4. [images, height, width, channels]

    Returns:
            np.ndarray:
                    A image with both edges divisible by 2.
    """
    if image_array.ndim == 2 or \
      (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or \
      (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array
    hei_pad = image_array.shape[hei_index] % 2
    wid_pad = image_array.shape[wid_index] % 2
    if hei_pad + wid_pad > 0:
        pad_width = []
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = \
         np.pad(image_array,
             pad_width,
             mode='constant', constant_values=values)
    return image_array


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps=30,
    resolution=None,
    disable_log: bool = False,
) -> None:
    """Convert an array to a video directly, gif not supported.

    Args:
            image_array (np.ndarray): shape should be (f * h * w * 3).
            output_path (str): output video file path.
            fps (Union[int, float, optional): fps. Defaults to 30.
            resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                    optional): (height, width) of the output video.
                    Defaults to None.
            disable_log (bool, optional): whether close the ffmepg command info.
                    Defaults to False.
    Raises:
            FileNotFoundError: check output path.
            TypeError: check input array.

    Returns:
            None.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
    else:
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]
    command = [
        '/usr/bin/ffmpeg',
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',
        '-s',
        f'{int(width)}x{int(height)}',  # size of one frame
        '-pix_fmt',
        'bgr24',
        '-r',
        f'{fps}',  # frames per second
        '-loglevel',
        'error',
        '-threads',
        '4',
        '-i',
        '-',  # The input comes from a pipe
        '-vcodec',
        'libx264',
        '-an',  # Tells FFMPEG not to expect any audio
        output_path,
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError('No buffer received.')
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def get_obj_faces(annot_type: str):
    template_obj_path_dict = {
        '3DETF_blendshape_weight': 'resources/obj_template/3DETF_blendshape_weight.obj',
        'FLAME_5023_vertices': 'resources/obj_template/FLAME_5023_vertices.obj',
        'BIWI_23370_vertices': 'resources/obj_template/BIWI_23370_vertices.obj',
        'flame_params_from_dadhead': 'resources/obj_template/flame_params_from_dadhead.obj',
        'inhouse_blendshape_weight': 'resources/obj_template/inhouse_blendshape_weight.obj',
        'meshtalk_6172_vertices': 'resources/obj_template/meshtalk_6172_vertices.obj'
    }

    faces = np.array(read_obj(template_obj_path_dict[annot_type])[1])
    if faces.min() == 1:
        faces = faces - 1
    return faces


def vis_model_out_proxy():
    import sys
    npz_path = sys.argv[1]
    audio_dir = sys.argv[2]
    out_dir = sys.argv[3]
    save_images = False 
    os.makedirs(out_dir, exist_ok=True)
    out_dict = np.load(npz_path, allow_pickle=True)
    print(list(out_dict.keys()))
    for wav_f in out_dict.keys():
        wav_path = os.path.join(audio_dir, wav_f)
        cur_out_dict = out_dict[wav_f]
        for datasetname, out in cur_out_dict.item().items():
            out = torch.Tensor(out)
            annot_type = dataset_config[datasetname]['annot_type']
            faces = get_obj_faces(annot_type)
            with torch.no_grad():
                if save_images:
                    out_size = (1024, 1024)
                else:
                    out_size = (512, 512)
                img_array = render_meshes(out.cuda(),
                                          torch.from_numpy(faces[None]).cuda(),
                                          out_size)
                if save_images:
                    channels = 4
                else:
                    channels = 3
                img_array = (img_array[..., :channels].cpu().numpy() *
                             255).astype(np.uint8)
            out_key = f'{wav_f[:-4]}'
            if save_images:
                out_images_dir = os.path.join(out_dir, datasetname, out_key)
                os.makedirs(out_images_dir, exist_ok=True)
                for imgidx, img in enumerate(img_array):
                    img_path = os.path.join(out_images_dir,
                                            f'{imgidx:04d}.png')
                    cv2.imwrite(img_path, img)
                continue
            out_video_path = os.path.join(out_dir,
                                          f'{out_key}_{datasetname}.mp4')
            out_video_dir = os.path.dirname(out_video_path)
            os.makedirs(out_video_dir, exist_ok=True)
            tmp_path = os.path.join(out_video_dir, 'tmp.mp4')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if annot_type == 'BIWI_23370_vertices':
                fps = 25
            else:
                fps = 30
            print(img_array.shape, img_array.dtype)
            array_to_video(img_array, output_path=tmp_path, fps=fps)

            cmd = f'ffmpeg -i {tmp_path} -i  {wav_path} -c:v copy -c:a aac -shortest -strict -2  {out_video_path} -y '
            print(cmd)
            subprocess.run(cmd, shell=True)
            os.remove(tmp_path)


if __name__ == '__main__':
    vis_model_out_proxy()
