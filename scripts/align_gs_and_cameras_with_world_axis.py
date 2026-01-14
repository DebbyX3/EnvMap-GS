import sys
import numpy as np
import struct
from plyfile import PlyData, PlyElement
import pycolmap
import os
import shutil
import collections
from scipy.spatial.transform import Rotation as R
import argparse
from pathlib import Path
import json
import matplotlib
from copy import deepcopy
import datetime
matplotlib.use('TkAgg')

#### FROM COLMAP CODEBASE ####

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

class Image(BaseImage):
    def __new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self = super(Image, cls).__new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids)
        self._name = name
        return self

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras

def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

#### END FROM COLMAP CODEBASE ####

class ImageForColmap:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

class DualWriter:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def parse_cameras(cameras_bin_path):
    # ************** READ COLMAP CAMERA FILE    

    # ***  WARNING: THIS SCRIPT ASSUMES THAT ALL CAMERAS HAVE THE SAME INTRINSICS ***
    # ***  SO IN THE CAMERA FILE WE WILL ONLY READ THE FIRST CAMERA INTRINSICS ***
    # *** (ALSO BEACUSE THERE IS ONLY ONE CAMERA IN THE CAMERA FILE IF THEY SHARE THE SAME INTRINSICS) ***

    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    #
    # In case of Pinhole camera model (example):
    # 1 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
    # 
    # In case of Simple Pinhole camera model (example):
    # 2 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
    #
    # In case of Simple Radial camera model (example):
    # 3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531

    cameras_bin_info = read_cameras_binary(cameras_bin_path) 

    cameras_info = {}

    for cam_id, cam in cameras_bin_info.items():
        # Camera info contains:
        # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
        # 0          1       2       3       4   5   6   7   8
        # Where PARAMS[] are:
        # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
        # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
        # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
        # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

        if cam.model == "PINHOLE":
            cameras_info[cam_id] = {'id': cam_id, 
                                    'type': cam.model, 
                                    'width': cam.width, 
                                    'height': cam.height, 
                                    'fx': cam.params[0], 
                                    'fy': cam.params[1], 
                                    'cx': cam.params[2], 
                                    'cy': cam.params[3]}
        # fx, fy, cx, cy, k1, k2, p1, p2
        # Based on the pinhole camera model. Additionally models radial and
        # tangential distortion (up to 2nd degree of coefficients).
        elif cam.model == "OPENCV":
            cameras_info[cam_id] = {'id': cam_id, 
                                    'type': cam.model, 
                                    'width': cam.width, 
                                    'height': cam.height, 
                                    'fx': cam.params[0], 
                                    'fy': cam.params[1], 
                                    'cx': cam.params[2], 
                                    'cy': cam.params[3],
                                    'k1': cam.params[4], 
                                    'k2': cam.params[5],
                                    'p1': cam.params[6], 
                                    'p2': cam.params[7]}

        print("--- Camera ID: " + str(cameras_info[cam_id]['id']) + " - " + cameras_info[cam_id]['type'])
        print(" Width: ", cameras_info[cam_id]['width'])
        print(" Height: ", cameras_info[cam_id]['height'])
        print(" fx: ", cameras_info[cam_id]['fx'])
        print(" fy: ", cameras_info[cam_id]['fy'])
        print(" cx: ", cameras_info[cam_id]['cx'])
        print(" cy: ", cameras_info[cam_id]['cy'])  

        '''if 'k1' in locals():
            print(" k1: ", k1)
        if 'k2' in locals():
            print(" k2: ", k2)'''

        break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

    # Create the camera intrinsic matrix with the first (and only one) camera's intrinsics
    intrinsic_matrix = np.array([[cameras_info[1]['fx'],    0,                      cameras_info[1]['cx']],
                                [0,                         cameras_info[1]['fy'],  cameras_info[1]['cy']],
                                [0,                         0,                      1]])

    return cameras_info, intrinsic_matrix

def parse_images(images_bin_path):
    """Parsa il file images.bin per estrarre le pose delle immagini"""
    # Images info contains two rows per image:

    #   1st row has:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   0         1   2   3   4   5   6   7   8          9

    #   2nd row has:
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    #   Example of this row:
    #   2362.39 248.498 58396       1784.7 268.254 59027        1784.7 268.254 -1
    #   X       Y       POINT3D_ID  X      Y       POINT3D_ID   X      Y       POINT3D_ID
    #   the last keypoint does not observe a 3D point in the reconstruction as the 3D point identifier is -1

    images_bin_info = read_images_binary(images_bin_path) 

    images_info = {}

    for img_id, img in images_bin_info.items():
        # COLMAP: ROTATION MATRIX 'R' FROM QUATERNION - FROM WORLD TO CAMERA
        qw, qx, qy, qz = img.qvec # please note that COLMAP writes/uses the quaternion in the order [qw, qx, qy, qz]
        rot_from_w_to_c = R.from_quat([qx, qy, qz, qw]).as_matrix() # but scipy uses the order [qx, qy, qz, qw] for quaternions

        # COLMAP: TRANSLATION VECTOR 'T' - FROM WORLD TO CAMERA
        tx, ty, tz = img.tvec
        trans_from_w_to_c = np.array([tx, ty, tz], dtype = float).reshape(3, 1)

        images_info[img_id] = {'id': img_id, 
                                'filename': img.name,
                                'rot_from_w_to_c': rot_from_w_to_c,
                                'trans_from_w_to_c': trans_from_w_to_c,
                                'camera_id': img.camera_id,
                                'points_2d': img.xys,
                                'point3D_ids': img.point3D_ids
                                }
    
        print("--- Image ID: " + str(images_info[img_id]['id']) + " - " + images_info[img_id]['filename'])
        
    return images_info

def write_images_binary_from_images_info(images_info, output_path):
    """
    Adatta images_info (output di parse_images)
    al formato richiesto da write_images_binary
    """

    images_for_colmap = {}

    for img_id, img in images_info.items():

        # rotazione world → camera
        R_w2c = img['rot_from_w_to_c']

        # matrice → quaternione (scipy: [x,y,z,w])
        q = R.from_matrix(R_w2c).as_quat()

        # torna a formato COLMAP [qw,qx,qy,qz]
        qvec = np.array([
            q[3],  # w
            q[0],  # x
            q[1],  # y
            q[2],  # z
        ], dtype=np.float64)

        # traslazione world → camera
        tvec = img['trans_from_w_to_c'].reshape(3).astype(np.float64)

        images_for_colmap[img_id] = ImageForColmap(
            id=img['id'],
            qvec=qvec,
            tvec=tvec,
            camera_id=img['camera_id'],
            name=img['filename'],
            xys=img['points_2d'],
            point3D_ids=img['point3D_ids'],
        )

    # scrittura binaria vera e propria
    write_images_binary(images_for_colmap, output_path)

def create_cameras_json(cameras_info, images_info, save_json_path):
    cams_json = []

    # assumiamo intrinsics condivise (come in GS vanilla)
    cam0 = next(iter(cameras_info.values()))
    fx = cam0["fx"]
    fy = cam0["fy"]
    width = cam0["width"]
    height = cam0["height"]

    for idx, (img_id, img) in enumerate(images_info.items()):
        R_wc = img['rot_from_w_to_c']           # (3,3)
        t_wc = img['trans_from_w_to_c'].reshape(3)

        # camera → world
        R_cw = R_wc.T
        C = -R_cw @ t_wc

        cams_json.append({
            "id": idx,
            "img_name": img["filename"] if isinstance(img, dict) else img.filename,
            "width": width,
            "height": height,
            "position": C.tolist(),
            "rotation": R_cw.tolist(),
            "fx": fx,
            "fy": fy
        })

    print("Saving aligned cameras.json to:\n" + save_json_path)
    with open(save_json_path, "w") as f:
        json.dump(cams_json, f, indent=2)

def compute_up_vector(images_info):
    up_vectors = []

    # ------------- compute up-vector for each camera
    for img_id, img in images_info.items():
        R_wc = img['rot_from_w_to_c']  # world -> camera

        # up camera in camera space (COLMAP: Y down)
        up_cam = np.array([0, -1, 0])

        # up in world space
        up_world = R_wc.T @ up_cam

        up_world = up_world / np.linalg.norm(up_world)
        up_vectors.append(up_world)

    up_vectors = np.stack(up_vectors, axis=0)  # (N,3)
    return up_vectors

def find_rot_align_pca(up_vectors):
    # ------------- PCA: primo asse principale
    cov = up_vectors.T @ up_vectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    up_global = eigvecs[:, np.argmax(eigvals)]
    up_global = up_global / np.linalg.norm(up_global)

    # forza verso +Y (evita flip)
    if up_global[1] < 0:
        up_global *= -1

    print("Estimated global up before aligning:", up_global)

    target_up = np.array([0, 1, 0])

    rot_align, rmsd = R.align_vectors([target_up], [up_global])
    R_align = rot_align.as_matrix()
    return R_align
    
def rotate_cameras(images_info, R_align):
    for img_id, img in images_info.items():
        R_wc = img['rot_from_w_to_c']
        t_wc = img['trans_from_w_to_c'].reshape(3)

        # camera center
        C = - R_wc.T @ t_wc

        # rotate world
        C_new = R_align @ C
        R_wc_new = R_wc @ R_align.T
        t_wc_new = - R_wc_new @ C_new

        img['rot_from_w_to_c_aligned'] = R_wc_new
        img['trans_from_w_to_c_aligned'] = t_wc_new.reshape(3, 1)

def rotate_gaussians(ply_path_to_align, R_align):
    ply = PlyData.read(ply_path_to_align)
    vertex = ply.elements[0]
    data = vertex.data

    # extract params
    # posizioni
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)

    # normali
    nxyz = np.stack([data['nx'], data['ny'], data['nz']], axis=1)

    # quaternion GS vanilla (w,x,y,z)
    quat_gs = np.stack([
        data['rot_0'],
        data['rot_1'],
        data['rot_2'],
        data['rot_3'],
    ], axis=1)

    # align rotations in quaternion
    q_align = R.from_matrix(R_align)

    # rotate positions and normals
    xyz_new = (R_align @ xyz.T).T
    nxyz_new = (R_align @ nxyz.T).T

    '''
    warning!
    In Gaussian Splatting vanilla:

    rot_0 = qw
    rot_1 = qx
    rot_2 = qy
    rot_3 = qz

    In scipy:

    [x, y, z, w]
    '''

    # rotate gaussians (quaternions): q_new = q_align ⊗ q_old
    # GS -> scipy format
    quat_scipy = np.stack([
        quat_gs[:, 1],  # x
        quat_gs[:, 2],  # y
        quat_gs[:, 3],  # z
        quat_gs[:, 0],  # w
    ], axis=1)

    rot_old = R.from_quat(quat_scipy)
    rot_new = q_align * rot_old   # left-multiply = world rotation

    quat_new_scipy = rot_new.as_quat()

    # back to GS format (w,x,y,z)
    quat_new_gs = np.stack([
        quat_new_scipy[:, 3],
        quat_new_scipy[:, 0],
        quat_new_scipy[:, 1],
        quat_new_scipy[:, 2],
    ], axis=1)

    return data, xyz_new, nxyz_new, quat_new_gs

def save_gs_ply(data, xyz_new, nxyz_new, quat_new_gs, save_ply_path):
    new_data = np.empty(data.shape, dtype=data.dtype)

    # copia tutto
    for name in data.dtype.names:
        new_data[name] = data[name]

    # sovrascrivi i campi raddrizzati
    new_data['x'] = xyz_new[:, 0]
    new_data['y'] = xyz_new[:, 1]
    new_data['z'] = xyz_new[:, 2]

    new_data['nx'] = nxyz_new[:, 0]
    new_data['ny'] = nxyz_new[:, 1]
    new_data['nz'] = nxyz_new[:, 2]

    new_data['rot_0'] = quat_new_gs[:, 0]
    new_data['rot_1'] = quat_new_gs[:, 1]
    new_data['rot_2'] = quat_new_gs[:, 2]
    new_data['rot_3'] = quat_new_gs[:, 3]

    # save
    vertex_out = PlyElement.describe(new_data, 'vertex')
    ply_out = PlyData([vertex_out], text=False)
    print("Saving aligned ply to:\n" + save_ply_path)
    ply_out.write(save_ply_path)

def realign_colmap_cameras(R_align, images_info):
    # the camera in colmap is defined as an equation relative to the gs world
    # so, if we rotate the gs world, we need to rotate the camera in the opposite direction
    # infact: p_cam = R_wc * P_world
    # goal: the point in the camera space after the transformation must remain the same to the previous one before the transformation
    # P_cam_new = P_cam_old

    # before:
    # P_cam_old = R_wc_old * P_world_old
    # after:
    # P_world' = R_align · P_world
    # if we want the same img, make the two equation equals:
    # P_cam_new = R_wc' · P_world' = R_wc · P_world
    # substituting:
    # R_wc' · (R_align · P_world) = R_wc · P_world
    # having:
    # R_wc' · R_align = R_wc  
    # that equals to 
    # R_wc' = R_wc · R_align⁻¹
    # (on paper makes lots more sense)

    R_align_inv = R_align.T # inverse

    for img_id, img in images_info.items():

        # R_w2c originale
        R_w2c = img['rot_from_w_to_c']

        # nuova rotazione:
        # R_w2c' = R_w2c @ R_align^{-1}
        R_w2c_new = R_w2c @ R_align_inv

        # aggiorna la struttura
        img['rot_from_w_to_c'] = R_w2c_new

        # tvec NON si tocca
        # img['trans_from_w_to_c'] = img['trans_from_w_to_c']

# ============================================================================
# UNITY EXPORT HELPERS (DO NOT AFFECT COLMAP / VANILLA PIPELINE)
# ============================================================================


    """
    Create a Unity-ready cameras.json.
    Uses physically rotated camera poses (camera centers + orientations).

    Creates a json file with pos, rot, intr coherent with 'model_unity.ply'.
    """

    cams_json = []

    cam0 = next(iter(cameras_info.values()))
    fx = cam0['fx']
    fy = cam0['fy']
    width = cam0['width']
    height = cam0['height']

    for idx, (img_id, img) in enumerate(images_info.items()):
        R_wc = img['rot_from_w_to_c']
        t_wc = img['trans_from_w_to_c'].reshape(3)

        # camera center in world
        C = -R_wc.T @ t_wc

        # rotate world (Unity alignment)
        C_u = R_align @ C
        R_cw_u = (R_wc @ R_align.T).T

        cams_json.append({
            "id": idx,
            "img_name": img['filename'],
            "width": width,
            "height": height,
            "position": C_u.tolist(),
            "rotation": R_cw_u.tolist(),
            "fx": fx,
            "fy": fy,
        })

    print("Saving Unity cameras.json to:" + save_json_path)
    with open(save_json_path, "w") as f:
        json.dump(cams_json, f, indent=2)

def export_unity_gs_ply(data, xyz, nxyz, quat_gs, out_ply_path):
    """
    Export a Unity-ready Gaussian Splatting PLY.

    Assumptions:
    - Input model is already aligned correctly in COLMAP space
    - Unity fix corresponds to a MIRROR on Y axis (same as scale Y = -1)
    - GS quaternion format: (w, x, y, z)

    This function:
    - Mirrors positions on Y
    - Mirrors normals on Y
    - Mirrors rotations correctly via rotation matrices (NO spiky artifacts)
    """

    # -------------------------------------------------
    # copy structured array
    # -------------------------------------------------
    new_data = np.empty(data.shape, dtype=data.dtype)
    for name in data.dtype.names:
        new_data[name] = data[name]

    # -------------------------------------------------
    # 1. POSITIONS (mirror Y)
    # -------------------------------------------------
    xyz_u = xyz.copy()
    xyz_u[:, 1] *= -1.0

    # -------------------------------------------------
    # 2. NORMALS (mirror Y)
    # -------------------------------------------------
    nxyz_u = nxyz.copy()
    nxyz_u[:, 1] *= -1.0

    # -------------------------------------------------
    # 3. ROTATIONS (correct quaternion mirroring)
    # -------------------------------------------------
    # GS quaternions: (w, x, y, z)
    # scipy wants:   (x, y, z, w)
    quat_xyzw = quat_gs[:, [1, 2, 3, 0]]

    # quaternion -> rotation matrices
    Rm = R.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3)

    # mirror matrix for Y axis
    F = np.diag([1.0, -1.0, 1.0])

    # apply reflection: R' = F * R * F
    Rm_u = F @ Rm @ F

    # back to quaternion
    quat_u_xyzw = R.from_matrix(Rm_u).as_quat()
    quat_u = quat_u_xyzw[:, [3, 0, 1, 2]]  # back to (w,x,y,z)

    # -------------------------------------------------
    # 4. write back to ply structure
    # -------------------------------------------------
    new_data['x'] = xyz_u[:, 0]
    new_data['y'] = xyz_u[:, 1]
    new_data['z'] = xyz_u[:, 2]

    new_data['nx'] = nxyz_u[:, 0]
    new_data['ny'] = nxyz_u[:, 1]
    new_data['nz'] = nxyz_u[:, 2]

    new_data['rot_0'] = quat_u[:, 0]
    new_data['rot_1'] = quat_u[:, 1]
    new_data['rot_2'] = quat_u[:, 2]
    new_data['rot_3'] = quat_u[:, 3]

    # -------------------------------------------------
    # save
    # -------------------------------------------------
    print("Saving UNITY GS ply to:\n", out_ply_path)
    vertex_out = PlyElement.describe(new_data, 'vertex')
    PlyData([vertex_out], text=False).write(out_ply_path)

def export_unity_cameras_json(images_info_world, cameras_info, out_json_path):
    """
    Export Unity-compatible cameras.json from COLMAP-aligned world cameras.
    Assumes:
    - images_world is a CLEAN COPY (not reused for COLMAP output)
    - GS model has been flipped on Y for Unity
    """

    cams_json = []

    for img_id, img in images_info_world.items():

        # -------------------------
        # COLMAP world → camera
        # -------------------------
        R_wc = img["rot_from_w_to_c"]      # world -> camera
        t_wc = img["trans_from_w_to_c"]    # world -> camera

        # -------------------------
        # Camera center in world
        # C = -R^T * t
        # -------------------------
        R_cw = R_wc.T
        cam_pos = -R_cw @ t_wc
        cam_pos = cam_pos.flatten()

        # -------------------------
        # FLIP Y (Unity world)
        # must match GS flip
        # -------------------------
        cam_pos[1] *= -1.0

        # -------------------------
        # Rotation for Unity
        # COLMAP: camera looks -Z
        # Unity: camera looks +Z
        # -------------------------
        R_unity = R_cw.copy()

        # flip Y axis in rotation
        R_unity[:, 1] *= -1.0
        R_unity[1, :] *= -1.0

        # flip forward (Z)
        R_unity[:, 2] *= -1.0

        # -------------------------
        # Intrinsics
        # -------------------------
        cam = cameras_info[img["camera_id"]]

        cams_json.append({
            "id": img_id,
            "img_name": img["filename"],
            "width": cam["width"],
            "height": cam["height"],
            "position": cam_pos.tolist(),
            "rotation": R_unity.tolist(),
            "fx": cam["fx"],
            "fy": cam["fy"],
        })

    import json
    with open(out_json_path, "w") as f:
        json.dump(cams_json, f, indent=2)

    print("Saved Unity cameras.json to:", out_json_path)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--sparse_folder", type=str, required=True, help="COLMAP sparse folder.")
parser.add_argument("--cameras_bin_path", type=str, default=None, help="Path to cameras.bin.")
parser.add_argument("--images_bin_path", type=str, default=None, help="Path to images.bin.")
parser.add_argument("--ply_path_to_align", type=str, required=True, default=None, help="Path to ply file to align.")
parser.add_argument("--save_ply_path", type=str, default=None, help="Path to save aligned ply file.")
parser.add_argument("--save_log_txt", type=str, default=None, help="Path to save log txt.")
parser.add_argument("--dataset_base_folder", type=str, required=True, help="Dataset base folder.")
parser.add_argument("--save_images_cameras_folder", type=str, help="Path to folder to save the aligned images.bin and cameras.bin")
parser.add_argument("--json_path_to_align", type=str, required=True, help="Path to cameras.json file to align.")
parser.add_argument("--save_json_path", type=str, help="Path to save aligned cameras.json file.")
parser.add_argument("--skip_images_cameras_bin", default=False, action="store_true", help="Skip creation of images.bin and cameras.bin")
parser.add_argument("--skip_cameras_json", default=False, action="store_true", help="Skip creation of cameras.json")
parser.add_argument("--save_unity_ply_path", type=str, default=None, help="Path to save aligned ply file for Unity.")
parser.add_argument("--save_unity_json_path", type=str, default=None, help="Path to save aligned cameras.json file for Unity.")


args = parser.parse_args()
sparse_folder = args.sparse_folder
cameras_bin_path = args.cameras_bin_path
images_bin_path = args.images_bin_path
ply_path_to_align = args.ply_path_to_align
save_ply_path = args.save_ply_path
save_log_txt = args.save_log_txt
dataset_base_folder = args.dataset_base_folder
save_images_cameras_folder = args.save_images_cameras_folder
json_path_to_align = args.json_path_to_align
save_json_path = args.save_json_path
skip_images_cameras_bin = args.skip_images_cameras_bin
skip_cameras_json = args.skip_cameras_json
save_unity_ply_path = args.save_unity_ply_path
save_unity_json_path = args.save_unity_json_path

if cameras_bin_path is None:
    cameras_bin_path = f'{sparse_folder}/cameras.bin'

if images_bin_path is None:
    images_bin_path = f'{sparse_folder}/images.bin'

if save_log_txt is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_log_txt = f'{dataset_base_folder}/align_with_world_axis_infos_{timestamp}.txt'

if save_ply_path is None:
    ply_filename = Path(ply_path_to_align).stem
    ply_extension = Path(ply_path_to_align).suffix
    ply_to_read_folder = str(Path(ply_path_to_align).parent)
    save_ply_path = f'{ply_to_read_folder}/{ply_filename}_aligned_to_world_axis{ply_extension}'

if save_unity_ply_path is None:
    ply_filename = Path(ply_path_to_align).stem
    ply_extension = Path(ply_path_to_align).suffix
    ply_to_read_folder = str(Path(ply_path_to_align).parent)
    save_unity_ply_path = f'{ply_to_read_folder}/{ply_filename}_aligned_to_world_axis_unity{ply_extension}'

if save_images_cameras_folder is None:
    save_images_cameras_folder = f'{dataset_base_folder}/sparse_aligned_world/0'
    os.makedirs(save_images_cameras_folder, exist_ok=True)

if save_json_path is None:
    json_path_to_align = str(Path(ply_path_to_align).parent)
    save_json_path = f'{json_path_to_align}/cameras_aligned_to_world_axis.json'

if save_unity_json_path is None:
    json_path_to_align = str(Path(ply_path_to_align).parent)
    save_unity_json_path = f'{json_path_to_align}/cameras_aligned_to_world_axis_unity.json'

if skip_images_cameras_bin or skip_cameras_json:
    print("WARNING: Skipping creation of the following files:")
    if skip_images_cameras_bin:
        print(" - images.bin & cameras.bin")
    if skip_cameras_json:
        print(" - cameras.json")

sys.stdout = DualWriter(save_log_txt)
sys.stderr = sys.stdout

reconstruction = pycolmap.Reconstruction(sparse_folder)
print(reconstruction.summary())

cameras_info, intrinsic_matrix = parse_cameras(cameras_bin_path)
images_info = parse_images(images_bin_path)

# ------------- up-vectors
up_vectors = compute_up_vector(images_info)

# ------------- find rotation to align up-vectors with global Y axis
R_align = find_rot_align_pca(up_vectors)

# ------------ apply rotation to all cameras
rotate_cameras(images_info, R_align)

# ----------------------- BEGIN 'COLMAP' FLOW -----------------------
# start from images.bin and model.ply
# compute r_align
# APPLY r_align to realign the colmap world
# - write new images.bin
# - write new ply
# from here on, this is the TRUE and REAL world
# 
# this is necessary to:
# - make the vanilla renderer work properly
# - make the environment map work properly
# - maintain coherence with colmap 

# ----------- apply same rotation to gaussians
data, xyz_new, nxyz_new, quat_new_gs = rotate_gaussians(ply_path_to_align, R_align)

# -------------- write new gaussians in ply in binary
save_gs_ply(data, xyz_new, nxyz_new, quat_new_gs, save_ply_path)

#------------------ realign (colmap) cameras
realign_colmap_cameras(R_align, images_info)
# keep separate copy for unity export to not mess with colmap cameras
images_info_world = deepcopy(images_info)

# ------------ save images.bin 
#(cameras.bin only has intrinsics that we did not change, so rewrite it as is)
if not skip_images_cameras_bin:
    print("Saving aligned images.bin to:\n" + f"{save_images_cameras_folder}/images.bin")
    write_images_binary_from_images_info(images_info, f"{save_images_cameras_folder}/images.bin")

    print("Copy original cameras.bin to (did not change):\n" + f"{save_images_cameras_folder}/cameras.bin")
    shutil.copy(cameras_bin_path, f"{save_images_cameras_folder}/cameras.bin")
else:
    print("Skipped images.bin creation.")
    print("Skipped cameras.bin copy.")

# ------------- create cameras.json
if not skip_cameras_json:
    create_cameras_json(cameras_info, images_info, save_json_path)
else:
    print("Skipped cameras.json creation.")

# ----------------------- BEGIN 'UNITY' FLOW -----------------------
# unity does not live in the same referene system of colmap/gs vanilla
# so we need to create separate files for unity
# colmap is:
# right-handed, y up (mathematic), camera look -z
# unity is:
# left-handed, y up (engine), camera look +z
#
# important:
# these functions are 'consumers' of the already aligned data

export_unity_gs_ply(data, xyz_new, nxyz_new, quat_new_gs, save_unity_ply_path)

# NOTE: cameras are still upside down in unity, not sure why, i don't have time to fix this
export_unity_cameras_json(images_info_world, cameras_info, save_unity_json_path)

'''
example call:

conda activate 3DGS-VR

python align_gs_and_cameras_with_world_axis.py --sparse_folder "C:\\Users\\User\\Desktop\\Gaussian Splatting\\datasets\\datasets_processed\\PAPER_eurographics_and_I3D\\fields\\train\\sparse_original\\0" --ply_path_to_align "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-2ndPass-NO_BG\\point_cloud_30000_no_bg.ply" --dataset_base_folder "C:\\Users\\User\\Desktop\\Gaussian Splatting\\datasets\\datasets_processed\\PAPER_eurographics_and_I3D\\fields\\train" --json_path_to_align "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-2ndPass-NO_BG\\cameras.json"
'''