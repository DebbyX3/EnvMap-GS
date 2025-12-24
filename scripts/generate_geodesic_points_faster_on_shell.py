import random
import sys
import open3d as o3d
import numpy as np
import struct
from plyfile import PlyData, PlyElement
import pycolmap
import PIL.Image as PILImage
import os
import collections
from scipy.spatial.transform import Rotation as R
import argparse
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
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

#### END FROM COLMAP CODEBASE ####

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


def icosphere(subdivisions=2, radius=1.0, center=np.array([0.0, 0.0, 0.0]), return_centroids=False):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]])
    vertices /= np.linalg.norm(vertices[0])
    vertices *= radius
    
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    
    def midpoint(v1, v2):
        mid = (v1 + v2) / 2.0
        return mid / np.linalg.norm(mid) * radius
    
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}
        
        def get_midpoint(i1, i2):
            if (i1, i2) not in midpoint_cache:
                if (i2, i1) in midpoint_cache:
                    return midpoint_cache[(i2, i1)]
                midpoint_cache[(i1, i2)] = len(vertices)
                vertices.append(midpoint(vertices[i1], vertices[i2]))
            return midpoint_cache[(i1, i2)]
        
        vertices = list(vertices)
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        
        faces = np.array(new_faces)
        vertices = np.array(vertices)
    
    if return_centroids:
        centroids = np.mean(vertices[faces], axis=1)
        return centroids + center, faces
    
    return vertices + center, faces

def save_as_ply(points, colors, filename="gaussians.ply"):
    normals = -points / np.linalg.norm(points, axis=1, keepdims=True)  # Verso il centro
    vertex_data = np.array([
        (p[0], p[1], p[2], c[0], c[1], c[2], n[0], n[1], n[2])
        for p, c, n in zip(points, colors, normals)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u4'), ('green', 'u4'), ('blue', 'u4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_element]).write(filename)

def find_most_distant_point(point_cloud, initial_point):
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    
    # Calculate the Euclidean distance from the initial point to all points in the point cloud
    distances = np.linalg.norm(points - initial_point, axis=1)
    
    # Find the index of the maximum distance
    max_distance_index = np.argmax(distances)
    
    # Get the most distant point
    most_distant_point = points[max_distance_index]
    
    # Get the maximum distance
    max_distance = distances[max_distance_index]
    
    return most_distant_point, max_distance

def calculate_circumradius(vertices, faces):
    circumradii = []
    for face in faces:
        a, b, c = vertices[face]
        # Lengths of sides of the triangle
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        # Semi-perimeter
        s = (ab + bc + ca) / 2
        # Area of the triangle using Heron's formula
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        # Circumradius formula
        circumradius = (ab * bc * ca) / (4 * area)
        circumradii.append(circumradius)
    return max(circumradii)

def calculate_circumradius(vertices, faces):
    # Vectorized computation for all faces at once
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    ab = np.linalg.norm(a - b, axis=1)
    bc = np.linalg.norm(b - c, axis=1)
    ca = np.linalg.norm(c - a, axis=1)

    s = (ab + bc + ca) / 2
    # Heron's formula for area
    area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
    # Avoid division by zero
    area[area == 0] = 1e-12

    circumradius = (ab * bc * ca) / (4 * area)
    return np.max(circumradius)

def create_circle(center, normal, radius, resolution=30):
    """
    Create a circle in 3D space using Open3D.
    
    Parameters:
    - center: The center of the circle.
    - normal: The normal vector of the circle plane.
    - radius: The radius of the circle.
    - resolution: The number of points to generate the circle.
    
    Returns:
    - circle: An Open3D LineSet representing the circle.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle_points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)]).T
    
    # Create a rotation matrix to align the circle with the normal vector
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k + k @ k * (1 / (1 + c))
    
    # Rotate and translate the circle points
    circle_points = circle_points @ rotation_matrix.T + center
    
    # Create the LineSet for the circle
    lines = [[i, (i + 1) % resolution] for i in range(resolution)]
    circle = o3d.geometry.LineSet()
    circle.points = o3d.utility.Vector3dVector(circle_points)
    circle.lines = o3d.utility.Vector2iVector(lines)
    return circle

# Funzione per calcolare la distanza euclidea tra due punti 3D
def distance_3d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def project_points(points_3D, rot, trans, K):
    """
    Proietta in batch punti 3D (Nx3) su immagine usando rot (3x3), trans (3x1), K (3x3).
    Ritorna Nx2 array di pixel e maschera validi (davanti alla camera).
    """
    pts_cam = (rot @ points_3D.T + trans).T  # Nx3
    z = pts_cam[:, 2]
    valid = z > 0
    pts_cam = pts_cam[valid]
    z = z[valid]
    pts_norm = pts_cam / z[:, None]
    pts_img = (K @ pts_norm.T).T  # Nx3
    pts_2d = pts_img[:, :2]
    return pts_2d, valid


parser = argparse.ArgumentParser(description="Generate geodesic points with color projection.")
parser.add_argument("--subdivisions", type=int, default=7, help="Number of icosphere subdivisions.")
parser.add_argument("--radius_mult", type=float, default=10, help="Radius multiplier for icosphere.")
parser.add_argument("--sparse_folder", type=str, required=True, help="COLMAP sparse folder.")
parser.add_argument("--cameras_bin_path", type=str, default=None, help="Path to cameras.bin.")
parser.add_argument("--images_bin_path", type=str, default=None, help="Path to images.bin.")
parser.add_argument("--images_folder", type=str, required=True, help="Images folder.")
parser.add_argument("--images_bg_folder", type=str, required=True, help="Background images folder.")
parser.add_argument("--dataset_base_folder", type=str, required=True, help="Dataset base folder.")
parser.add_argument("--dataset_distances_folder", type=str, required=True, help="Dataset distances folder.")
parser.add_argument("--save_ply_path", type=str, default=None, help="Path to save ply file.")
parser.add_argument("--save_log_txt", type=str, default=None, help="Path to save log txt.")
parser.add_argument("--show_points", action="store_true", default=False, help="Show points in Open3D viewer.")
#parser.add_argument("--shell_thickness_percent", type=float, default=0.3, help="Shell thickness percentage for generator.")
parser.add_argument("--object_threshold", type=float, required=True, default=None, help="Object threshold used to separate background from foreground.")
parser.add_argument("--infinity_threshold", type=float, required=False, help="Distance threshold to consider points at infinity (e.g. sky)")

args = parser.parse_args()

subdivisions = args.subdivisions
radius_mult = args.radius_mult
sparse_folder = args.sparse_folder
cameras_bin_path = args.cameras_bin_path
images_bin_path = args.images_bin_path
images_folder = args.images_folder
images_bg_folder = args.images_bg_folder
dataset_base_folder = args.dataset_base_folder
save_ply_path = args.save_ply_path
save_log_txt = args.save_log_txt
show_points = args.show_points
dataset_distances_folder = args.dataset_distances_folder
#shell_thickness_percent = args.shell_thickness_percent
object_threshold = args.object_threshold
infinity_threshold = args.infinity_threshold

if cameras_bin_path is None:
    cameras_bin_path = f'{sparse_folder}/cameras.bin'

if images_bin_path is None:
    images_bin_path = f'{sparse_folder}/images.bin'

# save_ply_path file name is below to use threshold in the name

if save_log_txt is None:
    save_log_txt = f'{dataset_base_folder}/dataset_info_donut_random_from_{object_threshold}_to_{infinity_threshold}.txt'

sys.stdout = DualWriter(save_log_txt)
sys.stderr = sys.stdout

reconstruction = pycolmap.Reconstruction(sparse_folder)
print(reconstruction.summary())

# LINESET to draw camera directions in 3d as 'vectors'
lineset = o3d.geometry.LineSet()
all_points = []
all_lines = []

cameras_data = {}

cameras_info, intrinsic_matrix = parse_cameras(cameras_bin_path)
images_info = parse_images(images_bin_path)


''' ------ GATHER INFO FOR EACH IMAGE AND FIND THE CAMERA CENTER '''

for image_id, img in images_info.items():   
    
    # ------- CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
    # Convert quaternion from world-to-camera (colmap) to camera-to-world
    # Please remember the convention! R_wc means "rotation from camera to world"! and NOT viceversa!
    rot_from_c_to_w = img['rot_from_w_to_c'].transpose()  # camera-to-world: R_wc =  R_cw.T
    images_info[image_id]['rot_from_c_to_w'] = rot_from_c_to_w # assign it to the image info array dict
    
    print("\n\nRotation matrix world to cam:\n", img['rot_from_w_to_c'])
    print("rotation matrix cam to world (converted)\n", rot_from_c_to_w)

    # ------- CREATE TRANSLATION ARRAY 'T' FROM TRANSLATION
    # Convert translation from world-to-camera (colmap) to camera-to-world
    trans_from_c_to_w = (-rot_from_c_to_w) @ img['trans_from_w_to_c'] # camera-to-world: t_wc = -R_wc @ t_cw (oppure: t_wc = -R_cw.T @ t_cw)
    images_info[image_id]['trans_from_c_to_w'] = trans_from_c_to_w # assign it to the image info array dict

    print("\nTranslation vector world to camera:\n", img['trans_from_w_to_c'])
    print("Translation vector camera to world (converted)\n", -(rot_from_c_to_w) @ img['trans_from_w_to_c'].reshape(3, 1))

    # ------- CREATE CAMERA COORDINATES + LINE DIRECTION

    # Store camera data
    cameras_data[image_id] = {
        "center": trans_from_c_to_w, # camera centers are the same as trans_from_c_to_w: -R^t * T
        "direction": rot_from_c_to_w
    }

    # Estrai il vettore di direzione della camera (asse z della camera nel mondo)
    # La "direction" è la matrice di rotazione camera-to-world: la terza colonna è la direzione forward della camera nel sistema mondo
    direction_vector = cameras_data[image_id]['direction'][:, 2]
   
    # cerca il punto finale per fare sta linea
    # punto di inizio è la camera stessa
    # punto finale = punto inizio + direzione * lunghezza vettore
    final_point = cameras_data[image_id]['center'].flatten() + direction_vector * 1.5

    # ora traccio linea
    all_points.append(cameras_data[image_id]['center'].flatten()) #altrimenti si lamenta poi open3d che non sono 1d
    all_points.append(final_point)

    # Aggiungi la linea tra gli ultimi due punti aggiunti
    idx = len(all_points)
    all_lines.append([idx - 2, idx - 1])  # Indici degli ultimi due punti


'''
# Accessing the values outside the loop
for camera_id, data in camera_data.items(): # camera_id = key, data = value
    print(f"Camera {camera_id} center: {data['center']}")
    print(f"Camera {camera_id} direction: {data['direction']}")
'''

# -------- Find center of all cameras (center of point cloud)

# Create new point cloud, add camera centers
cameras_centers = [data['center'].flatten() for data in cameras_data.values()]

cameras_point_cloud = o3d.geometry.PointCloud()
cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_centers)

center_of_scene = cameras_point_cloud.get_center()

most_distant_point, max_distance = find_most_distant_point(cameras_point_cloud, center_of_scene)

print("Most distant point:", most_distant_point)
print("Distance:", max_distance)
print("Center of scene:", center_of_scene)

sphere_radius = max_distance * radius_mult

# --------- create icosphere based on max_distance
ico_points_pos, ico_faces = icosphere(subdivisions = subdivisions, 
                            radius = sphere_radius,
                            center = center_of_scene, 
                            return_centroids = False) #keep false!!!!!!!!!!!

print("num of icosphere points generated: ", ico_points_pos.size/3)
print("radius of icosphere: ", sphere_radius)

if infinity_threshold is None:
    # Ottieni tutti i file delle immagini
    all_image_files = []
    for subdir, dirs, files in os.walk(images_folder):
        for file in files:
            all_image_files.append((subdir, file))

    # Scegli 3 immagini random
    random_samples = random.sample(all_image_files, min(3, len(all_image_files)))

    for subdir, file in random_samples:
        image_path = os.path.join(subdir, file)
        image_filename_base = Path(file).stem

        depth_filename = image_filename_base + "_distance.npz"
        depth_path = os.path.join(dataset_distances_folder, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Distance map not found for image {file}, skipping.")
            continue

        depth_map_file = np.load(depth_path)
        depth_map = depth_map_file['distance']

        plt.imshow(depth_map, cmap='viridis')
        plt.title("Metric Distance Map")
        plt.show()

    #ask after showing the first 3 distance maps
    # the threshold representing, e.g., the sky. all points beyond this distance are considered at infinity and splatted on the sphere
    print("Enter the threshold value for the infinity: ", flush=True)

    try:
        infinity_threshold = float(input())
    except Exception as e:
        print(e, flush=True)
else:    
    print(f"Using provided infinity threshold: {infinity_threshold}")
    print(f"__INFINITY_THRESHOLD__:{infinity_threshold}", flush=True)

# ------------- project 3d point onto the image

# Usiamo due dizionari per classificare i voti ---
point_votes_object = {}
point_votes_infinity = {}

for image_id, img_info in images_info.items():
    print(f"Processando immagine: {img_info['filename']}...")
    try:
        image_path = os.path.join(images_bg_folder, img_info['filename'])
        image_data = np.array(PILImage.open(image_path))

        dist_map_path = os.path.join(dataset_distances_folder, os.path.splitext(img_info['filename'])[0] + "_distance.npz")

        depth_map_file = np.load(dist_map_path)
        distance_map = depth_map_file['distance']

    except FileNotFoundError:
        print(f"  -> File non trovato, salto.")
        continue
    
    # Proietta tutti i punti dell'icosfera
    pts_2d, valid_mask = project_points(ico_points_pos, img_info['rot_from_w_to_c'], img_info['trans_from_w_to_c'], intrinsic_matrix)

    # 1. Ottieni gli indici originali dei punti che sono davanti alla camera
    original_indices_in_front = np.where(valid_mask)[0]

    # 2. Lavoriamo SOLO con i punti proiettati validi (quelli davanti alla camera)
    pts_2d_in_front = pts_2d
    
    # 3. Arrotonda queste coordinate
    rounded_pixels_in_front = np.round(pts_2d_in_front).astype(int)

    # 4. Filtra basandoti sui limiti dell'immagine
    mask_x = (rounded_pixels_in_front[:, 0] >= 0) & (rounded_pixels_in_front[:, 0] < image_data.shape[1])
    mask_y = (rounded_pixels_in_front[:, 1] >= 0) & (rounded_pixels_in_front[:, 1] < image_data.shape[0])
    in_frame_mask = mask_x & mask_y

    # 5. Ottieni gli indici finali e le coordinate dei pixel.
    #    Questi indici sono relativi all'array `original_indices_in_front`.
    final_local_indices = np.where(in_frame_mask)[0]

    # Gli indici originali nell'array `ico_points_pos` sono:
    visible_indices = original_indices_in_front[final_local_indices]
    
    # Le coordinate dei pixel sono:
    visible_pixels = rounded_pixels_in_front[final_local_indices]

    if len(visible_indices) == 0:
        continue
    
    # 6. Ora l'indicizzazione è sicura.
    pixel_distances = distance_map[visible_pixels[:, 1], visible_pixels[:, 0]]
    pixel_colors = image_data[visible_pixels[:, 1], visible_pixels[:, 0]]

    # Classifica ogni punto visibile come "cielo" o "oggetto" e aggiungi il voto
    for idx, dist, color in zip(visible_indices, pixel_distances, pixel_colors):
        # Ignora pixel trasparenti se presenti
        if len(color) == 4 and color[3] == 0:
            continue

        if dist >= infinity_threshold:
            # È un punto del cielo
            if idx not in point_votes_infinity:
                point_votes_infinity[idx] = []
            point_votes_infinity[idx].append(color)
        else:
            # È un punto di un oggetto di sfondo
            if idx not in point_votes_object:
                point_votes_object[idx] = []
            point_votes_object[idx].append(color)

    # Commenti:
    # - Questo codice evita il ciclo for pixel-per-pixel e sfrutta numpy per estrarre tutti i colori in un colpo solo.
    # - Funziona solo se l'immagine è caricata come array numpy (RGB o RGBA).
    # - Per immagini con canale alpha, scarta i pixel trasparenti.
    # - I colori e le coordinate vengono associati agli indici dei punti 3D proiettati.


print("\n--- Assemblaggio finale, filtraggio e spostamento del point cloud ---")

# 1. Estrai gli indici UNICI di tutti i punti che hanno ricevuto almeno un voto.
#    Questi sono i soli punti che esisteranno nel file finale.
infinity_indices_set = set(point_votes_infinity.keys())
object_indices_set = set(point_votes_object.keys())

# Diamo priorità al cielo: se un punto è in entrambi, è cielo.
object_indices_set -= infinity_indices_set 

# Converti in array numpy ordinati per un'indicizzazione coerente
final_infinity_indices = np.array(sorted(list(infinity_indices_set)))
final_object_indices = np.array(sorted(list(object_indices_set)))

total_validated_points = len(final_infinity_indices) + len(final_object_indices)
print(f"Punti totali candidati sull'icosfera: {len(ico_points_pos)}")
print(f"Punti validati finali: {total_validated_points} (di cui {len(final_infinity_indices)} cielo, {len(final_object_indices)} oggetti)")
print(f"Punti non visti (eliminati): {len(ico_points_pos) - total_validated_points}")

if total_validated_points == 0:
    print("ATTENZIONE: Nessun punto è stato validato. Il file .ply sarà vuoto.")
    final_points = np.array([])
    final_colors = np.array([])
else:
    # 2. Prepara gli array per i colori (solo per i punti validati)
    infinity_colors = np.array([np.mean(np.array(point_votes_infinity[idx]), axis=0) for idx in final_infinity_indices]).astype(np.uint8)
    object_colors = np.array([np.mean(np.array(point_votes_object[idx]), axis=0) for idx in final_object_indices]).astype(np.uint8)

    # 3. Posiziona i punti del cielo (già filtrati) ESATTAMENTE sulla sfera
    infinity_points = ico_points_pos[final_infinity_indices]

    # 4. Prendi i punti degli oggetti (già filtrati) e SPOSTALI VERSO L'INTERNO
    object_points_original = ico_points_pos[final_object_indices]

    # Guscio definito da soglia assoluta
    shell_thickness_val_gen = sphere_radius - object_threshold
    inner_radius_val_gen = object_threshold
    scaled_inner_radius_val_gen = sphere_radius * (object_threshold / infinity_threshold) 

    print("\n--- [VERIFICA GUSCIO GENERATORE] ---")
    print(f"Raggio Sfera Esterno (Generatore): {sphere_radius:.4f}")
    print(f"Spessore Guscio (Generatore): {shell_thickness_val_gen:.4f} (da soglia)")
    print(f"Raggio Sfera Interno (Generatore): {inner_radius_val_gen:.4f}")
    print(f"Raggio Sfera Interno (Generatore) Scalato: {scaled_inner_radius_val_gen:.4f}")
    print("-------------------------------------\n")

    # Shift random tra scaled_inner_radius_val_gen e sphere_radius
    shift_factors = np.random.uniform(scaled_inner_radius_val_gen, sphere_radius, size=(len(object_points_original),))
    vectors = object_points_original - center_of_scene
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    object_points_shifted = center_of_scene + unit_vectors * shift_factors[:, None]

    # 5. Combina i due set di punti e colori (che ora sono già filtrati)
    final_points = np.concatenate((infinity_points, object_points_shifted), axis=0)
    
    # Pulisci il canale alpha se presente
    if infinity_colors.shape and infinity_colors.shape[1] == 4: infinity_colors = infinity_colors[:,:3]
    if object_colors.shape and object_colors.shape[1] == 4: object_colors = object_colors[:,:3]
    final_colors = np.concatenate((infinity_colors, object_colors), axis=0)

# Salva il file .ply finale
if save_ply_path is None:
    save_ply_path = f'{dataset_base_folder}/points3D_{subdivisions}subd-{round(sphere_radius, 4)}radius-donut_between_{object_threshold}_and_{infinity_threshold}.ply'

print("Saving ply to: ", save_ply_path)
save_as_ply(final_points, final_colors, save_ply_path)

'''
print("\nInizio ricerca raggio ottimo")

# --------- find optimal radius of circles of point on the icosphere to cover all other circles
optimal_radius = calculate_circumradius(ico_points_pos, ico_faces)
print("Optimal radius for circles:", optimal_radius)
'''
'''
# --------- Create circles on the icosphere points
circles = []

for point in ico_points_pos:
    circle = create_circle(center=point, normal=point - center_of_scene, radius=optimal_radius)
    circles.append(circle)
'''


# ------- SHOW CAMERAS IN 3D (RED) + FORWARD VECTOR (GREEN)
# Paint camera coords red
cameras_point_cloud.paint_uniform_color([1, 0, 0])

# Create lineset
lineset.points = o3d.utility.Vector3dVector(all_points)
lineset.lines = o3d.utility.Vector2iVector(all_lines)

# Apply color to lineset
GREEN = [0.0, 1.0, 0.0]
lines_color = [GREEN] * len(lineset.lines)
lineset.colors = o3d.utility.Vector3dVector(lines_color)

print(f"__GEOD_FILENAME__:{os.path.basename(save_ply_path)}", flush=True)
print(f"__SCENE_CENTER__:{center_of_scene}", flush=True)
print(f"__SPHERE_RADIUS__:{sphere_radius}", flush=True)
print(f"__SCALED_INNER_RADIUS__:{scaled_inner_radius_val_gen}", flush=True)

# SE VUOI VISUALIZZARE SU OPEN3D, DIVIDI TUTTI I COLORI PER 255
# NON LI SALVO 'DIVISI' PERCHE GS LI VUOLE COME COLMAP, CIOE RGB CLASSICI
if show_points:
    print("\nVisualizzazione dei punti SPOSTATI in due set di colori...")
    
    # Crea un point cloud per gli oggetti (es. in blu)
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(object_points_shifted)
    pcd_objects.paint_uniform_color([1, 0, 0]) # Rosso per gli oggetti

    # Crea un point cloud per il cielo (es. in ciano)
    pcd_infinity = o3d.geometry.PointCloud()
    pcd_infinity.points = o3d.utility.Vector3dVector(infinity_points)
    pcd_infinity.paint_uniform_color([0, 1, 1]) # Ciano per il cielo

    print("Visualizzazione: Punti Blu = Oggetti (spostati all'interno), Punti Ciano = Cielo (sulla sfera)")
    o3d.visualization.draw_geometries([lineset, cameras_point_cloud, pcd_objects, pcd_infinity])

if show_points:
    print("\nVisualizzazione dei punti SPOSTATI con i loro colori reali...")
    
    if len(final_points) == 0:
        print("ATTENZIONE: Nessun punto da visualizzare.")
    else:
        # Crea un unico point cloud per la visualizzazione
        pcd_final_for_viz = o3d.geometry.PointCloud()
        
        # Assegna i punti finali (oggetti spostati + cielo sulla sfera)
        pcd_final_for_viz.points = o3d.utility.Vector3dVector(final_points)
        
        # Assegna i colori finali, convertendoli in formato 0-1 per Open3D
        final_colors_for_viz = final_colors / 255.0
        pcd_final_for_viz.colors = o3d.utility.Vector3dVector(final_colors_for_viz)

        # Visualizza il point cloud finale insieme alle telecamere
        print("Visualizzazione: Nuvola di punti finale con colori reali.")
        o3d.visualization.draw_geometries([lineset, cameras_point_cloud, pcd_final_for_viz])

if show_points:
    # Riapri e visualizza il file .ply appena salvato con Open3D
    if os.path.exists(save_ply_path):
        print(f"\nRiapertura e visualizzazione del file PLY appena salvato: {save_ply_path}")
        ply_pc = o3d.io.read_point_cloud(save_ply_path)
        o3d.visualization.draw_geometries([ply_pc])
    else:
        print(f"File PLY non trovato: {save_ply_path}")