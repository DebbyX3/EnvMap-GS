import numpy as np
import cv2
import argparse
import collections
import os
import struct
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from pathlib import Path
import argparse

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
    """Parsa il file images.txt per estrarre le pose delle immagini"""
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

def create_point_cloud_and_distance_map_from_center(rgb_image, depth_map, K, R, t, scene_center, show_plot=False):
    """
    Crea una point cloud a partire da un'immagine RGB, mappa di profondità, matrice K, rotazione R e traslazione t,
    e salva la mappa delle distanze da un centro arbitrario (centro della scena).
    
    Parametri:
    - rgb_image: H x W x 3, immagine a colori
    - depth_map: H x W, profondità per ogni pixel
    - K: 3x3 matrice intrinseca della camera
    - R: 3x3 matrice di rotazione (camera-to-world)
    - t: 3x1 vettore di traslazione (camera-to-world)
    - scene_center: 3x1 vettore che rappresenta il centro della scena (in coordinate mondo)

    Ritorna:
    - points_world: N x 3 punti 3D nel sistema mondo
    - colors: N x 3 colori RGB corrispondenti
    - distance_map: H x W mappa delle distanze dal centro
    """
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))  # colonne (x), righe (y)
    i = i.reshape(-1)
    j = j.reshape(-1)
    depths = depth_map.reshape(-1)

    # Inverti K
    K_inv = np.linalg.inv(K)

    # Costruisci coordinate omogenee dei pixel: [u, v, 1]
    pixels_hom = np.stack([i, j, np.ones_like(i)], axis=0)  # shape: 3 x N

    # Calcola raggi dalla camera: K_inv @ pixel * depth
    cam_points = K_inv @ pixels_hom  # shape: 3 x N
    cam_points = cam_points * depths  # scala ogni raggio per la profondità
    cam_points = cam_points.T  # shape: N x 3

    # Trasforma in coordinate mondo: X_world = R @ X_cam + t
    points_world = (R @ cam_points.T + t.reshape(3, 1)).T  # shape: N x 3

    # Calcola la distanza euclidea di ogni punto rispetto a un centro arbitrario
    distances = np.linalg.norm(points_world - scene_center.ravel(), axis=1)

    # Crea una mappa delle distanze della stessa forma della depth map
    distance_map = distances.reshape(depth_map.shape)

    if show_plot:
        # Mostra la mappa delle distanze e la depth map di partenza
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(depth_map, cmap='viridis')
        plt.colorbar(label='Depth')
        plt.title('Depth Map')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(distance_map, cmap='viridis')
        plt.colorbar(label='Distanza dal centro (m)')
        plt.title('Mappa delle distanze')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Estrai colori
    colors = rgb_image[j, i] / 255.0  # normalizza in [0, 1]

    return points_world, colors, distance_map

def show_point_cloud_open3d(points, colors=None, cameras_point_cloud=None):
    """
    Visualizza una point cloud con Open3D.

    Parametri:
    - points: N x 3 array numpy di punti 3D
    - colors: (opzionale) N x 3 array numpy con colori RGB normalizzati (0-1)
    """
    # Crea oggetto point cloud Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualizza
    if cameras_point_cloud is not None:
        o3d.visualization.draw_geometries([pcd, cameras_point_cloud])
    else:
        o3d.visualization.draw_geometries([pcd])

def estimate_scale_from_sparse(sparse_depth, dense_depth):
    mask = (sparse_depth > 0)
    sparse = sparse_depth[mask]
    dense = dense_depth[mask]

    if len(sparse) == 0:
        raise ValueError("Nessun punto valido trovato per stimare la scala.")

    scale = np.median(sparse / dense)
    return scale

def stima_funzione_scala(depth_png, depth_sparse, mostra_plot=True):
    """
    Stima la funzione di scala per trasformare depth PNG (0-255) in metri.
    
    Parametri:
        depth_png (np.ndarray): Mappa PNG (grayscale) con valori 0-255.
        depth_sparse (np.ndarray): Mappa sparsa (stessa risoluzione), valori in metri, 0 dove ignoto.
        mostra_plot (bool): Se True, mostra grafico del fit.

    Ritorna:
        a (float), gamma (float): Parametri della funzione: depth_metrica = a * (depth_rel) ** gamma
    """
    depth_rel = depth_png.astype(np.float32) / 255.0
    valid_mask = (depth_sparse > 0) & (depth_rel > 0)
    
    x = depth_rel[valid_mask]
    y = depth_sparse[valid_mask]

    # Fit log-log: log(y) = log(a) + gamma * log(x)
    log_x = np.log(x)
    log_y = np.log(y)
    A = np.vstack([np.ones_like(log_x), log_x]).T
    beta0, gamma = np.linalg.lstsq(A, log_y, rcond=None)[0]
    a = np.exp(beta0)

    if mostra_plot:
        x_plot = np.linspace(0.01, 1.0, 500)
        y_plot = a * x_plot ** gamma

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, s=2, alpha=0.4, label="Punti validi")
        plt.plot(x_plot, y_plot, 'r', label=f'Fit: a={a:.2f}, γ={gamma:.2f}')
        plt.xlabel("Valore normalizzato (depth PNG / 255)")
        plt.ylabel("Profondità metrica (metri)")
        plt.title("Stima funzione di scala depth")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return a, gamma

def depth_to_metric(depth_png, a, gamma):
    depth_rel = depth_png.astype(np.float32) / 255.0
    return a * np.power(depth_rel, gamma)

'''
# ---------------brgRmSmParkFullFramesCompletePipeline-------------------
cameras_bin_path = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0/cameras.bin"
images_bin_path = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0/images.bin"
images_folder = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/images"
depth_maps_folder = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/original/depths"
save_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances'
#depth_map_from_3DPoints_folder = ''
#depth_maps_colmap_folder = ""

# ---------------fieldsCompletePipeline-------------------
cameras_bin_path = "../datasets/colmap_reconstructions/fieldsCompletePipeline/sparse/0/cameras.bin"
images_bin_path = "../datasets/colmap_reconstructions/fieldsCompletePipeline/sparse/0/images.bin"
images_folder = "../datasets/colmap_reconstructions/fieldsCompletePipeline/images"
depth_maps_folder = "../datasets/colmap_reconstructions/fieldsCompletePipeline/video-depth-anything-metric/original/depths"
save_folder = '../datasets/colmap_reconstructions/fieldsCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances'

# ---------------personDeborah-------------------
cameras_bin_path = "../datasets/person/person_deborah/colmap/sparse/0/cameras.bin"
images_bin_path = "../datasets/person/person_deborah/colmap/sparse/0/images.bin"
images_folder = "../datasets/person/person_deborah/images"
depth_maps_folder = "../datasets/person/person_deborah/video-depth-anything-metric/renamed_to_match_imgs"
save_folder = "../datasets/person/person_deborah/video-depth-anything-metric/fromSceneCenter/distances"

# ---------------personDeborahUndistorted-------------------
cameras_bin_path = "../datasets/person/person_deborah_undistorted/sparse/0/cameras.bin"
images_bin_path = "../datasets/person/person_deborah_undistorted/sparse/0/images.bin"
images_folder = "../datasets/person/person_deborah_undistorted/images_undist"
depth_maps_folder = "../datasets/person/person_deborah_undistorted/video-depth-anything-metric/original_from_img_seq/images_undist_depths_npz"
save_folder = "../datasets/person/person_deborah_undistorted/video-depth-anything-metric/fromSceneCenter/distances"
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute distance maps from scene center using camera and depth info.")
    parser.add_argument("--cameras_bin_path", type=str, required=True, help="Path to cameras.bin")
    parser.add_argument("--images_bin_path", type=str, required=True, help="Path to images.bin")
    parser.add_argument("--images_folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--depth_maps_folder", type=str, required=True, help="Folder containing depth maps")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save distance maps")
    parser.add_argument("--silent", action="store_true", default=False, help="If set, suppress output messages")

    args = parser.parse_args()

    cameras_bin_path = args.cameras_bin_path
    images_bin_path = args.images_bin_path
    images_folder = args.images_folder
    depth_maps_folder = args.depth_maps_folder
    save_folder = args.save_folder
    silent = args.silent

    cameras_info, intrinsic_matrix = parse_cameras(cameras_bin_path)
    images_info = parse_images(images_bin_path)

    all_points = []
    all_colors = []

    cameras_centers = []
    i = 0

    ''' ------ GATHER INFO FOR EACH IMAGE AND FIND THE CAMERA CENTER '''

    for image_id, img in images_info.items():   
        
        # ------- CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
        # Convert quaternion from world-to-camera (colmap) to camera-to-world
        # Please remember the convention! R_wc means "rotation from camera to world"! and NOT viceversa!
        rot_from_c_to_w = img['rot_from_w_to_c'].transpose()  # camera-to-world: R_wc =  R_cw.T
        images_info[image_id]['rot_from_c_to_w'] = rot_from_c_to_w # assign it to the image info array dict
        
        if not silent:
            print("\n\nRotation matrix world to cam:\n", img['rot_from_w_to_c'])
            print("rotation matrix cam to world (converted)\n", rot_from_c_to_w)

        # ------- CREATE TRANSLATION ARRAY 'T' FROM TRANSLATION
        # Convert translation from world-to-camera (colmap) to camera-to-world
        trans_from_c_to_w = (-rot_from_c_to_w) @ img['trans_from_w_to_c'] # camera-to-world: t_wc = -R_wc @ t_cw (oppure: t_wc = -R_cw.T @ t_cw)
        images_info[image_id]['trans_from_c_to_w'] = trans_from_c_to_w # assign it to the image info array dict

        if not silent:
            print("\nTranslation vector world to camera:\n", img['trans_from_w_to_c'])
            print("Translation vector camera to world (converted)\n", -(rot_from_c_to_w) @ img['trans_from_w_to_c'].reshape(3, 1))

        # camera centers are the same as trans_from_c_to_w: -R^t * T
        cameras_centers.append(trans_from_c_to_w)

    # find the middle point between all the cameras centers
    cameras_centers_np = np.array(cameras_centers)
    scene_center = np.mean(cameras_centers_np, axis=0)

    if not silent:
        print("Center of scene (relative to cameras):", scene_center)

    # create output folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    if not silent:
        # Visualizza le posizioni delle camere (rosso) e il centro della scena (verde) con Open3D
        cameras_point_cloud = o3d.geometry.PointCloud()
        cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_centers)
        cameras_point_cloud.paint_uniform_color([1, 0, 0])  # rosso

        scene_center_point_cloud = o3d.geometry.PointCloud()
        scene_center_point_cloud.points = o3d.utility.Vector3dVector(scene_center.reshape(1, 3))
        scene_center_point_cloud.paint_uniform_color([0, 1, 0])  # verde

        o3d.visualization.draw_geometries([cameras_point_cloud, scene_center_point_cloud])

    ''' ------ LOAD IMAGES AND DEPTH MAPS, CREATE POINT CLOUDS AND FIND DISTANCE MAPS '''

    for image_id, img in images_info.items():
        image_path = os.path.join(images_folder, img['filename'])
        if not os.path.exists(image_path):
            if not silent:
                print(f"Image not found: {image_path}. Skipping.")
            continue
        # --------------- IMAGE
        # load image rgb
        rgb = cv2.imread(images_folder + "/" + img['filename'])[:, :, ::-1]  # da BGR a RGB

        # --------------- DENSE DEPTH MAP
        # mappa densa, da scalare
        
        # IN CASE OF PNG DEPTH MAPS
        #depth_dense_filename = img['filename'] 
        #depth_dense = depth_uint8 = cv2.imread(depth_maps_folder + "/" + depth_dense_filename, 
        #                            cv2.IMREAD_GRAYSCALE)#.astype(np.float32)              # depth map H x W
        
        # IN CASE OF DEPTH MAPS FROM DEPTH ANYTHING
        # Need to invert the map to match the colmap rerpresentation of lower vals = closer point (0: closest, 255: farthest) 
        #depth_dense = np.invert(depth_dense)

        # IN CASE OF NUMPY DEPTH MAPS
        depth_dense_filename = Path(img['filename']).stem + "_depth.npz" # remove file extension
        depth_dense = np.load(depth_maps_folder + "/" + depth_dense_filename)['depth']

        # IN CASE OF COLMAP DEPTH MAPS
        #depth_dense_filename = img['filename'] + '.geometric.bin' # get the filename of the depth map
        #depth_map_path = os.path.join(depth_maps_colmap_folder, depth_dense_filename)
        #depth_dense = read_array(depth_map_path)

        # --------------- SPARSE DEPTH MAP FROM 3D POINTS
        # mappa sparsa colmap da 3DPoints - sono nella scala di colmap
        #depth_sparse_filename = img['filename'] + "_depth.npy"
        #depth_from_3dPoints = np.load(depth_map_from_3DPoints_folder + "/" + depth_sparse_filename)

        # ---------------- ESITIMATE SCALE (1ST METHOD) 
        '''
        # normalize depth map to [0, 1]
        depth_dense = depth_dense / 255.0 
        # Applica il fattore di scala
        scale = estimate_scale_from_sparse(depth_from_3dPoints, depth_dense)    
        scale = 1
        depth_scaled = depth_dense * scale

        # Moltiplica per 5 tutti i valori massimi in depth_scaled
        depth_scaled = depth_scaled.astype(np.float32)
        max_val = np.max(depth_scaled)    
        depth_scaled[depth_scaled >= max_val] *= 5

        print(f"Fattore di scala stimato: {scale}")
        '''

        # ---------------- ESITIMATE SCALE (2ND METHOD) 
        '''
        a, gamma = stima_funzione_scala(depth_dense, depth_from_3dPoints, mostra_plot=True)
        depth_scaled = depth_to_metric(depth_dense, a, gamma)  
        '''

        '''Sembra solo peggio cercare di correggere le depth...'''

        # K are the camera intrinsics
        # R has to be camera-to-world
        # T has to be camera-to-world
        points, colors, distance_map = create_point_cloud_and_distance_map_from_center(rgb, 
                                                                                    depth_dense, 
                                                                                    intrinsic_matrix, 
                                                                                    img['rot_from_c_to_w'], 
                                                                                    img['trans_from_c_to_w'],
                                                                                    scene_center,
                                                                                    show_plot=False)
        
        # salva la mappa delle distanze in npz
        scene_distance_path = os.path.join(save_folder, f"{Path(img['filename']).stem}_distance.npz")
        np.savez_compressed(scene_distance_path, distance=distance_map)
        
        if not silent:
            print(f"Saved distance map {Path(img['filename']).stem}_distance.npz")

        # Accumula punti e colori
        all_points.append(points)
        all_colors.append(colors)


    # Dopo il ciclo: concatena e visualizza la nuvola totale
    '''
    if all_points and all_colors:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        # ------- SHOW CAMERAS IN 3D (RED)
        # Dopo aver calcolato tutti i camera_center
        # Create new point cloud, add camera centers
        cameras_point_cloud = o3d.geometry.PointCloud()
        cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_centers)

        # Paint them red
        cameras_point_cloud.paint_uniform_color([1, 0, 0])

        show_point_cloud_open3d(all_points, all_colors, cameras_point_cloud=cameras_point_cloud)

    else:
        print("No points or colors to display.")
    '''