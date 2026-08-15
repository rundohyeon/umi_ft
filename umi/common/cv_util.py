from __future__ import annotations

from typing import Dict, Tuple, Any

import json
import pathlib
import math
import copy
import numpy as np
import cv2
import scipy.interpolate as si


def _aruco_get_predefined_dictionary(predefined: str):
    dict_id = getattr(cv2.aruco, predefined)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def _aruco_make_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _aruco_detect_markers(img, aruco_dict, parameters):
    # OpenCV >= 4.7 removed the functional cv2.aruco.detectMarkers API.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return detector.detectMarkers(img)
    return cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_dict, parameters=parameters
    )


def _aruco_make_dictionary_subset(aruco_dict, tag_id_offset: int):
    bytes_list = aruco_dict.bytesList[tag_id_offset:]
    marker_size = aruco_dict.markerSize
    if hasattr(cv2.aruco, "Dictionary"):
        return cv2.aruco.Dictionary(bytes_list, marker_size)
    return cv2.aruco.Dictionary_create_from(bytes_list, marker_size)


def _aruco_make_charuco_board(aruco_dict, grid_size, square_length_mm, tag_length_mm):
    square_length = square_length_mm / 1000
    marker_length = tag_length_mm / 1000
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            size=grid_size,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=aruco_dict,
        )
    return cv2.aruco.CharucoBoard_create(
        squaresX=grid_size[0],
        squaresY=grid_size[1],
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=aruco_dict,
    )

# =================== intrinsics ===================

def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    focal_length is fx; aspect_ratio is fy/fx (default 1).
    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # Pinhole part of K: focal_length is fx; aspect_ratio is fy/fx (OpenCamera-style).
    f = float(intr_data['focal_length'])
    aspect = float(intr_data.get('aspect_ratio', 1.0))
    fx = f
    fy = f * aspect
    px = float(intr_data['principal_pt_x'])
    py = float(intr_data['principal_pt_y'])

    # OpenCV fisheye radial coefficients (k1..k4), not Kannala–Brandt.
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [fx, 0, px],
            [0, fy, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict


def parse_orb_slam_fisheye_intrinsics(yaml_path: str) -> Dict[str, np.ndarray]:
    """Read ORB-SLAM KannalaBrandt8 fisheye intrinsics into OpenCV format."""
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open camera intrinsics YAML: {yaml_path}")

    def read_real(key: str) -> float:
        node = fs.getNode(key)
        if node.empty():
            raise KeyError(f"Missing {key!r} in camera intrinsics YAML: {yaml_path}")
        return float(node.real())

    try:
        camera_type = fs.getNode("Camera.type").string()
        if camera_type and camera_type != "KannalaBrandt8":
            raise ValueError(
                f"Unsupported Camera.type={camera_type!r}; expected KannalaBrandt8"
            )

        fx = read_real("Camera1.fx")
        fy = read_real("Camera1.fy")
        cx = read_real("Camera1.cx")
        cy = read_real("Camera1.cy")
        k1 = read_real("Camera1.k1")
        k2 = read_real("Camera1.k2")
        k3 = read_real("Camera1.k3")
        k4 = read_real("Camera1.k4")
        w = int(round(read_real("Camera.width")))
        h = int(round(read_real("Camera.height")))
    finally:
        fs.release()

    return {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ),
        "D": np.array([[k1, k2, k3, k4]], dtype=np.float64).T,
    }


def parse_fisheye_intrinsics_file(intrinsics_path: str) -> Dict[str, np.ndarray]:
    """Load fisheye intrinsics from OpenCamera JSON or ORB-SLAM YAML."""
    path = pathlib.Path(intrinsics_path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r") as f:
            return parse_fisheye_intrinsics(json.load(f))
    if suffix in (".yaml", ".yml"):
        return parse_orb_slam_fisheye_intrinsics(str(path))
    raise ValueError(
        f"Unsupported intrinsics file extension {suffix!r}; expected .json/.yaml/.yml"
    )


def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray], 
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


class FisheyeRectConverter:
    def __init__(self, K, D, DIM, out_size, out_fov, auto_scale=True):
        self.base_intr_dict = {
            "K": np.asarray(K, dtype=np.float64),
            "D": np.asarray(D, dtype=np.float64),
            "DIM": np.asarray(DIM, dtype=np.int64),
        }
        self.out_size = tuple(int(x) for x in out_size)
        self.out_fov = float(out_fov)
        self.auto_scale = bool(auto_scale)
        self._map_cache = {}
        self.map1 = None
        self.map2 = None
        self._ensure_maps(tuple(int(x) for x in self.base_intr_dict["DIM"]))

    def _make_out_K(self):
        out_size = np.array(self.out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(self.out_fov/180*np.pi/2)
        return np.array([
            [out_f, 0, out_size[0]/2],
            [0, out_f, out_size[1]/2],
            [0, 0, 1],
        ], dtype=np.float32)

    def _ensure_maps(self, input_size):
        input_size = tuple(int(x) for x in input_size)
        if input_size not in self._map_cache:
            intr_dict = self.base_intr_dict
            if self.auto_scale and input_size != tuple(intr_dict["DIM"]):
                intr_dict = convert_fisheye_intrinsics_resolution(
                    intr_dict, target_resolution=input_size)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                intr_dict["K"],
                intr_dict["D"],
                np.eye(3),
                self._make_out_K(),
                self.out_size,
                cv2.CV_16SC2)
            self._map_cache[input_size] = (map1, map2)

        self.map1, self.map2 = self._map_cache[input_size]
    
    def forward(self, img):
        h, w = img.shape[:2]
        self._ensure_maps((w, h))
        rect_img = cv2.remap(img, 
            self.map1, self.map2,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return rect_img


# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result


def get_aruco_dict(predefined:str
                   ) -> Any:
    return _aruco_get_predefined_dictionary(predefined)

def detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: Any, 
        marker_size_map: Dict[int, float], 
        fisheye_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True):
    K = fisheye_intr_dict['K']
    D = fisheye_intr_dict['D']
    param = _aruco_make_detector_parameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejectedImgPoints = _aruco_detect_markers(img, aruco_dict, param)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        
        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            undistorted, marker_size_m, K, np.zeros((1,5)))
        tag_dict[this_id] = {
            'rvec': rvec.squeeze(),
            'tvec': tvec.squeeze(),
            'corners': this_corners.squeeze()
        }
    return tag_dict

def get_charuco_board(
        aruco_dict=None,
        tag_id_offset=50,
        grid_size=(8, 5), square_length_mm=50, tag_length_mm=30):

    if aruco_dict is None:
        aruco_dict = _aruco_get_predefined_dictionary("DICT_4X4_100")
    aruco_dict = _aruco_make_dictionary_subset(aruco_dict, tag_id_offset)
    board = _aruco_make_charuco_board(
        aruco_dict=aruco_dict,
        grid_size=grid_size,
        square_length_mm=square_length_mm,
        tag_length_mm=tag_length_mm,
    )
    return board

def draw_charuco_board(board, dpi=300, padding_mm=15):
    grid_size = np.array(board.getChessboardSize())
    square_length_mm = board.getSquareLength() * 1000

    mm_per_inch = 25.4
    board_size_pixel = (grid_size * square_length_mm + padding_mm * 2) / mm_per_inch * dpi
    board_size_pixel = board_size_pixel.round().astype(np.int64)
    padding_pixel = int(padding_mm / mm_per_inch * dpi)
    board_img = board.generateImage(outSize=board_size_pixel, marginSize=padding_pixel)
    return board_img

def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =========== image mask ====================
def canonical_to_pixel_coords(coords, img_shape=(3000, 4000)):
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts

def pixel_coords_to_canonical(pts, img_shape=(3000, 4000)):
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords

def draw_canonical_polygon(img: np.ndarray, coords: np.ndarray, color: tuple):
    pts = canonical_to_pixel_coords(coords, img.shape[:2])
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, pts, color=color)
    return img

def get_mirror_canonical_polygon():
    # Custom mirror mask for the indy gripper. Hand-marked on 0619 demo (4000x3000),
    # remapped to the 16:9 1920x1080 camera setting (2026-07-09) via per-axis
    # focal ratio about the principal points (same lens, 16:9 = vertical crop):
    #   new = (old - c_old) * (f_new / f_old) + c_new
    # Mirrors sit at the bottom-left/right corners; left box is mirrored for the right.
    left_pts = [
        [2, 630],
        [332, 629],
        [332, 1284],
        [2, 1284]
    ]
    resolution = [1080, 1920]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_mirror_crop_slices(img_shape=(1080,1920), left=True):
    # Remapped 4000x3000 -> 1920x1080 (2026-07-09), same transform as
    # get_mirror_canonical_polygon.
    left_pts = [
        [214, 559],
        [286, 351]
    ]
    resolution = [1080, 1920]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    if not left:
        left_coords[:,0] *= -1
    left_pts = canonical_to_pixel_coords(left_coords, img_shape=img_shape)
    left_pts = np.round(left_pts).astype(np.int32)
    slices = (
        slice(np.min(left_pts[:,1]), np.max(left_pts[:,1])), 
        slice(np.min(left_pts[:,0]), np.max(left_pts[:,0]))
    )
    return slices


def get_gripper_canonical_polygon():
    # Custom gripper-body (jaw) mask for the indy gripper, used in the policy
    # observation (stage 07 / inference). Hand-marked "W" on 0619 demo (4000x3000):
    # two finger humps + bottom band, with a center V-notch so the grasped film
    # stays visible. Single concave polygon (not left/right symmetric).
    # Remapped 4000x3000 -> 1920x1080 (2026-07-09), same transform as
    # get_mirror_canonical_polygon. Humps raised 30px and V-notch inner points
    # pulled 40px toward center (user request, 0709 session check).
    body_pts = [
        [2, 1254],
        [1929, 1284],
        [1525, 843],
        [1207, 895],
        [1000, 1189],
        [720, 894],
        [335, 853]
    ]
    resolution = [1080, 1920]
    body_coords = pixel_coords_to_canonical(body_pts, resolution)
    coords = body_coords[None, ...]   # (1, N, 2): single polygon
    return coords

def get_finger_canonical_polygon(height=0.308, top_width=0.334, bottom_width=1.289):
    # Defaults remapped for the 16:9 1920x1080 setting (2026-07-09): the old 4:3
    # trapezoid (0.37 / 0.25 / 1.4 at 4000x3000) mapped through the per-axis focal
    # ratio lands at these height-normalized values on the new frame.
    # image size
    resolution = [1080, 1920]
    img_h, img_w = resolution

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords

def draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())
        
    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img

def get_gripper_with_finger_mask(img, height=0.37, top_width=0.25, bottom_width=1.4, color=(0,0,0)):
    # image size
    img_h = img.shape[0]
    img_w = img.shape[1]

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = np.array([[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]], dtype=np.int32)

    img = cv2.fillPoly(img, points, color=color, lineType=cv2.LINE_AA)
    return img

def inpaint_tag(img, corners, tag_scale=1.4, n_samples=16):
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center
    
    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0,1,2,3,4], list(scaled_corners) + [scaled_corners[0]], 
        axis=0)(np.linspace(0,4,n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:,1], 0, img.shape[0]-1), 
        np.clip(sample_points[:,0], 0, img.shape[1]-1)
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)
    
    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None,...].astype(np.int32), 
        color=median_color.tolist())
    return img

# =========== other utils ====================
def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform
