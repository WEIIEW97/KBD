# constants
SUBFIX = "DEPTH/raw"
CAMERA_TYPE = "N09ASH24DH0057"
BASEDIR = f"data/{CAMERA_TYPE}/image_data"
H = 480
W = 640
EPSILON = 1e-6
UINT16_MIN = 0
UINT16_MAX = 65535

ANCHOR_POINT = [H // 2, W // 2]

AVG_DIST_NAME = "avg_depth_50x50_anchor"
AVG_DISP_NAME = "avg_disp_50x50_anchor"
GT_DISP_NAME = "actual_disp"
MEDIAN_DIST_NAME = "median_depth_50x50_anchor"
MEDIAN_DISP_NAME = "median_disp_50x50_anchor"
GT_DIST_NAME = "actual_depth"
GT_ERROR_NAME = "absolute_error"
GT_DISP_ERROR_NAME = "absolute_disp_error"
FOCAL_NAME = "focal"
BASLINE_NAME = "baseline"

OUT_PARAMS_FILE_NAME = "KBD_model_fitted_params.json"
LINEAR_OUT_PARAMS_FILE_NAME = "linear_" + OUT_PARAMS_FILE_NAME
OUT_FIG_COMP_FILE_NAME = "compare.jpg"
OUT_FIG_RESIDUAL_FILE_NAME = "fitted_residual.jpg"
OUT_FIG_ERROR_RATE_FILE_NAME = "error_rate.jpg"
OUT_FIG_GLOBAL_PREDICTION_FILE_NAME = "global_KBD_prediction.jpg"
OUT_FIG_LINEAR_PREDICTION_FILE_NAME = "piecewise_prediction.jpg"
MAPPED_COLUMN_NAMES = ["actual_depth", "focal", "baseline", "absolute_error"]
MAPPED_PAIR_DICT = {
    "距离(mm)": "actual_depth",
    "相机焦距": "focal",
    "相机基线": "baseline",
    "绝对误差/mm": "absolute_error",
}

MAPPED_PAIR_DICT_DEBUG = {
    "距离(mm)": "actual_depth",
    "相机焦距": "focal",
    "相机基线": "baseline",
    "绝对误差/mm": "absolute_error",
    "fit plane dist/mm": "fit_plane",
}
