# top mode parsing
mode = 'N9'

class Config:
    def __init__(self, mode="N9", cy=0, cx=0):
        self.SUBFIX = "DEPTH/raw"
        self.FOCAL_MULTIPLIER = 1.6
        self.H = 480
        self.W = 640
        self.EPSILON = 1e-6
        self.UINT16_MIN = 0
        self.UINT16_MAX = 65535
        self.DISP_VAL_MAX_UINT16 = 32767
        self.EVAL_WARNING_RATE = 0.5

        self.ANCHOR_POINT = [self.H // 2, self.W // 2]
        self.TARGET_POINTS = [300, 500, 600, 1000, 1500, 2000]
        self.TARGET_THRESHOLDS = [0.02, 0.02, 0.02, 0.02, 0.04, 0.04]

        self.AVG_DIST_NAME = "avg_depth_50x50_anchor"
        self.AVG_DISP_NAME = "avg_disp_50x50_anchor"
        self.GT_DISP_NAME = "actual_disp"
        self.MEDIAN_DIST_NAME = "median_depth_50x50_anchor"
        self.MEDIAN_DISP_NAME = "median_disp_50x50_anchor"
        self.GT_DIST_NAME = "actual_depth"
        self.GT_ERROR_NAME = "absolute_error"
        self.GT_DISP_ERROR_NAME = "absolute_disp_error"
        self.GT_DIST_ERROR_NAME = "absolute_depth_error"
        self.KBD_ERROR_NAME = "absolute_kbd_error"
        self.FOCAL_NAME = "focal"
        self.BASLINE_NAME = "baseline"
        self.KBD_PRED_NAME = "kbd_pred"

        self.OUT_PARAMS_FILE_NAME = "KBD_model_fitted_params.json"
        self.LINEAR_OUT_PARAMS_FILE_NAME = "linear_" + self.OUT_PARAMS_FILE_NAME
        self.OUT_FIG_COMP_FILE_NAME = "compare.jpg"
        self.OUT_FIG_RESIDUAL_FILE_NAME = "fitted_residual.jpg"
        self.OUT_FIG_ERROR_RATE_FILE_NAME = "error_rate.jpg"
        self.OUT_FIG_GLOBAL_PREDICTION_FILE_NAME = "global_KBD_prediction.jpg"
        self.OUT_FIG_LINEAR_PREDICTION_FILE_NAME = "piecewise_prediction.jpg"
        self.MAPPED_COLUMN_NAMES = [
            "actual_depth",
            "focal",
            "baseline",
            "absolute_error",
        ]
        self.MAPPED_PAIR_DICT = {
            "DISTANCE(mm)": "actual_depth",
            "Camera_Focal": "focal",
            "Camera_Baseline": "baseline",
            "Absolute_error/mm": "absolute_error",
        }

        if mode == "N9":
            self.FOCAL_MULTIPLIER = 1.6
            self.H = 480
            self.W = 640
            self.TARGET_THRESHOLDS = [0.02, 0.02, 0.02, 0.02, 0.04, 0.04]
            self.NEAR_THR = 0.02
            self.FAR_THR = 0.04
        elif mode == "M1F":
            self.FOCAL_MULTIPLIER = 1
            self.H = 400
            self.W = 640
            self.TARGET_THRESHOLDS = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
            self.NEAR_THR = 0.02
            self.FAR_THR = 0.02
        else:
            raise ValueError(
                f"unsupported mode {mode}, please recheck the requirements."
            )
        
        if cy != 0 and cx != 0:
            self.ANCHOR_POINT = [cy, cx]

conf = Config(mode)

# constants
SUBFIX = conf.SUBFIX
# FOCAL_MULTIPLIER = conf.FOCAL_MULTIPLIER
FOCAL_MULTIPLIER = 1
H = conf.H
W = conf.W
EPSILON = conf.EPSILON
EVAL_WARNING_RATE = conf.EVAL_WARNING_RATE
DISP_VAL_MAX_UINT16 = conf.DISP_VAL_MAX_UINT16

UINT16_MIN = conf.UINT16_MIN
UINT16_MAX = conf.UINT16_MAX

# ANCHOR_POINT = conf.ANCHOR_POINT
ANCHOR_POINT = [205, 310]
TARGET_POINTS = conf.TARGET_POINTS
TARGET_THRESHOLDS = conf.TARGET_THRESHOLDS
NEAR_THR = conf.NEAR_THR
FAR_THR = conf.FAR_THR

AVG_DIST_NAME = conf.AVG_DIST_NAME
AVG_DISP_NAME = conf.AVG_DISP_NAME
GT_DISP_NAME = conf.GT_DISP_NAME
MEDIAN_DIST_NAME = conf.MEDIAN_DIST_NAME
MEDIAN_DISP_NAME = conf.MEDIAN_DISP_NAME
GT_DIST_NAME = conf.GT_DIST_NAME
GT_ERROR_NAME = conf.GT_ERROR_NAME
GT_DISP_ERROR_NAME = conf.GT_DISP_ERROR_NAME
GT_DIST_ERROR_NAME = conf.GT_DIST_ERROR_NAME
KBD_ERROR_NAME = conf.KBD_ERROR_NAME
FOCAL_NAME = conf.FOCAL_NAME
BASLINE_NAME = conf.BASLINE_NAME
KBD_PRED_NAME = conf.KBD_PRED_NAME

OUT_PARAMS_FILE_NAME = conf.OUT_PARAMS_FILE_NAME
LINEAR_OUT_PARAMS_FILE_NAME = conf.LINEAR_OUT_PARAMS_FILE_NAME
OUT_FIG_COMP_FILE_NAME = conf.OUT_FIG_COMP_FILE_NAME
OUT_FIG_RESIDUAL_FILE_NAME = conf.OUT_FIG_COMP_FILE_NAME
OUT_FIG_ERROR_RATE_FILE_NAME = conf.OUT_FIG_ERROR_RATE_FILE_NAME
OUT_FIG_GLOBAL_PREDICTION_FILE_NAME = conf.OUT_FIG_GLOBAL_PREDICTION_FILE_NAME
OUT_FIG_LINEAR_PREDICTION_FILE_NAME = conf.OUT_FIG_LINEAR_PREDICTION_FILE_NAME
MAPPED_COLUMN_NAMES = conf.MAPPED_COLUMN_NAMES
MAPPED_PAIR_DICT = conf.MAPPED_PAIR_DICT
