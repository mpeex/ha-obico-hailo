DOMAIN = "obico_ml_hailo"
PLATFORMS = ["binary_sensor", "switch", "camera", "sensor"]
DEFAULT_NAME = "Obico ML (Hailo)"
DEFAULT_URL = "http://172.30.32.1:3333"
DEFAULT_DETECTION_INTERVAL = 1  # in seconds
DEFAULT_THRESHOLD = 0.2
DEFAULT_AUTO_START = False

CONF_URL = "url"
CONF_DETECTION_INTERVAL = "detection_interval"
CONF_CAMERA_ENTITY = "camera_entity"
CONF_THRESHOLD = "threshold"
CONF_AUTO_START = "auto_start"

ATTR_ERROR_DETECTED = "error_detected"
ATTR_IMAGE_WITH_ERRORS = "image_with_errors"
