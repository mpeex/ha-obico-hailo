DOMAIN = "obico_ml_hailo"
PLATFORMS = ["binary_sensor", "switch", "camera", "sensor"]
DEFAULT_NAME = "Obico ML (Hailo)"
DEFAULT_URL = "http://obico_ml_hailo:3333/detect/"
DEFAULT_INTERVAL = 10  # in seconds
DEFAULT_THRESHOLD = 0.2

CONF_URL = "url"
CONF_INTERVAL = "interval"
CONF_CAMERA_ENTITY = "camera_entity"
CONF_THRESHOLD = "threshold"

ATTR_ERROR_DETECTED = "error_detected"
ATTR_IMAGE_WITH_ERRORS = "image_with_errors"
