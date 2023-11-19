import cv2
import torchvision.transforms.functional as F

from feature_extraction_2d.SensorData_python3_port import SensorData


def color_images_from_sensor_data(sens: SensorData, image_size=None, frame_skip=1):
    """
    Returns a generator object for color images contained
    in the sensor data as torch tensors.
    """
    for f in range(0, len(sens.frames), frame_skip):
        color = sens.frames[f].decompress_color(sens.color_compression_type)
        if image_size is not None:
            color = cv2.resize(
                color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        yield F.to_tensor(color)


def color_images_from_sens_file(path, image_size=None, frame_skip=1):
    """
    Returns a generator object for color images contained
    in a sens file as torch tensors.
    """
    data = SensorData(path)
    yield from color_images_from_sensor_data(data, image_size, frame_skip)
