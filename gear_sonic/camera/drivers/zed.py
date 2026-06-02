"""Stereolabs ZED RGB driver for Gear Sonic data collection."""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

try:
    import pyzed.sl as sl
except ImportError:
    sl = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class ZEDConfig:
    """Configuration for a single RGB view from a ZED camera."""

    image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    resolution: str = "HD720"
    view: str = "right"


class ZEDSensor(Sensor, SensorServer):
    """Publishes one ZED eye as an RGB image under the requested mount key."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: ZEDConfig = ZEDConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        if sl is None:
            raise RuntimeError("pyzed.sl is not installed or not visible in this environment")

        self._zed_config = config
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        self.camera = sl.Camera()
        self.image = sl.Mat()

        init_params = sl.InitParameters()
        init_params.camera_resolution = self._resolve_resolution(config.resolution)
        init_params.camera_fps = config.fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        self.runtime_params = sl.RuntimeParameters()
        self.view = self._resolve_view(config.view)

        if self._run_as_server:
            self.start_server(port)

        print(
            f"Done initializing ZED sensor: view={config.view}, "
            f"output={config.image_dim[0]}x{config.image_dim[1]}@{config.fps}"
        )

    @staticmethod
    def _resolve_resolution(resolution: str):
        mapping = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        key = resolution.upper()
        if key not in mapping:
            raise ValueError(f"Unsupported ZED resolution: {resolution}")
        return mapping[key]

    @staticmethod
    def _resolve_view(view: str):
        key = view.lower()
        if key == "left":
            return sl.VIEW.LEFT
        if key == "right":
            return sl.VIEW.RIGHT
        raise ValueError(f"Unsupported ZED view: {view}")

    def read(self) -> dict[str, Any] | None:
        if self.camera.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None

        self.camera.retrieve_image(self.image, self.view)
        image = self.image.get_data()
        if image is None or image.size == 0:
            return None

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = image[..., :3]

        width, height = self._zed_config.image_dim
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        image = np.ascontiguousarray(image, dtype=np.uint8)
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: image},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        width, height = self._zed_config.image_dim
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.camera.close()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
