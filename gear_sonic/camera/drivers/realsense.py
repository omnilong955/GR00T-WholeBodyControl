"""Intel RealSense RGB camera driver.

Requires the ``pyrealsense2`` SDK — install with::

    pip install pyrealsense2

See https://github.com/IntelRealSense/librealsense for hardware-specific instructions.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyrealsense2 as rs

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense color stream."""

    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for a single Intel RealSense color camera."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = RealSenseConfig(),
        serial: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")

        for device in devices:
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(f"    Firmware version: {device.get_info(rs.camera_info.firmware_version)}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        if not serial:
            raise RuntimeError(
                "RealSense serial is required. Pass --left-wrist-device-id and "
                "--right-wrist-device-id for D405 cameras."
            )
        if serial not in serials:
            raise RuntimeError(
                f"RealSense serial {serial} not found. Available serials: {serials}"
            )
        self.config.enable_device(serial)

        try:
            self.config.enable_stream(
                rs.stream.color,
                config.color_image_dim[0],
                config.color_image_dim[1],
                rs.format.rgb8,
                config.fps,
            )
            self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        self._realsense_config = config
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        if self._run_as_server:
            self.start_server(port)
        print(f"Done initializing RealSense sensor: {serial}")

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(f"ERROR! Failed to wait for frames: {e}")
            return None

        color_frame = frames.get_color_frame()

        if not color_frame:
            print("WARNING! No color frame")
            return None

        try:
            color_image = np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"ERROR! Failed to convert color frame to numpy array: {e}")
            return None

        if color_image.size == 0:
            print("WARNING! Empty color image")
            return None

        width, height = self._realsense_config.color_image_dim
        if color_image.shape[1] != width or color_image.shape[0] != height:
            color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_AREA)

        color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
        current_time = time.time()
        timestamps = {self.mount_position: current_time}
        images = {self.mount_position: color_image}
        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self._realsense_config.color_image_dim[1],
                        self._realsense_config.color_image_dim[0],
                        3,
                    ),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
