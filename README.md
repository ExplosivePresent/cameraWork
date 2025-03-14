Real-time Stereo vision algorithm using OpenCV2 stereoBM. 

`stereoTesting.py` was tested and tuned for the WSVD Stereo Video dataset, and `stereoCamera.py` was tested and tuned for the ELP GS800P-V83 USB Stereo camera.

# Stereo Vision to Point Cloud Script (stereoCam.py)

This Python script generates a 3D point cloud from stereo images using OpenCV. It computes disparity maps from left and right images and converts them into 3D coordinates, saving the result as a `.pcd` point cloud file.

## Dependencies

- `numpy`
- `opencv-python`

Install using:

```bash
pip install numpy opencv-python
```

# High-level Code Breakdown 

## Real-time Stereo to Point-Cloud

The script `stereoTesting.py` takes a frame of both cameras using a single `cv.VideoCapture()` object, then splits it into left and right grayscale frames. Those frames are upsampled to better find disparities, passed through `cv.StereoBM`, and then downsampled to speed up the point-cloud generation. 

See OpenCV documentation for `stereoBM` here: [https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html](url)

Ref: [https://github.com/Parag-IIT/PointCloud-Generation-from-Stereo-imageso/blob/main/StereoSGBM.py](url), [https://learnopencv.com/depth-perception-using-stereo-camera-python-c/](url)

## Real-time Stereo to Point-Cloud, with ROS2

This script requires additional dependencies, namely `ROS2` and `rclpy`

`pcd_publisher.py` does the same job as `stereoCam.py`, but repackaged to work as a ROS2 Node and publishing to the `points` topic as a `sensor_msgs.PointCloud2` message. This means changing from a `while True:` loop to doing the stereo & point-cloud algorithm within the publisher's `timer_callback()` Once those changes have restructured the previous script, the ROS2 Node is pretty typical.

Ref: [https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo](url)
