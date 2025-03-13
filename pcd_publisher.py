import sys
import os

import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

import numpy as np
import cv2 as cv
from stereoCamera import stereo

class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')

        self.points = np.empty([19200,3])
        self.cap = cv.VideoCapture(0)          #create capture
        self.Q = np.float32([[1, 0, 0, 0],      #Q for point disparity to 3d points conversion later
                            [0, -1, 0, 0],
                            [0, 0, 1.9048 * 0.05, 0],  # Focal length multiplication obtained experimentally.
                            [0, 0, 0, 1]])
        # I create a publisher that publishes sensor_msgs.PointCloud2 to the
        # topic 'pcd'. The value '10' refers to the history_depth, which I
        # believe is related to the ROS1 concept of queue size.
        # Read more here:
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'points', 10)
        timer_period = 1 / 60.0
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def timer_callback(self):
        #intake frame, perform stereo, output pointcloud, publish pointcloud
        ret, frame = self.cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            raise ValueError("Frame not received")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Take the resolution/shape dimensions and find half width
        h, w = gray.shape
        half = w // 2

        # split into left and right halves
        left = gray[:, :half]  # shape is (240,320)
        right = gray[:, half:]

        # #upsampling
        for i in range(1):
            h, w = left.shape
            left = cv.pyrUp(left, dstsize=(w * 2, h * 2))
            h, w = right.shape
            right = cv.pyrUp(right, dstsize=(w * 2, h * 2))

        disparity = stereo.compute(left, right)

        ###Additional postprocessing
        disparity = disparity.astype(np.float32)
        # downsampling
        for i in range(2):
            h, w = left.shape
            left = cv.pyrDown(left, dstsize=(w // 2, h // 2))
            h, w = right.shape
            right = cv.pyrDown(right, dstsize=(w // 2, h // 2))
            h, w = disparity.shape
            disparity = cv.pyrDown(disparity, dstsize=(w // 2, h // 2))

        #####Point Cloud generation
        # Reproject points into 3D
        points_3D = cv.reprojectImageTo3D(disparity, self.Q)
        self.points = points_3D.reshape(-1,3)
        # # Get color points TODO: implement color point cloud?
        # colors = cv.cvtColor(left, cv.COLOR_BGR2RGB)
        # use the point_cloud() function to convert the numpy array
        # into a sensor_msgs.PointCloud2 object. The second argument is the
        # name of the frame the point cloud will be represented in. The default
        # (fixed) frame in RViz is called 'map'
        self.pcd = point_cloud(self.points, 'map')  #FIXME: points_3D out may not be 1:1 with point_cloud() in
        # Then I publish the PointCloud2 object
        self.pcd_publisher.publish(self.pcd)


def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message

    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

    """
    # In a PointCloud2 message, the point cloud is stored as an byte
    # array. In order to unpack it, we also include some parameters
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()

    # The fields specify what the bytes represents. The first 4 bytes
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [sensor_msgs.PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which
    # coordinate frame it is represented in.
    header = std_msgs.Header(frame_id=parent_frame)

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),  # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_publisher = PCDPublisher()
    rclpy.spin(pcd_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcd_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()