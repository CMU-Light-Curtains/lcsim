#!/usr/bin/env python
import glob
import os
import sys
import time
import random
import time
import numpy as np
import cv2
import sys
import os
import random
import math
import json
import rospy
from threading import Thread, Lock
from cv_bridge import CvBridge, CvBridgeError

fpath = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(fpath + "../build")
import pylc_lib as pylc

import pickle
import numpy as np
np.set_printoptions(suppress=True)
import tf
from numpy.linalg import inv
from geometry_msgs.msg import TransformStamped
import tf2_msgs.msg
import sensor_msgs.point_cloud2 as pc2



"""
We need to write a code whose inputs is similar to that of our ROS code,
and generate a JSON to match current one exactly
"""

"""
We can load from the json in config and start from there
sensors_multi_bike.json

"""

def convert(points):
    size = len(points)
    np_pts = np.ones((size, 4)).astype(np.float32)
    for i in range(0, size):
        ros_point = points[i]
        np_pts[i, 0] = ros_point.x
        np_pts[i, 1] = ros_point.y
        np_pts[i, 2] = ros_point.z
    return np_pts

def get_arc(r):
    #arc_curtain = CurtainMaker::makeCircleCurtain(Point2D(0,0),arc_radius*2,-M_PI_2,M_PI_2);
    arc = pylc.makeCircleCurtain(r*2, -np.pi/2., np.pi/2)
    arc[:,1] = 0.
    arc[:,3] = 1.
    print(arc.shape)
    return arc

def get_flat(z):
    points = []
    for x in np.arange(-10., 10., 0.01):
        points.append([x, 0, z, 1])
    return np.array(points).astype(np.float32)

CONFIG_FILE = fpath + "../config/" + "sensors_real.json"
SENSOR_SETUP = [{"C": "camera01", "L": "laser01"}]
SIZE = [512,512]
COUNT = 1

#CONFIG_FILE = pkg_path + "/config/" + "sensors_multi_bike.json"
#CONFIG_FILE = pkg_path + "/config/" + "sensors_gopro_out.json"
#CONFIG_FILE = pkg_path + "/config/" + "sensors_gopro_in.json"
#CONFIG_FILE = pkg_path + "/config/" + "sensors_gopro_out_fb.json"
#SENSOR_SETUP = '[{"C": "camera01", "L": "laser01"}, {"C": "camera02", "L": "laser01"}, {"C": "camera03", "L": "laser02"}, {"C": "camera04", "L": "laser02"}]'

#CONFIG_FILE = pkg_path + "/config/" + "sensors_gopro_indv_fb.json"
#SENSOR_SETUP = '[{"C": "camera01", "L": "laser01"}, {"C": "camera02", "L": "laser02"}, {"C": "camera03", "L": "laser03"}, {"C": "camera04", "L": "laser04"}]'

# CONFIG_FILE = pkg_path + "/config/" + "sensors_gopro_indv_fb2.json"
# SENSOR_SETUP = '[{"C": "camera01", "L": "laser01"}, {"C": "camera011", "L": "laser01"}, {"C": "camera02", "L": "laser02"}, {"C": "camera022", "L": "laser02"}, {"C": "camera03", "L": "laser03"}, {"C": "camera033", "L": "laser03"}, {"C": "camera04", "L": "laser04"}, {"C": "camera044", "L": "laser04"}]'
# SIZE = [1920,1080]
# COUNT = 8

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

tkill = False
class CurtainOpt(object):

    def get_arc(self, r):
        #arc_curtain = CurtainMaker::makeCircleCurtain(Point2D(0,0),arc_radius*2,-M_PI_2,M_PI_2);
        arc = pylc.makeCircleCurtain(r*2, -np.pi/2., np.pi/2)
        arc[:,1] = 0.
        arc[:,3] = 1.
        return arc

    def __init__(self, rosmode, oflipped=False):
        self.rosmode = rosmode
        self.oflipped = oflipped
        self.datum_processor = pylc.DatumProcessor()
        self.bridge = CvBridge()
        self.config_file = None
        self.pset = False

    def load_data_fromfile(self, config_file, sensor_setup):
        # Load config file
        self.config_file = config_file
        with open(config_file) as handle:
            json_sensors = json.loads(handle.read())

        self.load_data(json_sensors, sensor_setup)
        self.pset = True

    def load_data(self, json_sensors, sensor_setup):

        # Load json sensors
        self.json_sensors = json_sensors

        # Load sensor setup
        self.soi = sensor_setup

        # Get Names
        self.c_names = []
        self.l_names = []
        self.tfarray = []
        for obj in self.soi:
            self.c_names.append(obj["C"])
            self.l_names.append(obj["L"])

        # Setup
        self.setup_lc(self.json_sensors["lc"])

        # Run TF Thread
        if self.rosmode:
            global tkill
            tkill = True
            time.sleep(0.5)
            tkill = False
            self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
            self.tf_thread = Thread(target=self.tf_thread_func)
            self.tf_thread.daemon = True
            self.tf_thread.start()

        self.counter = 5.
        self.pset = True

    def get_basic(self, baseline=0.2, laser_fov=80, intrinsics=[400., 0., 256, 0., 400., 256., 0., 0., 1.], width=512, height=512, distortion=[0.000000, 0.000000, 0.000000, 0.000000, 0.000000]):
        params = {
            "sensors": [
            ],
            "lc": [
                {
                    "type": "laser",
                    "id": "laser01",
                    "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": -0.0,
                    "fov": laser_fov,
                    "attach": "hero"
                },
                {
                    "type": "camera",
                    "id": "camera01",
                    "x": -baseline, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": -0.0,
                    "width": width,
                    "height": height,
                    "fov": -1,
                    "intrinsics": intrinsics,
                    "distortion": distortion,
                    "limit": 1.0,
                    "attach": "hero",
                    "rgb": True
                }
            ]
        }
        sensor_setup = [{"C": "camera01", "L": "laser01"}]
        return params, sensor_setup

    def check_for_update(self):
        if self.config_file is None: return
        # Load config file
        with open(self.config_file) as handle:
            json_sensors = json.loads(handle.read())

        if json_sensors == self.json_sensors:
            pass
        else:
            print("Update")
            self.json_sensors = json_sensors
            self.setup_lc(self.json_sensors["lc"])

    def tf_thread_func(self):
        global tkill
        while 1:
            time.sleep(0.01)
            for tf in self.tfarray:
                tf.header.stamp = rospy.Time.now()
                tfm = tf2_msgs.msg.TFMessage([tf])
                self.pub_tf.publish(tfm)
            if tkill: break

    def update_sensors(self):
        self.setup_lc(self.json_sensors["lc"])

    def set_camera_fov(self, fov):
        for sensor in self.json_sensors["lc"]:
            if sensor["id"] not in self.c_names: continue
            sensor["fov"] = fov

    def compute_transform(self, x, y, z, roll, pitch, yaw):
        transform = np.eye(4)
        transform = tf.transformations.euler_matrix(yaw*np.pi/180.,pitch*np.pi/180.,roll*np.pi/180.)
        transform[0,3] = x
        transform[1,3] = y
        transform[2,3] = z
        return np.matrix(transform)

    def lc_spec_to_tf(self, lc_spec, base_frame, child_frame):
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = base_frame
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.child_frame_id = child_frame
        transform =   self.compute_transform(lc_spec["z"], -lc_spec["x"], -lc_spec["y"], 0, 0, 0) * self.compute_transform(0, 0, 0, -90 + lc_spec["yaw"], lc_spec["roll"], -90 + lc_spec["pitch"])
        quat = tf.transformations.quaternion_from_matrix(transform)
        tf_msg.transform.translation.x = transform[0,3]
        tf_msg.transform.translation.y = transform[1,3]
        tf_msg.transform.translation.z = transform[2,3]
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        return tf_msg

    def setup_lc(self, sensors):
        self.camera_data = dict()
        self.camera_names = []
        self.laser_data = dict()
        self.tfarray = []

        for sensor in sensors:
            if sensor["type"] == "camera":
                # Check if valid
                if sensor["id"] not in self.c_names: continue
                self.camera_names.append(sensor["id"])

                # Attachment?
                attachment = sensor["attach"]
                if len(attachment):
                    attachment = "/" + attachment

                # Matrix
                if "intrinsics" in sensor.keys():
                    matrix = np.array(sensor["intrinsics"]).reshape((3,3)).astype(np.float32)
                else:
                    cx = sensor["width"] / 2.0
                    cy = sensor["height"] / 2.0
                    fx = sensor["width"] / (
                            2.0 * math.tan(float(sensor["fov"]) * math.pi / 360.0))
                    fy = fx
                    matrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1.]).reshape((3,3))

                # Distortion
                distortion = np.array(sensor["distortion"]).reshape(1,5).astype(np.float32)

                # Transform
                transform = np.eye(4)
                transform = tf.transformations.euler_matrix(sensor["yaw"]*np.pi/180.,sensor["pitch"]*np.pi/180.,sensor["roll"]*np.pi/180.)
                transform[0,3] = sensor["x"]
                transform[1,3] = sensor["y"]
                transform[2,3] = sensor["z"]

                # W Transform
                wTc = np.eye(4)
                wTc = self.compute_transform(sensor["z"], -sensor["x"], -sensor["y"], 0, 0, 0) * self.compute_transform(0, 0, 0, -90 + sensor["yaw"], sensor["roll"], -90 + sensor["pitch"])
                if self.oflipped: wTc = self.compute_transform(sensor["x"], sensor["y"], sensor["z"], 0, 0, 0) * self.compute_transform(0, 0, 0, sensor["roll"], sensor["pitch"], sensor["yaw"])

                # Set
                self.camera_data[sensor["id"]] = {"attach":sensor["attach"], "matrix":matrix, "transform":transform, "wTc":wTc, "fov": sensor["fov"], "distortion": distortion, "limit": sensor["limit"], "width": sensor["width"], "height": sensor["height"], "name": sensor["id"]}

                # Publish TF?
                if self.rosmode:
                    self.tfarray.append(self.lc_spec_to_tf(sensor, sensor["attach"], sensor["attach"] + "/camera/depth/"+sensor["id"]))

            elif sensor["type"] == "laser":
                # Check if valid
                if sensor["id"] not in self.l_names: continue

                # W Transform
                wTl = np.eye(4)
                wTl = self.compute_transform(sensor["z"], -sensor["x"], -sensor["y"], 0, 0, 0) * self.compute_transform(0, 0, 0, -90 + sensor["yaw"], sensor["roll"], -90 + sensor["pitch"])
                if self.oflipped: wTl = self.compute_transform(sensor["x"], sensor["y"], sensor["z"], 0, 0, 0) * self.compute_transform(0, 0, 0, sensor["roll"], sensor["pitch"], sensor["yaw"])

                self.laser_data[sensor["id"]] = {"fov": sensor["fov"], "name": sensor["id"], "wTl":wTl}

                # Publish TF?
                if self.rosmode:
                    self.tfarray.append(self.lc_spec_to_tf(sensor, sensor["attach"], sensor["attach"] + "/laser/"+sensor["id"]))

        # Iterate each camera to compute laser transform
        for camera_name, data in self.camera_data.iteritems():
            wTc = data["wTc"]
            cTw = inv(wTc)

            # Cam to World
            data["cam_to_world"] = wTc.astype(np.float32)

            # Iterate each laser
            data["cam_to_laser"] = dict()
            for sensor in sensors:
                if sensor["type"] == "laser":
                    laser_name = sensor["id"]
                    wTl = self.laser_data[laser_name]["wTl"]
                    lTw = inv(wTl)

                    # Get Transform
                    lTc = inv(np.dot(lTw, wTc))
                    data["cam_to_laser"][laser_name] = lTc.astype(np.float32)

                    # print(camera_name)
                    # print(laser_name)
                    # print(lTc)

        # L Datums
        self.l_datums = []
        for laser_name, data in self.laser_data.iteritems():
            datum = pylc.Datum()
            datum.type = "laser"
            datum.laser_name = laser_name
            datum.fov = data["fov"]
            self.l_datums.append(datum)

        # C Datum
        self.c_datums = []
        for i in range(0, len(self.camera_names)):
            datum = pylc.Datum()
            datum.type = "camera"
            datum.camera_name = self.camera_names[i]
            datum.rgb_matrix = self.camera_data[datum.camera_name]["matrix"].astype(np.float32)
            datum.limit = self.camera_data[datum.camera_name]["limit"]
            datum.depth_matrix = self.camera_data[datum.camera_name]["matrix"].astype(np.float32)
            datum.cam_to_world = self.camera_data[datum.camera_name]["cam_to_world"]
            datum.cam_to_laser = self.camera_data[datum.camera_name]["cam_to_laser"]
            datum.fov = self.camera_data[datum.camera_name]["fov"]
            datum.distortion = self.camera_data[datum.camera_name]["distortion"]
            datum.imgh = self.camera_data[datum.camera_name]["height"]
            datum.imgw = self.camera_data[datum.camera_name]["width"]
            self.c_datums.append(datum)

        # Set
        self.datum_processor.setSensors(self.c_datums, self.l_datums)

        # Save
        # with open('laser_data.pickle', 'wb') as handle:
        #     pickle.dump(self.laser_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('camera_data.pickle', 'wb') as handle:
        #     pickle.dump(self.camera_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        all_data = [self.laser_data, self.camera_data]
        with open('sensor_real.json', 'w') as f:
            json.dump(all_data, f, cls=NumpyEncoder)

    def compute(self, pts_arr, depth_arr):
        self.counter += 1

        # C Datum and Inputs
        inputs = []
        frame_names = dict()
        input_names = dict()
        headers = dict()
        transforms = dict()
        for i in range(0, len(self.camera_names)):
            height = self.camera_data[self.camera_names[i]]["height"]
            width = self.camera_data[self.camera_names[i]]["width"]
            depth = depth_arr[i]
            if depth.shape[0] != height or depth.shape[1] != width:
                raise("Size mismatch")
            ros_depth_image = self.bridge.cv2_to_imgmsg(depth, "32FC1")

            input = pylc.Input()
            input.camera_name = self.camera_names[i]
            frame_names[self.camera_names[i]] = self.camera_names[i]
            #input.ros_depth_image = ros_depth_image
            input.depth_image = depth

            # Convert Points
            #pts = func(10)
            pts = pts_arr[i]
            temp_arr = []
            temp_arr.append(pts)
            input.design_pts_multi = temp_arr
            inputs.append(input)
            transforms[self.camera_names[i]] = self.camera_data[self.camera_names[i]]["cam_to_world"]
            input_names[self.camera_names[i]] = i

        # Process
        # ros_outputs = Outputs()
        pylc_output = pylc.Output()
        pylc.processPointsJoint(self.datum_processor, inputs, input_names, self.soi, pylc_output, True, True)

        # Images
        npimgs = []
        for i in range(len(pylc_output.images_multi)):
            lcimg = pylc_output.images_multi[i][0]
            npimgs.append(lcimg) # Adds 5ms

        if self.rosmode:
            fullcloud = pylc_output.full_cloud
            return fullcloud, npimgs
        else:
            fullcloud = pylc_output.full_cloud_eig
            return fullcloud, npimgs

if __name__ == "__main__":
    rosmode = True
    from sensor_msgs.msg import PointCloud2
    if rosmode: rospy.init_node('opt', anonymous=True)
    co = CurtainOpt(rosmode)
    #co.load_data(params, sensor_setup)
    co.load_data_fromfile(CONFIG_FILE, SENSOR_SETUP)

    if rosmode: publisher = rospy.Publisher('/opt/cloud', PointCloud2, queue_size=1)
    i=1
    dist = 4
    while 1:
        i+=0.1
        dist = dist + 0.01
        time.sleep(0.01)
        start = time.time()
        if not rosmode: continue

        # Try something
        #co.set_camera_fov(i)
        #co.update_sensors()
        co.check_for_update()

        # Compute Cloud
        pts = get_flat(dist)
        depth = np.ones((SIZE[1],SIZE[0])).astype(np.float32)*5
        allpts = []
        alldepths = []
        for i in range(0, COUNT):
            allpts.append(pts)
            alldepths.append(depth)

        cloud, npimgs = co.compute(allpts, alldepths)
        print(time.time() - start)

        cloud.header.frame_id = "hero"
        cloud.header.stamp = rospy.Time.now()

        publisher.publish(cloud)


    rospy.spin()

