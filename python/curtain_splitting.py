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

fpath = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(fpath + "../build")
import pylc_lib as pylc

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from helper import *

with open('sensor_real.json') as f:
    data = json.load(f)

laser_obj = data[0]["laser01"]
l_datums = []
ldatum = pylc.Datum()
ldatum.type = "laser"
ldatum.laser_name = laser_obj["name"]
ldatum.fov = 200
l_datums.append(ldatum)

camera_obj = data[1]["camera01"]
c_datums = []
cdatum = pylc.Datum()
cdatum.type = "camera"
cdatum.camera_name = camera_obj["name"]
cdatum.rgb_matrix = np.asarray(camera_obj["matrix"]).astype(np.float32)
cdatum.limit = camera_obj["limit"]
cdatum.depth_matrix = np.asarray(camera_obj["matrix"]).astype(np.float32)
cdatum.cam_to_world = np.asarray(camera_obj["cam_to_world"]).astype(np.float32)
cam_to_laser = np.asarray(camera_obj["cam_to_laser"]["laser01"]).astype(np.float32)
cam_to_laser_dict = {"laser01": cam_to_laser}
cdatum.cam_to_laser = cam_to_laser_dict
cdatum.fov = camera_obj["fov"]
cdatum.distortion = np.asarray(camera_obj["distortion"]).astype(np.float32)
cdatum.imgh = camera_obj["height"]
cdatum.imgw = camera_obj["width"]
c_datums.append(cdatum)

# Create Pylc object
datum_processor = pylc.DatumProcessor()
datum_processor.setSensors(c_datums, l_datums)
fitting_module = pylc.Fitting(datum_processor)

class Manager():
    def __init__(self):
        # Initial Targets
        self.targets = np.array([
        # [0.53629035,  1.8993506 ,  3.    ,      0.        ],
        # [-1.9193548 ,  6.038961  ,  3.   ,       0.        ],
        # [-0.14112903,  7.689394  ,  3.   ,       0.        ],
        # [ 2.9637096,   4.5508657  ,  3.   ,       1.        ]

        # [-2.1169355 ,  4.848485 ,   3.  ,        0.        ],
        # [-1.9193548  , 6.038961 ,   3.  ,        0.        ],
        # [-0.14112903 , 7.689394 ,   3.  ,        0.        ],
        # [ 2.6814516  , 5.146104 ,   3.  ,        1.        ]

        # [-4.91129 ,   7.472944 ,  3.   ,      0.       ],
        # [-2.1451614 , 7.9329004 , 3.   ,      0.       ],
        # [ 1.8629032 , 7.4188313,  3.  ,       0.       ],
        # [-1.9193548 , 2.7651515,  3.  ,       1.       ]

        # # Complex Curve
        # [-1.60887097 , 4.06385281,  3.    ,      0.        ],
        # [-3.44354839 , 7.06709957 , 3.    ,      0.        ],
        # [-0.14112903 , 7.689394  ,  3.    ,      0.        ],
        # [ 0.        ,  5.63311688 , 3.   ,       1.        ],
        # [-0.02822581 , 3.2521645 ,  4.   ,       1.        ],
        # [ 1.60887097 , 2.49458874 , 4.   ,       1.        ]

        # # Z
        # [-2.314516  ,  7.851732  ,  3.   ,       0.        ],
        # [ 0.33870968 , 7.71645   ,  6.   ,       0.        ],
        # [ 0.22580644 , 5.7954545 ,  5.   ,       1.        ],
        # [-2.3709676 ,  5.524892  ,  5.5   ,      1.        ],
        # [-2.032258  ,  3.6580086 , 10.5   ,      1.        ],
        # [ 1.2701613 ,  3.6309524 ,  4.      ,    1.        ]

        # # Crash
        # [ 0.33870968 , 7.71645  ,   6.     ,     0.        ],
        # [-0.02822581 , 5.90368  ,   5.    ,      1.        ],
        # [-2.3709676  , 5.524892  ,  5.5   ,      1.        ],
        # [-2.032258   , 3.6580086 , 10.5   ,      1.        ],
        # [ 1.2701613 ,  3.6309524 ,  4.         , 1.        ]

        [-2.85080645 , 7.60822511 , 3.  ,        0.        ],
        [-0.14112903 , 7.689394  ,  3.  ,        0.        ],
        [ 1.60887098 , 6.66125536 , 4.  ,        1.        ],
        [ 3.24596763 , 8.06818199 , 4.  ,        1.        ],
        [ 0.59274194 , 9.36688312 , 4.  ,        1.        ],
        [-2.39919355,  9.36688312 , 4. ,         1.        ],
        [-1.77822581 , 6.33658009 , 4.  ,        1.        ],
        [ 0.70564516 , 5.76839827,  4.  ,        1.        ],
        [ 2.22983871 , 5.03787879 , 4.  ,        1.        ],
        [-0.39516129 , 3.82034632 , 4.  ,        1.        ],
        [-2.68145161 , 5.1461039,   4.    ,      1.        ]

        # [-2.85080647 , 7.60822535 , 3.      ,    0.        ],
        # [-0.14112903 , 7.689394   , 3.      ,    0.        ],
        # [ 1.60887098 , 6.66125536 , 4.      ,    1.        ],
        # [ 3.24596763 , 8.06818199 , 4.      ,    1.        ],
        # [ 0.59274197 , 9.36688328 , 4.      ,    1.        ],
        # [-2.39919353 , 9.36688328 , 4.      ,    1.        ],
        # [-1.77822578 , 6.33658028,  4.      ,    1.        ],
        # [ 0.70564514 , 5.76839828,  4.      ,    1.        ],
        # [ 1.69354839 , 4.82142857,  4.      ,    1.        ],
        # [-0.3951613  , 3.82034636,  4.      ,    1.        ],
        # [-2.68145156 , 5.14610386 , 4.      ,    1.        ],
        # [-3.81048387 , 8.06818182 , 4.      ,    1.        ],
        # [-0.53629032 , 8.69047619,  4.       ,   1.        ],
        # [ 3.10483871 , 6.87770563,  4.       ,   1.        ],
        # [ 0.50806452 , 3.00865801 , 4.       ,   1.        ],
        # [-1.89112903,  3.11688312 , 4.       ,   1.        ]

        # # Weird Result
        # [-1.35483871 , 2.57575758 , 4.   ,       1.        ],
        # [ 1.27016129 , 6.68831169 , 4.  ,        1.        ],
        # [-1.83467742 , 7.95995671,  4.  ,        1.        ],
        # [-1.35483871 , 4.57792208 , 4.  ,        1.        ],
        # [ 1.94758065 , 3.06277056 , 4.   ,       1.        ],

        # # Weird Result
        # [-1.2701613,  2.5757575,  4. ,        1.       ],
        # [ 1.2701613 , 6.6883116,  4. ,        1.       ],
        # [-1.8346775 , 7.9599566 , 4. ,        1.       ],
        # [-1.3548387 , 4.577922 ,  4. ,        1.       ],
        # [ 1.9475807 , 3.0627706 , 4.  ,       1.       ]

        ]).astype(np.float32)

        self.targetpts = np.array([
            [ 2.4274194 ,  5.8225107 ,  4.   ,       1.        ],
            [-1.2701613 ,  7.149351 ,   4.  ,        1.        ]
        ])

        self.clusters = []

        # Display
        #plt.figure(2)
        #plt.cla()
        plt.figure(1)
        plt.cla()

        # Mode
        self.OPT = False
        self.CONTROL = "B"

        # Click
        if self.CONTROL == "A":
            self.click = Click(plt.gca(), self.targets)
        elif self.CONTROL == "B":
            self.click = Click(plt.gca(), self.targetpts)

        # Keyboard
        self.kb = plt.gca().figure.canvas.mpl_connect('key_press_event',self.press)

        pass

    def press(self, event):
        if event.key == " ":
            self.OPT = not self.OPT
            print("Press")
        elif event.key == "control":
            if(self.CONTROL == "B"):
                self.CONTROL = "A"
                self.click = Click(plt.gca(), self.targets)
            elif(self.CONTROL == "A"):
                self.CONTROL = "B"
                self.click = Click(plt.gca(), self.targetpts)

    def loop(self):
        global datum_processor

        if self.OPT == False:
            if self.CONTROL == "A":
                self.targets = self.click.cv
            elif self.CONTROL == "B":
                self.targetpts = self.click.cv

        # Draw all targets
        plt.figure(1)
        squaresize = 0.3
        for i in range(0, self.targets.shape[0]):
            X = self.targets[i,0]
            Y = self.targets[i,1]
            rect = patches.Rectangle((X - squaresize/2.,Y - squaresize/2.),squaresize,squaresize,linewidth=1,edgecolor='r',facecolor='none')
            plt.gca().add_patch(rect)
        for i in range(0, self.targetpts.shape[0]):
            X = self.targetpts[i,0]
            Y = self.targetpts[i,1]
            ss = 0.15
            rect = patches.Rectangle((X - ss/2.,Y - ss/2.),ss,ss,linewidth=1,edgecolor='g',facecolor='green')
            plt.gca().add_patch(rect)

        ####
        # WE NEED TO ELIMATE THOSE TARGET PTS THAT CANNOT BE SEEN BY CAMERA?

        # Pass Targets into function
        paths = [self.targets]
        splineParams = pylc.SplineParamsVec()
        start_time = time.time()
        output = []
        spline = pylc.fitBSpline(self.targets, splineParams, True)
        plt.scatter(spline[:,0], spline[:,1], s=0.01, c='r')
        projPoints = pylc.solveT(splineParams, self.targetpts)

        # Test Error
        #self.computeError(self.targets, self.targetpts, True)

        # Draw Outputs
        for i in range(0, projPoints.shape[0]):
            X = projPoints[i,0]
            Y = projPoints[i,1]
            rect = patches.Rectangle((X - squaresize/2.,Y - squaresize/2.),squaresize,squaresize,linewidth=1,edgecolor='b',facecolor='none')
            plt.gca().add_patch(rect)

            Xp = self.targetpts[i,0]
            Yp = self.targetpts[i,1]
            l = mlines.Line2D([X,Xp], [Y,Yp])
            plt.gca().add_line(l)

        if self.OPT:
            print("PRESSED")
            self.clusters = fitting_module.curtainSplitting(spline, "camera01", "laser01")
            self.OPT = False

        #self.targets = curveParamsNew
        #print(curveParamsNew)

        offset = 0
        colors = ['b', 'c', 'r', 'g', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
        for idx, cluster in enumerate(self.clusters):
            center = cluster[cluster.shape[0]/2, :] + offset
            plt.text(center[0],center[1],str(idx))
            plt.scatter(cluster[:,0]+offset, cluster[:,1]+offset, s=0.5, color=colors[idx])



            # # HACK
            # angles = datum_processor.splineToAngles(cluster, "camera01", "laser01", True)
            # dpoints = angles.design_pts.T
            # plt.scatter(dpoints[:,0], dpoints[:,2]+0.3, s=1.0, c='g')

        # # Plot
        # print("-")
        # for i in range(self.targets.shape[0]):
        #     plt.text(self.targets[i,0],self.targets[i,1],str(i))

        plt.figure(1)
        plt.minorticks_on()
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-7, 7)
        plt.ylim(0, 10)
        plt.pause(0.001)
        plt.cla()

manager = Manager()
while 1:
    manager.loop()