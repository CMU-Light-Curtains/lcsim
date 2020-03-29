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
            # [-2.1389081 ,  4.866832   , 3.       ,   0.        ],
            # [ 0.05645161 , 7.527056 ,   3.0702305 ,  0.        ],
            # [ 2.8790324  , 4.9296536 ,  3.     ,     1.        ]

            [-4.61654282 , 5.24932289 , 3.         , 1.        ],
            [-3.41206789 , 5.38161945 , 4.09122086 , 1.        ],
            [-3.45479202 , 6.33987856 , 3.77472115 , 1.        ],
            [-3.02016129 , 7.71645022 , 4.23183441 , 1.        ],
            [-1.91935484 , 9.04220779 , 4.24547625 , 1.        ],
            [-0.90322581 , 8.87987013 , 4.         , 1.        ],
            [-0.67741935 , 6.76948052 , 4.         , 1.        ],
            [-0.22580645 , 3.98268398 , 4.         , 1.        ],
            [ 0.875      , 3.95562771 , 4.         , 1.        ],
            [ 1.07258065 , 5.68722944 , 4.         , 1.        ],
            [ 2.28629032 , 7.25649351 , 4.         , 1.        ],
            [ 2.28629032 , 6.03896104 , 4.         , 1.        ],
            [ 3.18951613 , 5.79545455 , 4.         , 1.        ]
        ]).astype(np.float32)

        self.targetpts = np.array([
            # [ 1.86290323 , 6.14718615 , 4.   ,       1.        ],
            # [-0.93145161 , 6.87770563,  4.  ,        1.        ],
            # [-1.72177419 , 5.4978355 ,  4.  ,        1.        ],
            # [-1.46774194 , 6.17424242,  4.  ,        1.        ],
            # [-0.4233871  , 7.36471861,  4.  ,        1.        ],
            # [ 0.59274194 , 7.36471861,  4.  ,        1.        ],
            # [ 1.41129032 , 6.71536797 , 4.  ,        1.        ],
            # [ 2.39919355 , 5.47077922 , 4.    ,      1.        ]

            [-1.18548387 , 8.85281385 , 4.    ,      1.        ],
            [-3.89516129 , 5.33549784 , 4.    ,      1.        ],
            [-4.57258065 , 5.254329   , 4.    ,      1.        ],
            [-4.26209677 , 5.20021645 , 4.    ,      1.        ],
            [-3.55645161 , 5.30844156 , 4.    ,      1.        ],
            [-1.83467742 , 9.04220779 , 4.    ,      1.        ],
            [-1.52419355 , 8.98809524 , 4.    ,      1.        ],
            [-0.875      , 8.87987013 , 4.    ,      1.        ],
            [-0.14112903 , 3.95562771 , 4.    ,      1.        ],
            [ 0.25403226 , 3.92857143 , 4.    ,      1.        ],
            [ 0.53629032 , 3.92857143 , 4.    ,      1.        ],
            [ 0.81854839 , 3.90151515 , 4.    ,      1.        ],
            [ 2.25806452 , 7.28354978 , 4.    ,      1.        ],
            [ 2.22983871 , 6.9047619  , 4.     ,     1.        ],
            [ 3.10483871 , 5.76839827 , 4.     ,     1.        ],
            [ 2.68145161 , 5.79545455 , 4.     ,     1.        ],
            [ 2.28629032 , 6.49891775 , 4.     ,     1.        ],
            [-3.5        , 5.68722944 , 4.     ,     1.        ],
            [-3.52822581 , 6.06601732 , 4.     ,     1.        ],
            [ 2.34274194 , 6.03896104 , 4.     ,     1.        ]
        ])

        # Display
        plt.figure(2)
        plt.cla()
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

        # special
        self.SPEC = False

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
        elif event.key == "x":
            self.SPEC = not self.SPEC
            print("SPEC")

    def computeError(self, curveParams, targetPts, draw=False):
        splineParams = pylc.SplineParamsVec()
        spline = pylc.fitBSpline(curveParams, splineParams, True)
        projPoints = pylc.solveT(splineParams, targetPts)
        ranges = projPoints[:,2]
        error = np.sum(ranges, 0)
        #print(error)

        #spline = np.delete(spline, np.s_[300:500], axis = 0)
        #print("A")
        angles = fitting_module.splineToAngles(spline, "camera01", "laser01", True)

        dpoints = angles.design_pts.T
        accels = angles.accels
        peaks = angles.peaks
        veloc = angles.velocities
        summed_peak = angles.summed_peak
        # if draw:
        #     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
        #     if dpoints.shape[0] != 0:
        #         plt.scatter(dpoints[:,0], dpoints[:,2], s=1.0, c='g')
        #     plt.figure(2)
        #     plt.plot(accels)
        #     plt.figure(1)

        alpha = 0
        if self.SPEC:
            alpha = 0.0001
        else:
            alpha = 0

        c1 = alpha*np.sum(np.power(peaks,2))
        c2 = alpha*summed_peak
        c3 = alpha*np.sum(np.power(accels,2))



        #summed_peak = 0.001*np.sum(peaks)
        print(c1)
        terror = error + c1

        return terror

    def update(self, curveParams, targetPts):
        curveParamsNew = curveParams.copy()

        # Iterate each node
        for i in range(0, curveParams.shape[0]):
            #if i == 0 or i == curveParams.shape[0] - 1: continue

            # Iterate X,Y, Alpha
            for j in range(0, 3):
                # Ignore Alpha for now
                #if j == 2: continue
                #if j == 0 or j == 1: continue

                # Iterate each diff
                #for k in range(0, 3):

                # Step
                if j == 0 or j == 1:
                    h = 0.01
                    lr = 0.05
                if j == 2:
                    h = 0.01
                    lr = 0.5

                # Previous
                curveParamsPrev = curveParams.copy();
                curveParamsPrev[i,j] -= h;

                errorPrev = self.computeError(curveParamsPrev, targetPts)

                curveParamsNext =  curveParams.copy(); curveParamsPrev[i,j] += h;
                errorNext = self.computeError(curveParamsNext, targetPts)

                # Gradient
                grad = (errorNext - errorPrev)/(2.*h)

                # I NEED TO MAKE SURE ALPHA DOESNT GO BELOW 1.5 or above 10
                # Clip Gradient so value dont exceed?

                # Update
                curveParamsNew[i,j] -= lr*grad

                # Clip value
                if j == 2:
                    if curveParamsNew[i,j] < 1.5: curveParamsNew[i,j] = 1.5
                    elif curveParamsNew[i,j] > 10: curveParamsNew[i,j] = 10

                # NEED TO CLIP IF THE motion CASUES IT TO EXPLODE

        #stop
        print(curveParamsNew)

        return curveParamsNew

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
        output = pylc.Output()
        spline = pylc.fitBSpline(self.targets, splineParams, True)
        plt.scatter(spline[:,0], spline[:,1], s=0.5, c='r')
        projPoints = pylc.solveT(splineParams, self.targetpts)

        # Test Error
        self.computeError(self.targets, self.targetpts, True)

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
            curveParams = self.targets
            targetPts = self.targetpts
            start_time = time.time()
            self.targets = self.update(curveParams, targetPts)
            print(time.time() - start_time)

        #self.targets = curveParamsNew
        #print(curveParamsNew)


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
        plt.figure(2)
        plt.cla()

manager = Manager()
while 1:
    manager.loop()