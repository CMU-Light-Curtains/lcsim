##!/usr/bin/env python
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
            # I NEED TO HANDLE CASE WHERE POINT IS NOT VISIBLE, SO THE COST IS NOT BIASED JUST FOR ACCELERATION


            # [ 1.15725806,  4.87554113,  4       ],
            # [ 0.22580644,  6.20129871,  4],
            # [-0.62096774,  3.11688312,  4       ], ##
            # [-1.49596774,  6.12012987,  4.        ],
            # [-2.34274194,  5.01082251,  4.        ],
            # [-3.41532258,  6.09307359,  4.        ]


            # # A - Sometimes making it out of order is better
            # [-4,5, 1.5],
            # [-4.092742,  6.525974,  8.027994],
            # [3,5, 1.5],

            # # Out of bounds path
            # [-4,5, 1.5],
            # [-1.9475807,  2.5757575,  8.027994],
            # [3,5, 1.5],

            # # Xsort fails here, but angle sort works
            # [-2.455645,   7.012987,   1.5      ],
            # [-2.060484,   4.442641,   8.027994 ],
            # [ 2.8225806,  7.012987,   1.5      ],
            # [ 2.21758065,  4.469697,   4.       ]

            #
            # # Angle fails but xsort works
            # [-3.923387 ,  7.310606  , 1.5      , 0],
            # [ 0.4798387 , 8.419913 ,  8.027994 , 1],
            # [ 0.4516129 , 7.121212 ,  1.5      , 2],
            # [ 3.641129 ,  5.6331167,  4.       , 3],
            # [-0.4798387 , 6.0930734,  4.       , 4],
            # [-1.1008065 , 5.876623 ,  4.       , 5],
            # [-2.625   ,   5.6872296 , 4.       , 6]

            # # Weird
            #  [-2.6814516 ,  9.09632  ,   8.027994 ,   0.        ],
            # [ 1.326613   , 7.4188313 ,  1.5    ,     1.        ],
            # [-1.5806452  , 7.310606  ,  4.    ,      2.        ],
            # [ 0.02822581,  7.310606  ,  4.  ,        3.        ]

            # Just one point case
            #[ 0.02822581,  7.310606  ,  4.  ,        3.        ]

            # [ 0.02822581,  7.310606  ,  4.    ,      0.        ],
            # [ 2.1733871  , 7.44588745,  4.   ,       1.        ],
            # [ 0.        ,  8.98809528,  4.   ,       2.        ],
            # [ 2.17338705,  9.17748928,  4.   ,       3.        ],
            # [-1.72177419 , 8.96103896,  4.   ,       4.        ],
            # [-1.63709677 , 7.28354978 , 4.   ,       5.        ]

            # [ 0.19758065 , 4.5238094 ,  4.    ,      0.        ],
            # [ 1.3830645  , 4.3885283 ,  4.    ,      1.        ],
            # [ 1.5241935  , 5.8225107 ,  4.    ,      3.        ],
            # [-1.5524193  , 5.849567 ,   4.    ,      4.        ],
            # [-1.5806452  , 4.632035 ,   4.    ,      5.        ],
            # [-0.02822581 , 5.9577923 ,  4.    ,      6.        ]

            # [ 0.02822581 , 7.310606   , 4.   ,       3.        ],
            # [ 2.17338705 , 7.52705622 , 4.  ,        1.        ],
            # [-1.9193548 ,  8.90692616 , 4.  ,        1.        ],
            # [ 1.0725807 ,  8.77164459,  4.  ,        1.        ],
            # [-0.70564514,  8.90692616,  4.  ,        1.        ],
            # [-2.93548393 , 9.58333302,  4.  ,        1.        ],
            # [ 1.12903225 , 9.69155884 , 4.  ,        1.        ],
            # [-1.5524193  , 7.52705622 , 4.  ,        1.        ],
            # [ 2.56854839 , 9.33982684 , 4.  ,        1.        ]

            # Crashes
            # [-0.33870968,  5.5790043 ,  4.  ,        0.        ],
            # [-0.53629035 , 7.148268 ,   4.    ,      5.        ]

            #[0.0790382,   2.77403,         1, 0]

            # [-2.65322581,  8.77164502 , 1.  ,        0.        ],
            # [-0.79032258,  7.36471861 , 4.  ,        1.        ],
            # [ 2.65322581,  8.33874459 , 4.  ,        1.        ],
            # [ 2.45564516,  6.47186147,  4.  ,        1.        ],
            # [ 1.77822581,  4.9025974 ,  4.  ,        1.        ],
            # [ 0.31048387,  8.01406926,  4.  ,        1.        ],
            # [-2.14516129,  6.30952381 , 4.  ,        1.        ],
            # [ 0.16935484,  5.98484848 , 4.  ,        1.        ]


            # # Good for Real
            # [-2.51209677,  9.39393939,  1, 0,],
            # [-1.18548387,  3.38744589,  1.,0],
            # [1.49596774,  3.33333333,  1.,0],
            # [2.96370968,  9.39393939,  1.,0],
            # [-0.56451613,  4.33441558,  1.,0],
            # [0.90322581,  4.36147186,  1.,0],
            # [0.,          8.60930736,  1.,0],
            # [0.76209677,  8.55519481,  1.,0],

            # # For Fake
            # [-3.81048387,  8.5021645,   1.,0,],
            # [3.24596774,  8.36688312,  1.,0,],
            # [0.19758065,  2.52164502,  4.,0]
            #

            [-3.2459676,   8.44697,     1., 0.],
            [ 3.1330645,   8.582252,    1., 0.],
            [-0.05645161,  1.1958874,   1., 0.]

            # [0.4798387, 8.419913,  8.027994,  1.      ],
            # [0.5927419, 6.931818,  1.5,       2.       ]


        ]).astype(np.float32)

        # Demonstrate a case where unordered is actually better! like this

        # Display
        plt.figure(1)
        plt.cla()


        # Click
        self.click = Click(plt.gca(), self.targets)


        pass


    def loop(self):
        global datum_processor

        self.targets = self.click.cv

        # Draw all targets
        plt.figure(1)
        squaresize = 0.3
        for i in range(0, self.targets.shape[0]):
            X = self.targets[i,0]
            Y = self.targets[i,1]
            rect = patches.Rectangle((X - squaresize/2.,Y - squaresize/2.),squaresize,squaresize,linewidth=1,edgecolor='r',facecolor='none')
            plt.gca().add_patch(rect)

        # Pass Targets into function

        paths = [self.targets]
        output = pylc.Output()
        #output.output_pts_set = [np.zeros((3,1),dtype=np.float32)]
        #self.targets = np.flip(self.targets)
        start_time = time.time()

        fitting_module.curtainNodes(self.targets, "camera01", "laser01", output, True)
        print(time.time() - start_time)
        #print("--- %s seconds ---\n" % (time.time() - start_time)*1000)

        output_pts_set = output.output_pts_set
        spline_set = output.spline_set
        for i in range(0, len(output_pts_set)):
            output_pts = output_pts_set[i]
            spline = spline_set[i]
            plt.scatter(spline[:,0], spline[:,1], s=1.0, c='r')
            plt.scatter(output_pts[0, :], output_pts[2, :], s=1.0, c='b')


        # output_pts = output.output_pts
        # spline = output.spline
        # plt.scatter(spline[:,0], spline[:,1], s=1.0, c='r')
        # plt.scatter(output_pts[0, :], output_pts[2, :], s=1.0, c='b')

        # output_pts = output_pts.T
        # for i in range(0, output_pts.shape[0]):
        #     print(str(output_pts[i,0]) + " " + str(output_pts[i,2]))
        # stop
        # print(output_pts)
        # stop

        # Plot
        print("-")
        for i in range(self.targets.shape[0]):
            plt.text(self.targets[i,0],self.targets[i,1],str(i))

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