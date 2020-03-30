import os
import sys
import numpy as np
import matplotlib.pyplot as plt

build_dir = os.path.dirname(os.path.abspath(__file__)) + '/../build'
sys.path.append(build_dir)  # should contain pylc_lib compiled .so file
import pylc_lib
from sim import LCDevice


class Planner:
    def __init__(self, lc_device, pts_per_cam_ray=80, debug=False):
        self._lc_device = lc_device
        self._pts_per_cam_ray = pts_per_cam_ray
        self._debug = debug

        self._planner = pylc_lib.Planner(lc_device.datum_processor, self._debug)
        self._umap_hw = None

    def confidence2entropy(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3+)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x  : x in camera frame.
                                - z  : z in camera frame.
                                - c+ : confidence score of various factors, lying in [0, 1].
        Returns:
            entropy_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) entropy map of detector.
                             Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                             Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                             Axis 2 corresponds to (x, z, c):
                                 - x : x in camera frame.
                                 - z : z in camera frame.
                                 - e : entopy.
        """
        xz = confidence_map[:, :, :2]  # (X, Z, 2)
        p  = confidence_map[:, :, 2:]  # (X, Z, K)
        e  = (-p * np.log(1e-5 + p)).mean(axis=2, keepdims=True)  # (X, Z, 1)
        entropy_map = np.concatenate((xz, e), axis=2)  # (X, Z, 3)
        return entropy_map

    def get_design_points(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3+)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c+):
                                - x  : x in camera frame.
                                - z  : z in camera frame.
                                - c+ : confidence score of various factors, lying in [0, 1].
        Returns:
            design_points: (np.ndarray, dtype=float32, shape=(N, 2)) point cloud of design points.
                        Each point (axis 1) contains (x, z) location of design point in camera frame.

        """
        uncertainty_map = self.confidence2entropy(confidence_map)  # (X, Z, 3)

        if self._umap_hw is None or self._umap_hw != uncertainty_map.shape[:2]:
            self._umap_hw = uncertainty_map.shape[:2]
            umap_h, umap_w = self._umap_hw
            x_min, x_max = uncertainty_map[:, :, 0].min(), uncertainty_map[:, :, 0].max()
            z_min, z_max = uncertainty_map[:, :, 1].min(), uncertainty_map[:, :, 1].max()

            if self._debug:
                im = np.flip(uncertainty_map[:, :, 2].T, axis=0)
                plt.imshow(im, cmap='hot')
                plt.title("Uncertainty")
                plt.show()

            self._planner.constructGraph(umap_w, umap_h, x_min, x_max, z_min, z_max, self._pts_per_cam_ray)

            if self._debug:
                graph = self._planner.getGraphForVis()
                graph = np.array(graph)  # (RAYS, NODES_PER_RAY, 2)
                graph = graph.transpose(1, 0, 2)  # (NODES_PER_RAY, RAYS, 2)
                nodes = graph[:, :, 0].ravel()  # (NODES_PER_RAY, RAYS)
                nEdges = graph[:, :, 1]  # (NODES_PER_RAY, RAYS)

                x = np.array([node.x for node in nodes])
                z = np.array([node.z for node in nodes])
                r = np.array([node.r for node in nodes])
                tcam = np.array([node.theta_cam for node in nodes])
                tlas = np.array([node.theta_las for node in nodes])
                ki = np.array([node.ki for node in nodes])
                kj = np.array([node.kj for node in nodes])

                u = []
                for ki_, kj_ in zip(ki, kj):
                    if ki_ == -1 or kj_ == -1:
                        u_ = 0
                    else:
                        u_ = uncertainty_map[ki_, kj_, 2]
                    u.append(u_)

                plt.scatter(x, z, c=u, cmap='hot', s=1)
                plt.title("PYLC_PLANNER: Camera Ray Points w/ Interpolated Uncertainties", fontsize='x-large')
                plt.show()

                mean_connectivity = f"{nEdges[:, :-1].mean() / self._pts_per_cam_ray * 100:.2f}%"
                print(f"PYLC_PLANNER: mean connectivity across adjacent rays: {mean_connectivity}")

        design_points = self._planner.optimizedDesignPts(uncertainty_map[:, :, 2])
        design_points = np.array(design_points)

        if self._debug:
            flattened_umap = uncertainty_map.reshape(-1, 3)
            x, z, u = flattened_umap[:, 0], flattened_umap[:, 1], flattened_umap[:, 2]
            plt.scatter(x, z, c=u, cmap='hot')
            plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            plt.show()

        return design_points


if __name__ == '__main__':
    # LC Device used in Virtual KITTI.
    lc_device = LCDevice(
        CAMERA_PARAMS={
            'width': 1242,
            'height': 375,
            'fov': 81.16352842604304,
            # This matrix given in the README is not consistent with the
            # compute_camera_instrics_matrix function.
            # 'matrix': np.array([[725,   0, 620.5],
            #                     [  0, 725, 187.0],
            #                     [  0,   0,     1]], dtype=np.float32),
            'distortion': [0, 0, 0, 0, 0]
        },
        LASER_PARAMS={
            'y': -3.0  # place laser 3m to the right of camera
        }
    )

    planner = Planner(lc_device, debug=True)

    cmap = np.load("/home/sancha/repos/lcsim/python/example/confidence_map.npy")
    planner.get_design_points(cmap)
