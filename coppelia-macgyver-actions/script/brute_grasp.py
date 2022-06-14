#!/usr/bin/env python
import os
import rospy
import open3d as o3d
import numpy as np
import logging
import transforms3d
# from fetch_robot import Fetch_Robot
from regrasp_planner import RegripPlanner
from tf_util import TF_Helper, PandaPosMax_t_PosMat, transformProduct, getMatrixFromQuaternionAndTrans, getTransformFromPoseMat, align_vectors, quaternion_to_euler, pose_to_array

from scipy.spatial.transform import Rotation as R
from scipy.special import softmax
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
import copy
import pandaplotutils.pandageom as pandageom
import pandaplotutils.pandactrl as pandactrl
from utils import robotmath as rm
from utils import dbcvt as dc
from rail_segmentation.srv import SearchTable


def detect_table_and_placement(tf_helper, robot):
    # call the table searcher server for searching table
    rospy.wait_for_service('table_searcher/search_table')
    tableSearcher = rospy.ServiceProxy('table_searcher/search_table', SearchTable)

    # analyze where to manipulate the object
    try:
        tableresult = tableSearcher()
    except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            return False, None

    r = R.from_quat([tableresult.orientation.x, tableresult.orientation.y, tableresult.orientation.z, tableresult.orientation.w])
    # need to adjust the rotation of the table
    original_r = r.as_euler('zyx')
    table_quaternion = R.from_euler('zyx', [0,original_r[1], original_r[2]]).as_quat()

    robot.addCollisionTable("table", tableresult.center.x, tableresult.center.y, tableresult.center.z, \
                        table_quaternion[0], table_quaternion[1], table_quaternion[2], table_quaternion[3], \
                        tableresult.width * 1.8, tableresult.depth * 2, 0.001)
    # robot.attachTable("table")
    robot.addCollisionTable("table_base", tableresult.center.x, tableresult.center.y, tableresult.center.z - 0.3, \
                    table_quaternion[0], table_quaternion[1], table_quaternion[2], table_quaternion[3], \
                    tableresult.width, tableresult.depth, 0.6)

    # show the position tf in rviz
    tf_helper.pubTransform("place_pos", ((tableresult.centroid.x, tableresult.centroid.y, tableresult.centroid.z), \
                                    (tableresult.orientation.x, tableresult.orientation.y, tableresult.orientation.z, tableresult.orientation.w)))

    return True, [[[tableresult.centroid.x, tableresult.centroid.y, tableresult.centroid.z], \
                                    [tableresult.orientation.x, tableresult.orientation.y, tableresult.orientation.z, tableresult.orientation.w]]]

# detectino_object will detect the object and return the object pose
# return: issuccess, poseMat_of_object


def detection_object(tf_helper, robot, object_name, isSim):
    print("IS SIM? ", isSim)
    if not isSim:
        # call the pose estimation node
        # get the object searcher trigger to control the tracker
        rospy.wait_for_service('searchObject')
        objectSearcherTrigger = rospy.ServiceProxy('searchObject', SearchObject)

        ## launch the tracker
        objectSearcherTrigger(True, 1, Pose())

        raw_input("running pose estimation")

        try:
                # add the object to the moveit
                target_transform = tf_helper.getTransform('/base_link', '/' + object_name)
                robot.addCollisionObject(object_name + "_collision", target_transform, "objects/" + object_name + ".stl")
                print("Added robot collision")
        except Exception as e:# if failure, then return False
                objectSearcherTrigger(False, 0, Pose())
                print e
                return False, None
        
        objectSearcherTrigger(False, 0, Pose())
    else:
        try:
            print(object_name)
            target_transform = tf_helper.getTransform('/base_link', '/' + object_name)
            # target_transform = tf_helper.getTransform('/world', '/' + object_name)
            # print(target_transform[0], quaternion_to_euler(target_transform[1]))

            # target_transform_table = tf_helper.getTransform('/Table', '/' + object_name)
            print("GETTING TRANSFORM!", target_transform)
            robot.addCollisionObject(object_name + "_collision", target_transform, "/home/olorin/Desktop/CRI/MacgyverProject/Coppelia/coppeliasim-macgyver/catkin_ws/src/macgyver_actions/objects/" + object_name + ".stl")
            print("Added collision as well!!")
        except Exception as e:# if failure, then return False
                print e
                # notworking
                return False, None

    if not isSim:
        # stop updating octo map
        rospy.wait_for_service('stop_octo_map')
        try:
            octoclient = rospy.ServiceProxy('stop_octo_map', StopOctoMap)
            octoclient()
        except rospy.ServiceException as e:
            print ("Fail to stop octo map controller: %s"%e)
            return False, None

    return True, target_transform


class GraspPlanner:
    def __init__(self, top_k_grasp = 20, debug_lvl = 1):
        self._setup_logger()
        self.debug_lvl = debug_lvl
        self.top_k_grasp = top_k_grasp
        self.reset()


    def reset(self, seed=None):
        self.plan_result = None
        self.goal_pose_costs = []
        self.goal_poses = []
        self.grasp_pose_idx = 0
        self.policy_info = {}
        self.no_valid_path_cnt = 0
        self.max_no_valid_path_cnt = 15

    def _setup_logger(self):
        self.logger = logging.getLogger('BFSGPolicy')
        self.logger.setLevel(logging.DEBUG)

    def visualize_pointclouds(self, pcs):
        if self.debug_lvl >= 1:
            # visualize pc
            geometries = pcs
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=(0, 0, 0))
            geometries.append(coord_frame)
            o3d.visualization.draw_geometries(geometries)

    def grasp_candidate_cost(self, goal_pose, pc, robot_qpos, adjust_X_G=True):
        """Compute cost for the candidate grasp"""
        X_G = goal_pose.copy()
        # Transform cloud into gripper frame
        X_GW = np.linalg.inv(goal_pose)
        pts = np.asarray(pc.points).T
        p_GC = np.dot(X_GW[:3, :3], pts) + X_GW[:3, 3][:, None]

        # Crop to a region inside of the finger box.
        crop_min = [-0.017088/2, -0.04, 0.044862+0.0584-0.017088/2]
        crop_max = [0.017088/2, 0.04, 0.044862+0.0584+0.017088/2]

        indices = np.all((crop_min[0] <= p_GC[0, :], p_GC[0, :] <= crop_max[0],
                            crop_min[1] <= p_GC[1, :], p_GC[1, :] <= crop_max[1],
                            crop_min[2] <= p_GC[2, :], p_GC[2, :] <= crop_max[2]),
                            axis=0)

        if adjust_X_G and np.sum(indices) > 0:
            p_GC_y = p_GC[1, indices]
            p_Gcenter_y = (p_GC_y.min() + p_GC_y.max()) / 2.0

            X_G[:3, 3] += np.dot(X_G[:3, :3], [0.0, p_Gcenter_y, 0.0])

            goal_pose = X_G.copy()
            X_GW = np.linalg.inv(goal_pose)

        n_GC = np.dot(X_GW[:3, :3], np.asarray(pc.normals)[indices, :].T)

        cost = 0.0
        # Penalize deviation of the gripper from vertical.
        # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
        #cost = 20.0*X_G[:3, :3][2, 1]
        # Encourage lower grasps
        #cost = -10.0*X_G[2, 3]  # FIXME: Here's actually higher grasps

        # Reward sum |dot product of normals with gripper y|^2
        cost -= np.sum(n_GC[1, :]**2)
        return cost, goal_pose

    def generate_grasp_candidate_antipodal(self, robot, pc, obj_pose, idx, robot_qpos):
        """Generate one antipodal candidate grasp and compute its cost"""
        def _generate_grasp_candidate_antipodal(normal):
            """Use the normal for gripper y axis"""
            Gy = normal  # gripper y axis aligns with normal
            # make orthonormal x axis, aligned with world up
            x = np.array([0.0, 0.0, 1.0])
            if np.abs(np.abs(np.dot(x, Gy)) - 1.0) < 1e-6:
                # normal was pointing straight up.  reject this sample.
                return np.inf, None
            Gx = x - np.dot(x, Gy)*Gy
            Gx /= np.linalg.norm(Gx)
            # gripper z axis is towards object
            Gz = np.cross(Gx, Gy)
            R_WG = np.vstack((Gx, Gy, Gz)).T
            p_GS_G = np.array([0.0, 0.04-1e-3, 0.044862+0.0584])

            # Try orientations from the center out
            #min_pitch=-np.pi/3.0
            #max_pitch=np.pi/3.0
            #alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
            min_pitch = -np.pi
            max_pitch = 0
            alpha = np.array([0.5, 1.0, 0.0, 0.8, 0.2, 0.65, 0.35])
            for theta in (min_pitch + (max_pitch - min_pitch)*alpha):
                # Rotate the object in the hand by a random rotation (around the normal).
                R_WG2 = np.dot(R_WG, transforms3d.euler.euler2mat(0.0, theta, 0.0, 'sxyz'))

                # Use G for gripper frame
                p_SG_W = np.dot(-R_WG2, p_GS_G)
                p_WG = p_WS + p_SG_W
                # print(p_WG, R_WG2)

                obj_grasp_pos = np.eye(4)
                obj_grasp_pos[:3, :3] = R_WG2
                obj_grasp_pos[:3, 3] = p_WG

                cost, goal_pose_x = self.grasp_candidate_cost(obj_grasp_pos, pc, robot_qpos, adjust_X_G=True)
                print(cost, goal_pose_x)
                # if np.isinfinite
                if not np.isfinite(cost):
                    continue

                obj_grasp_trans_obframe = getTransformFromPoseMat(obj_grasp_pos) #Tranfrom gripper posmatx to (trans,rot)
                obj_grasp_trans_obframe =    transformProduct(obj_grasp_trans_obframe, [[+0.02,0,0],[0,0,0,1]]) #adjust the grasp pos to be a little back 
                # obj_grasp_trans_obframe = transformProduct(obj_grasp_trans_obframe, [[0.08,0,0],[0,0,0,1]]) # try to move the gripper forward little
                obj_pre_grasp_trans =    transformProduct(obj_grasp_trans_obframe, [[-0.08,0,0],[0,0,0,1]]) #adjust the grasp pos to be a little back 
                obj_pre_grasp_trans = transformProduct(obj_pose, obj_pre_grasp_trans)
                obj_grasp_trans = transformProduct(obj_pose, obj_grasp_trans_obframe)


                pre_grasp_ik_result = robot.solve_ik_collision_free_in_base(obj_pre_grasp_trans, 30)

                if pre_grasp_ik_result == None:
                    print('check on grasp ', idx, pre_grasp_ik_result)
                    continue

                
                # goal_pose_p = getTransformFromPoseMat(obj_pre_grasp_trans)
                plan = robot.planto_pose(obj_pre_grasp_trans)
                print(plan)
                robot.display_trajectory(plan)
                robot.execute_plan(plan)
                raw_input("What to do? ")
                return cost, obj_grasp_trans


            
            print("Returning infinite")
            return np.inf, None

        # S for sample point/frame
        p_WS = np.asarray(pc.points[10 * idx])
        n_WS = np.asarray(pc.normals[10 * idx])
        # print(n_WS, np.linalg.norm(n_WS))
        # done
        assert np.isclose(np.linalg.norm(n_WS), 1.0, rtol=0.001)

        # Use both n_WS and negated n_WS as point normal to generate grasp
        cost, goal_pose = _generate_grasp_candidate_antipodal(n_WS)
        neg_cost, neg_goal_pose = _generate_grasp_candidate_antipodal(-n_WS)
        if np.isfinite(cost) and cost <= neg_cost:
            return cost, goal_pose
        elif np.isfinite(neg_cost) and neg_cost <= cost:
            return neg_cost, neg_goal_pose

        return np.inf, None

    def sample_grasps(self, robot, pc, obj_pose, robot_qpos):
        """Randomly sample and ranks candidate grasps. Return True if found a grasp."""
        from tqdm import tqdm
        # self.logger.debug(f'[ grasp_sampler ]: Start sampling '
        #                     f'among {len(pc.points)} object points')
        for i in tqdm(range(len(pc.points)/10), disable=(self.debug_lvl == 0)):
            cost, goal_pose = self.generate_grasp_candidate_antipodal(robot, pc, obj_pose, i, robot_qpos)
            if np.isfinite(cost):
                self.goal_pose_costs.append(cost)
                self.goal_poses.append(goal_pose)

        indices = np.asarray(self.goal_pose_costs).argsort()
        indices = indices[:self.top_k_grasp]  # take only top_k_grasp for speed

        if indices.size > 0:
            self.goal_pose_costs = np.asarray(self.goal_pose_costs)[indices]
            self.goal_poses = np.asarray(self.goal_poses)[indices]

            # self.logger.debug(f'[ grasp_sampler ]: Sampled {len(self.goal_poses)} candidate grasps')

            cost = self.goal_pose_costs[self.grasp_pose_idx]
            ##antipodal_cost = cost + 10.0*goal_pose.p[-1]
            antipodal_cost = cost
            goal_pose = self.goal_poses[self.grasp_pose_idx]
            # with np.printoptions(precision=3, suppress=True):
            #     self.logger.debug(f'[ select_next_grasp ]: '
            #                         f'{self.grasp_pose_idx+1}/{len(self.goal_poses)}. '
            #                         f'grasp_pose='
            #                         f'{np.hstack((goal_pose.p, goal_pose.q))} with '
            #                         f'total_cost={cost:.2f} '
            #                         f'antipodal_cost={antipodal_cost:.2f}')
            plan = robot.planto_pose(goal_pose)
            robot.display_trajectory(plan)
            robot.execute_plan(plan)

            return True
        else:
            return False


if __name__ == '__main__':
    gp = GraspPlanner()

    pc = o3d.io.read_point_cloud('/home/olorin/Desktop/CRI/MacgyverProject/YCBDataset/models/006_mustard_bottle/google_16k/nontextured.ply')
    print(pc)
    gp.sample_grasps(pc, np.eye(4))