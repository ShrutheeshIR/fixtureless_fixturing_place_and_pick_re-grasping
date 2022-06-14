#!/usr/bin/env python
import os
import rospy
import numpy as np
from fetch_robot_updated import Fetch_Robot
from regrasp_planner import RegripPlanner
from tf_util import TF_Helper, PandaPosMax_t_PosMat, transformProduct, getMatrixFromQuaternionAndTrans, getTransformFromPoseMat, align_vectors, getQuaternionFromAxisAngle, matrixfromQuaternion

# from rail_segmentation.srv import SearchTable
from scipy.spatial.transform import Rotation as R
from scipy.special import softmax
from manipulation.grip.fetch_gripper import fetch_grippernm

from std_srvs.srv import Empty
# from rail_manipulation_msgs.srv import SegmentObjects 

import pandaplotutils.pandageom as pandageom
import pandaplotutils.pandactrl as pandactrl
from utils import robotmath as rm
from utils import dbcvt as dc
from database import dbaccess as db
from sim_utils import *
import time
from segmenter import *

np.set_printoptions(suppress=True, precision = 5)




def normalize(x):
    return x/np.linalg.norm(x)


def add_collision_objects(col_obs):
    for col_ob in col_obs:
        name, (position, orientation), dimensions = col_ob, col_obs[col_ob]['pose'], col_obs[col_ob]['dimension']
        robot.addCollisionBoundingBox(name, position, orientation, dimensions)
        print("added ", name)



def get_waypoints(target, tool, obstacles):


    obstacle_pose = pose_to_array(obstacles[0].bounding_volume.pose.pose)

    target_pose = pose_to_array(target.bounding_volume.pose.pose)

    print("target ", target.bounding_volume.pose.pose.position)
    print("obstacle ", obstacles[0].bounding_volume.pose.pose.position,obstacles[0].bounding_volume.dimensions )

    #FIXME this hot mess of a code :( -----------------> This somehow works, either do not touch, or revamp, nothing in between!
    transformation_matrix_for_wp = getMatrixFromQuaternionAndTrans(obstacle_pose[1], target_pose[0])
    rotate_by_neg_90 = R.from_euler('xyz', [0,90,0], degrees=True).as_dcm()
    transformation_matrix_for_wp[:3, :3] = np.matmul(transformation_matrix_for_wp[:3, :3], rotate_by_neg_90)
    

    fwd_dst = obstacles[0].bounding_volume.dimensions.x/2 - (obstacles[0].bounding_volume.pose.pose.position.x - target.bounding_volume.pose.pose.position.x) - 0.12
    bck_dst = -(obstacles[0].bounding_volume.dimensions.x/2 - (obstacles[0].bounding_volume.pose.pose.position.x - target.bounding_volume.pose.pose.position.x)) - 0.12
    
    print(fwd_dst, bck_dst)

    #TODO  : change 0.035 with height of tool!
    wp1 = np.asarray([bck_dst, 0, 0.00, 1])
    wp2 = np.asarray([fwd_dst, 0, 0.00, 1])
    print(wp1, wp2)


    waypoint1 = np.matmul(transformation_matrix_for_wp, wp1.reshape((4,1)))
    waypoint2 = np.matmul(transformation_matrix_for_wp, wp2.reshape((4,1)))

    waypoint1 /= waypoint1[-1]
    waypoint2 /= waypoint2[-1]
    waypoint_1 = waypoint1[:3]
    waypoint_2 = waypoint2[:3]
    print("\n\n\nWAYPOINTS!")
    print(waypoint_1, waypoint_2)
    
    waypoint2 = np.asarray(waypoint_2).reshape((3,))
    waypoint1 = np.asarray(waypoint_1).reshape((3,))
    #FIXME this as well :
    # waypoint2[-1] = waypoint1[-1]

    return waypoint1, waypoint2
    

def get_push_ori_from_waypoints(waypoint1, waypoint2, object_name, axis_constraint):
    direction_vector = (waypoint2 - waypoint1).reshape((3,1))
    direction_vector /= np.linalg.norm(direction_vector)

    if axis_constraint:
        # raw_input("Printed pose of object in gripper frame")


        object_pose_in_base_frame = tf_helper.getTransform('/base_link', '/computed_' + object_name)
        W_T_B = getMatrixFromQuaternionAndTrans(object_pose_in_base_frame[1], object_pose_in_base_frame[0])
        W_R_B =  W_T_B[:3, :3]
        r_z = W_R_B[:3, 2].reshape((3,1))

        r_z = normalize(r_z)
        # print(direction_vector, direction_vector.shape, r_z, r_z.shape)
        rot_angle = np.arccos(np.dot(direction_vector.reshape((1,3)), r_z))
        rot_axis = normalize(np.cross(direction_vector, r_z, axis=0).squeeze())
        # print(rot_angle, rot_axis, direction_vector, r_z)
        # thus I have rotation angle and rotation axis
        rot_matrix = matrixfromQuaternion(getQuaternionFromAxisAngle(rot_axis, -rot_angle))
        final_body_ori = np.dot(rot_matrix, W_R_B)


    else:

        y_vector = np.asarray([0,0,-1])
        x_vector = normalize(np.cross(y_vector, direction_vector, axis=0).squeeze())
        final_body_ori = np.eye(3)
        final_body_ori[:3,0] = x_vector
        final_body_ori[:3,1] = y_vector
        final_body_ori[:3,2] = direction_vector.squeeze()

    return final_body_ori


def get_end_effector_pose_waypoint(waypoint, orientation):

    body_pose = np.eye(4)
    body_pose[:3, :3] = orientation
    body_pose[:3, 3] = waypoint

    object_pose_in_ee_frame = tf_helper.getTransform('/gripper_link', '/computed_' + object_name)
    print("obj pose in ee frame", object_pose_in_ee_frame)
    B = getMatrixFromQuaternionAndTrans(object_pose_in_ee_frame[1], object_pose_in_ee_frame[0])
    final_ee_pose_in_body_frame = np.matmul(body_pose, np.linalg.inv(B))
    final_ee_pose_1 = final_ee_pose_in_body_frame.copy()
    print(final_ee_pose_1)
    wp = getTransformFromPoseMat(final_ee_pose_1)
    return wp


def get_ee_poses(waypoint1, waypoint2, orientation):
    return [get_end_effector_pose_waypoint(waypoint1, orientation), get_end_effector_pose_waypoint(waypoint2, orientation)]
    




if __name__ == '__main__':

    np.random.seed(1234)
    print("STARTING NODE!")
    # object_name = 'cube'
    isSim = True
    rospy.init_node('simtest_node')


    # Things I am hard coding
    '''
    tool dimensions in its own frame, tool pose.
    target pose.

    For obstacle avoidance for planning.
    target bounding box and pose (identity)
    obstacle bounding box and pose (may not be identity, this gives direction vector)


    '''
    #TODO
    '''
    - specify a grasp buffer
    - do not hardcode
    '''


    #FIXME later : hardcoding
    objects_mapping = {0 : 'obstacle1', 1 : 'obstacle2', 2 : 'target', 3 : 'tool', 4 : 'dummy', 5 : 'dumm2'}
    obj_dim = [0.03, 0.03, 0.30]
    reqd_len = 0.10
    pickup_height = 0.10
    object_name = 'screwdriver'
    single_axis_constraint = True
    





    print("CREATING ROBOT")
    robot = Fetch_Robot(sim=isSim)
    tf_helper = TF_Helper()

    robot.openGripper()
    time.sleep(1.0)

    print("CREATED ROBOT")
    base = pandactrl.World(camp = [700,300,1400], lookatp=[0,0,0])
    this_dir, this_file = os.path.split(os.path.realpath(__file__))
    objpath = os.path.join(os.path.split(this_dir)[0], "objects", object_name + '.stl')
    handpkg = fetch_grippernm
    print("STARTING!!!!")

    gdb = db.GraspDB()
    planner = RegripPlanner(objpath, handpkg, gdb)


    result, manipulation_trans_on_table = detect_table_and_placement(tf_helper, robot)
    print "table detection and placement analysis"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()


    #TODO : comeup with a better name:
    segmented_objects_dict = {}

    
    segmented_objects = segment_tabletop(tf_helper)
    for idx, seg_obj in enumerate(segmented_objects):
        bounding_box_msg = seg_obj.bounding_volume
        position, orientation = pose_to_array(bounding_box_msg.pose.pose) 
        dimensions = bounding_box_msg.dimensions
        print(R.from_quat(orientation).as_dcm(), position)
        print(dimensions)

        segmented_objects_dict[objects_mapping[idx]] = seg_obj

        # segmented_objects[idx].name = objects_mapping[idx]

        # if not target
        # robot.addCollisionBoundingBox(objects_mapping[idx], position, orientation, [dimensions.x, dimensions.y, dimensions.z])
        if objects_mapping[idx] != 'tool':
            print("Hello")

        else:
            get_pose_and_publish(object_name, seg_obj, tf_helper, robot)

        # raw_input("Added : %d"%idx)

    # ohnonono
    # exit
    #DO LATER

    object_pose_in_base_link = tf_helper.getTransform('base_link', '/' + object_name)
    object_pose_in_table_link = tf_helper.getTransform('place_pos', '/' + object_name)

    waypoint1, waypoint2 = get_waypoints(segmented_objects_dict['target'], segmented_objects_dict['tool'], [segmented_objects_dict['obstacle1'], segmented_objects_dict['obstacle2']])
    push_direction = get_push_ori_from_waypoints(waypoint1, waypoint2, object_name, single_axis_constraint)
    result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(robot, planner, object_pose_in_base_link, object_pose_in_table_link, obj_dim, reqd_len, given_grasps = None, object_name=object_name)


    print("grasping object")
    if result:
        print("---SUCCESS---")
    else:
        print("---FAILURE---")
        exit()
    result = pickup(robot, tf_helper, pickup_height, object_name)
    waypoints = get_ee_poses(waypoint1, waypoint2, push_direction)
    print(waypoints)
    push_with_waypoints(robot, waypoints, 'target')



    # print "pick up object"
    # if result:
    #     print "---SUCCESS---"
    # else:
    #     print "---FAILURE---"
    #     exit()

