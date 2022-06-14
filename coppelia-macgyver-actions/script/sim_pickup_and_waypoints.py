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
np.set_printoptions(suppress=True)


# get direction in body frame of reference
# basically multiply by 90 degrees about z axis.



# def calculate_final_orientation_body(W_T_B, W_T_V):
    #B_T_V = B_T_W * W_T_V, i.e. (bottle in world frame)-1 times vector in body frame
    # return np.linalg.inv(W_T_B) @ W_T_V


def normalize(x):
    return x/np.linalg.norm(x)


def add_collision_objects(col_obs):
    for col_ob in col_obs:
        name, (position, orientation), dimensions = col_ob, col_obs[col_ob]['pose'], col_obs[col_ob]['dimension']
        robot.addCollisionBoundingBox(name, position, orientation, dimensions)
        print("added ", name)

if __name__ == '__main__':

    np.random.seed(1234)
    print("STARTING NODE!")
    # object_name = 'cube'
    object_name = 'screwdriver'
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

    # obj_dim = [0.03, 0.05, 0.395]
    # obj_dim = [0.04, 0.18, 0.25]
    obj_dim = [0.03, 0.03, 0.30]
    reqd_len = 0.10
    # reqd_len = None
    pickup_height = 0.10


    collision_objs = {'target' : {'dimension' : [0.1, 0.1, 0.2]},
                      'obstacle1' : {'dimension' : [0.3, 0.1, 0.5]},
                      'obstacle2' : {'dimension' : [0.3, 0.1, 0.5]}
                     }
    single_axis_constraint = True

    target_dimensions = [0.1, 0.1, 0.2]
    target_name = 'target'

    obstacle_dimensions = [0.4, 0.1, 0.5]

    # waypoint1 = np.asarray([0.605652916431, 0.15, 0.6113])
    # waypoint2 = np.asarray([0.754177355766, 0.15, 0.6113])



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


    raw_input("Waiting?")
    print(tf_helper)
    print(robot)
    result, object_pose_in_base_link = segment_objects(tf_helper, robot, object_name=object_name, isSim=not isSim)

    object_pose_in_table_link = tf_helper.getTransform('place_pos', '/' + object_name)
    target_pose = tf_helper.getTransform('base_link', '/' + target_name)
    done

    '''
    Doing collision objects here. REFACTOR LATER!!
    '''
    for collision_obj in collision_objs:
        print(collision_obj)
        print(collision_objs[collision_obj])
        pose = tf_helper.getTransform('base_link', '/' + collision_obj)
        collision_objs[collision_obj]['pose'] = pose

    # print(collision_objs)
    add_collision_objects(collision_objs)
    print(np.asarray(quaternion_to_euler(collision_objs['obstacle1']['pose'][1])) * 180/np.pi)
    print("**************")

    z_rot = R.from_euler('xyz', [0,0,180]).as_dcm()


    transformation_matrix_for_wp = getMatrixFromQuaternionAndTrans(collision_objs['obstacle1']['pose'][1], target_pose[0])
    print(collision_objs['obstacle1']['pose'][0], target_pose[0])

    fwd_dst = collision_objs['obstacle1']['dimension'][0]/2 - (collision_objs['obstacle1']['pose'][0][0] - target_pose[0][0]) + 0.15
    bck_dst = -(collision_objs['obstacle1']['dimension'][0]/2 + (collision_objs['obstacle1']['pose'][0][0] - target_pose[0][0])) + 0.20

    print(fwd_dst, bck_dst)

    #TODO  : change 0.035 with height of tool!
    wp1 = np.asarray([fwd_dst, 0, -0.05, 1])
    wp2 = np.asarray([bck_dst, 0, -0.05, 1])

    # print(wp1, wp2)


    waypoint1 = np.matmul(transformation_matrix_for_wp, wp1.reshape((4,1)))
    waypoint2 = np.matmul(transformation_matrix_for_wp, wp2.reshape((4,1)))


    # transformation_matrix_for_wp[:3, :3] = np.matmul(z_rot,transformation_matrix_for_wp[:3, :3])

    waypoint1 /= waypoint1[-1]
    waypoint2 /= waypoint2[-1]
    waypoint_1 = waypoint1[:3]
    waypoint_2 = waypoint2[:3]
    print("\n\n\nWAYPOINTS!")
    print(waypoint_1, waypoint_2)
    
    waypoint2 = np.asarray(waypoint_2).reshape((3,))
    waypoint1 = np.asarray(waypoint_1).reshape((3,))
    

    #delete later
    waypoint2[-1] = waypoint1[-1]
    
    print("\n\n\nWAYPOINTS!")
    print(waypoint1, waypoint2)


    print(object_pose_in_base_link)
    print(object_pose_in_table_link)
    # nomore
    # result, grasp_trans = grasp_estimation(tf_helper, robot, object_name=object_name, object_path=objpath, isSim=isSim)


    print "object pose estimation"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()



    direction_vector = (waypoint2 - waypoint1).reshape((3,1))
    direction_vector /= np.linalg.norm(direction_vector)

    if single_axis_constraint:
        # raw_input("Printed pose of object in gripper frame")


        object_pose_in_base_frame = tf_helper.getTransform('/base_link', '/' + object_name)
        W_T_B = getMatrixFromQuaternionAndTrans(object_pose_in_base_frame[1], object_pose_in_base_frame[0])
        W_R_B =  W_T_B[:3, :3]
        r_z = W_R_B[:3, 2].reshape((3,1))


        r_z = normalize(r_z)

        print(direction_vector, direction_vector.shape, r_z, r_z.shape)

        rot_angle = np.arccos(np.dot(direction_vector.reshape((1,3)), r_z))

        rot_axis = normalize(np.cross(direction_vector, r_z, axis=0).squeeze())
        print(rot_angle, rot_axis, direction_vector, r_z)
        # thus I have rotation angle and rotation axis
        rot_matrix = matrixfromQuaternion(getQuaternionFromAxisAngle(rot_axis, -rot_angle))

        final_body_ori = np.dot(rot_matrix, W_R_B)
        final_body_pos = waypoint1
        print("-*************FINAL BODY POS *************")
        print(final_body_ori)
        # print(waypoint1, waypoint2)
        # dome


    else:

        y_vector = np.asarray([0,0,-1])
        x_vector = normalize(np.cross(y_vector, direction_vector, axis=0).squeeze())
        final_body_ori = np.eye(3)
        final_body_ori[:3,0] = x_vector
        final_body_ori[:3,1] = y_vector
        final_body_ori[:3,2] = direction_vector.squeeze()
        final_body_pos = waypoint1


        # z_rot = R.from_euler('xyz', [0,0,180*np.pi/180]).as_dcm()

        # transformation_matrix_for_wp = getMatrixFromQuaternionAndTrans(collision_objs['obstacle1']['pose'][1], waypoint1)
        # # correction from coppelia to robot coordinate frame.
        # print("BEFORE")
        # print(transformation_matrix_for_wp[:3, :3])
        # transformation_matrix_for_wp[:3, :3] = np.matmul(transformation_matrix_for_wp[:3, :3], z_rot)
        # print("AFTER")
        # print(transformation_matrix_for_wp[:3, :3])
        # print(direction_vector)
        
        
        # transformation_matrix_for_wp[:3, 2] = direction_vector.squeeze()
        # transformation_matrix_for_wp[:3, 0] = transformation_matrix_for_wp[:3, 1]
        # transformation_matrix_for_wp[:3, 1] = normalize(np.cross(transformation_matrix_for_wp[:3, 0].squeeze(), direction_vector, axis=0)).squeeze()
        # final_body_ori = transformation_matrix_for_wp[:3, :3]
        # final_body_pos = waypoint1

        print("-*************FINAL BODY POS *************")
        print(final_body_ori)
        # print(waypoint1, waypoint2)
        # done




    final_body_pose = np.eye(4)
    final_body_pose[:3, :3] = final_body_ori
    final_body_pose[:3, 3] = final_body_pos
    print(final_body_pose)
    # done





    result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(robot, planner, object_pose_in_base_link, object_pose_in_table_link, obj_dim, reqd_len, given_grasps = None, object_name=object_name)
    #result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(planner, object_pose_in_base_link, object_name=object_name)
    print("grasping object")
    if result:
        print("---SUCCESS---")
    else:
        print("---FAILURE---")
        exit()
    result = pickup(robot, tf_helper, pickup_height, object_name)
    print "pick up object"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()

    # exit()
    # dine
    # raw_input("Reposition!")

    # raw_input("Going to plan now")


    # Reorienting logic


    # target_position = np.asarray(target_pose[0])
    # z_angle = quaternion_to_euler(collision_objs['obstacle1'][pose][1])[-1]
    # vert_line = [0, target_position[1], 0]

    object_pose_in_ee_frame = tf_helper.getTransform('/gripper_link', '/' + object_name)
    B = getMatrixFromQuaternionAndTrans(object_pose_in_ee_frame[1], object_pose_in_ee_frame[0])

    final_ee_pose_in_body_frame = np.matmul(final_body_pose, np.linalg.inv(B))
    final_ee_pose_1 = final_ee_pose_in_body_frame.copy()
    wp1 = getTransformFromPoseMat(final_ee_pose_1)
    print("1st waypoint")
    print(final_body_pose)

    final_body_pose = np.eye(4)
    final_body_pose[:3, :3] = final_body_ori
    final_body_pose[:3, 3] = waypoint2
    final_ee_pose_in_body_frame = np.matmul(final_body_pose, np.linalg.inv(B))
    final_ee_pose_2 = final_ee_pose_in_body_frame.copy()
    wp2 = getTransformFromPoseMat(final_ee_pose_2)
    print("2nd waypoint")
    print(final_body_pose)






    push_with_waypoints(robot, [wp1, wp2], target_name)
    # follow_waypoints(robot, [wp1, wp2])
    robot.openGripper()
    # robot.detachManipulatedObject(object_name + '_collision')




    # set_ArmPos(robot)
    # raw_input("finished moving to defult art pose?")

