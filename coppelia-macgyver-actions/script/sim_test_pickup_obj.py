#!/usr/bin/env python
import os
import rospy
import numpy as np
from fetch_robot_updated import Fetch_Robot
from regrasp_planner import RegripPlanner
from tf_util import TF_Helper, PandaPosMax_t_PosMat, transformProduct, getMatrixFromQuaternionAndTrans, getTransformFromPoseMat, align_vectors

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






if __name__ == '__main__':

    np.random.seed(1234)
    print("STARTING NODE!")
    object_name = 'bottle'
    isSim = True
    rospy.init_node('simtest_node')



    print("CREATING ROBOT")
    robot = Fetch_Robot(sim=isSim)
    tf_helper = TF_Helper()


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

    # raw_input("Waiting?")
    result, object_pose_in_base_link = detection_object(tf_helper, robot, object_name=object_name, isSim=isSim)

    # c_x, c_y, c_z = manipulation_trans_on_table[0][0]
    # o_x, o_y, o_z, o_w = manipulation_trans_on_table[0][1]
    # print(c_x, c_y, c_z, o_x, o_y, o_z, o_w)
    # object_pose_in_table_link = tf_helper.getTransform(((c_x, c_y, c_z), \
    #                                 (o_x, o_y, o_z, o_w)), '/' + object_name)


    object_pose_in_table_link = tf_helper.getTransform('place_pos', '/' + object_name)
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

    # raw_input("go grasp object")

    # init_grasp_pose = getMatrixFromQuaternionAndTrans(grasp_trans[1], grasp_trans[0])


    # result, init_grasps = planner.getAllGrasps()
    # print(result, init_grasps)
    # print("get init grasps")
    # if result:
    #     print("---SUCCESS---")
    # else:
    #     print("---FAIL")
    #     print("Number of init grasp: ", len(init_grasps))
    # #raw_input("Press enter to continue")
    # exit()

    obj_dim = [0.075, 0.075, 0.30]
    reqd_len = 0.24

    result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(robot, planner, object_pose_in_base_link, object_pose_in_table_link, obj_dim, reqd_len, given_grasps = None, object_name=object_name)
    #result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(planner, object_pose_in_base_link, object_name=object_name)
    print("grasping object")
    if result:
        print("---SUCCESS---")
    else:
        print("---FAILURE---")
        exit()

    # raw_input("Ready to pick up object?")

    result = pickup(robot, tf_helper, 0.15)
    print "pick up object"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()

    # set_ArmPos(robot)
    raw_input("finished moving to defult art pose?")