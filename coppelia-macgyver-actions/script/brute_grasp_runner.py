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
from brute_grasp import *



def normalize(x):
    return x/np.linalg.norm(x)


if __name__ == '__main__':

    np.random.seed(1234)
    print("STARTING NODE!")
    isSim = True
    rospy.init_node('simtest_node')
    object_name = 'bottle'
    pc = o3d.io.read_point_cloud('/home/olorin/Desktop/CRI/MacgyverProject/YCBDataset/models/006_mustard_bottle/google_16k/nontextured.ply')
    o3d.visualization.draw_geometries([pc])
    dibe



    print("CREATING ROBOT")
    robot = Fetch_Robot(sim=isSim)
    tf_helper = TF_Helper()
    gp = GraspPlanner()




    result, manipulation_trans_on_table = detect_table_and_placement(tf_helper, robot)
    print "table detection and placement analysis"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()


    # raw_input("Waiting?")
    result, object_pose_in_base_link = detection_object(tf_helper, robot, object_name=object_name, isSim=isSim)

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



    gp.sample_grasps(robot, pc, object_pose_in_base_link, np.eye(4))


    # result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(robot, planner, object_pose_in_base_link, object_pose_in_table_link, obj_dim, reqd_len, given_grasps = None, object_name=object_name)
    #result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(planner, object_pose_in_base_link, object_name=object_name)
    print("grasping object")
    if result:
        print("---SUCCESS---")
    else:
        print("---FAILURE---")
        exit()

