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
    object_name = 'cube'
    isSim = True
    rospy.init_node('simtestpush_node')

    print("CREATING ROBOT")
    robot = Fetch_Robot(sim=isSim)
    tf_helper = TF_Helper()

    print("CREATED ROBOT")
    # waypoint1 = [0.555652916431, 0.3333, 0.6513]
    # waypoint2 = [0.754177355766, 0.03, 0.6513]
    # result, object_pose_in_base_link = detection_object(tf_helper, robot, object_name=object_name, isSim=isSim)

    # wp1 = (np.asarray([0.605652916431, 0.3333, 0.6513]), np.array([0.45227263, 0.55649989, 0.13225048, 0.50270061]))
    robot.openGripper()
    # robot.attachManipulatedObject(object_name + "_collision")
    # wp1 = np.array([[-0.69754939,  0.07250761, -0.71285868,  0.83128325],
    #    [-0.17905384, -0.98094321,  0.07543297,  0.13038506],
    #    [-0.69380442,  0.18025831,  0.6972391 ,  0.76842832],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    wp1 = np.array([[ 0.42021711, -0.74173055,  0.52273642,  0.46655808],
       [ 0.46575564,  0.67069739,  0.57726657, -0.29319569],
       [-0.77877421,  0.00089015,  0.62730371,  0.7796876 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

                    
    # wp1[:3, :3] = np.eye(3)
    # transformation = np.eye(3)
    # transformation[:3, :3] = R.from_euler('xyz', [180+10,120,0], degrees=True).as_dcm()
    # wp1[:3, :3] = np.matmul(transformation, wp1[:3, :3])
    # print(wp1)
    # np.save('waypoint1.npy', wp1)
    wp1 = getTransformFromPoseMat(wp1)
    # wpx = (np.load('wp1_0.npy'), np.load('wp1_1.npy'))
    # wpx[0][0] = 0.76
    # wpx[0][1] = 0.03
    # wpx[0][2] = 0.63    
    # print(getMatrixFromQuaternionAndTrans(wpx[1], wpx[0]))

    # print(wpx)

    # raw_input("Moving forward")

    # result, manipulation_trans_on_table = detect_table_and_placement(tf_helper, robot)
    # print "table detection and placement analysis"
    # if result:
    #     print "---SUCCESS---"
    # else:
    #     print "---FAILURE---"
    #     exit()


    # result, object_pose_in_base_link = detection_object(tf_helper, robot, object_name=object_name, isSim=isSim)
    # robot.removeCollisionObject(object_name)

    # move_forward(robot)
    gotowp(robot, wp1)

