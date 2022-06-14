#!/usr/bin/env python
import os
import rospy
import numpy as np
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
from rail_manipulation_msgs.srv import SegmentObjects
from database import dbaccess as db
import time


def deg2rad(deg):
    return np.pi * deg / 180.0

def rad2deg(rad):
    return rad * 180.0 / np.pi

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


def segment_objects(tf_helper, robot, object_name, isSim):
    rospy.wait_for_service('table_searcher/segment_objects')
    tableSearcher = rospy.ServiceProxy('table_searcher/segment_objects', SegmentObjects)
    try:
        tableresult = tableSearcher()
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))
        return False, None 
    print("Done with table and objects segment.") 
    # print(tableresult.segmented_objects.objects[0].point_cloud.orientation)
    return 


def detection_object(tf_helper, robot, object_name, isSim):
    print("IS SIM? ", isSim)
    if not isSim:
        # call the pose estimation node
        # get the object searcher trigger to control the tracker
        print("Waiting for!")
        rospy.wait_for_service('searchObject')
        print("Waited for!")

        tableSearcher = rospy.ServiceProxy('table_searcher/segment_objects', SegmentObjects)

        # objectSearcherTrigger = rospy.ServiceProxy('searchObject', SearchObject)

        ## launch the tracker
        print("Launching!")
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


def grasp_object(robot, planner, object_pose, object_pose_table, obj_dim = None, grasp_dist_len=None, given_grasps=None, object_name = None):

    # this function will return both pre grasp pose and grasp pose in base link frame
    def find_grasping_point_for_push(planner, tran_base_object, obj_pose_table, obj_dim, grasp_dist_len, gripper_pos_list=None):
        if gripper_pos_list == None:
            # print("*********")
            # print(tran_base_object[0], quaternion_to_euler(tran_base_object[1]))
            # print(obj_pose_table[0], quaternion_to_euler(obj_pose_table[1]))
            # print('******************')
            # filter out based on placement so we know which is the actuall grasp
            gripper_pos_list = planner.getGraspsbyPlacementPose(obj_pose_table)

            # print("***************")
            # done
            # print("RES1 : ", gripper_pos_list)
            if len(gripper_pos_list) == 0:
                # we have to try all grasps
                gripper_pos_list = planner.getAllGrasps()
            # print("RES2 : ", gripper_pos_list)
        else:
            pre_definedGrasps = []
            for obj_grasp_pos, jaw_width in gripper_pos_list:
                pre_definedGrasps.append([obj_grasp_pos, jaw_width])
            possibleGrasps = planner.getAllGrasps()
            gripper_pos_list = []
            for t_pose, _ in pre_definedGrasps:
                grasp_inv_rot = np.transpose(t_pose[:3,:3])
                grasp_trans = t_pose[:,3:]

                grasp_temp = []
                rot_diff = []
                tran_diff = []
                for ppose, pwidth in possibleGrasps:
                    rot_diff.append(np.linalg.norm(R.from_dcm(np.dot(grasp_inv_rot,ppose[:3,:3])).as_rotvec()))
                    tran_diff.append(np.linalg.norm(ppose[:,3:] - grasp_trans))
                    grasp_temp.append([ppose, pwidth, 0])
                tran_diff = softmax(tran_diff)
                rot_diff = softmax(rot_diff)
                for i in range(len(grasp_temp)):
                    grasp_temp[i][2] = tran_diff[i] + rot_diff[i]

                def sortfun(e):
                        return e[2]
                grasp_temp.sort(key=sortfun)

                gripper_pos_list.append((grasp_temp[0][0], grasp_temp[0][1]))

        max_dim, min_dim = np.argmax(obj_dim), np.argmin(obj_dim)
        length, width = obj_dim[max_dim], obj_dim[min_dim]
        print "Going through this many grasp pose: " ,len(gripper_pos_list)
        for i, (obj_grasp_pos, jaw_width) in enumerate(gripper_pos_list):


                grasp_point_ob = obj_grasp_pos[:3, 3].squeeze()
                print(abs(grasp_point_ob[1]), grasp_point_ob[max_dim])
                # Check 1 to grasp at the end!
                if grasp_dist_len:
                    #TODO do not hardcode
                    # print("Checking")
                    if grasp_point_ob[max_dim] > -grasp_dist_len :
                        continue
                    # else:
                        # print("GOOOOD")
                
                
                    # if (abs(grasp_point_ob[max_dim]) + length/2) < grasp_dist_len or grasp_point_ob[max_dim] > 0.9:
                    #     continue

                obj_grasp_trans_obframe = getTransformFromPoseMat(obj_grasp_pos) #Tranfrom gripper posmatx to (trans,rot)
                obj_grasp_trans_obframe =    transformProduct(obj_grasp_trans_obframe, [[+0.07,0,0],[0,0,0,1]]) #adjust the grasp pos to be a little back 
                obj_pre_grasp_trans =    transformProduct(obj_grasp_trans_obframe, [[-0.08,0,0],[0,0,0,1]]) #adjust the grasp pos to be a little back 
                obj_pre_grasp_trans = transformProduct(tran_base_object, obj_pre_grasp_trans)
                obj_grasp_trans = transformProduct(tran_base_object, obj_grasp_trans_obframe)



                # Check 2 for top down grasp only!
                topdown = True
                if topdown:
                    gripper_x_axis = getMatrixFromQuaternionAndTrans(obj_pre_grasp_trans[1], obj_pre_grasp_trans[0])[:3, 0]
                    object_z_axis = getMatrixFromQuaternionAndTrans(tran_base_object[1], tran_base_object[0])[:3, 2] 
                    reqd_z_axis = np.asarray([0,0,-1])
                    # rot_angle = np.arccos(np.dot(gripper_x_axis.reshape((1,3)), object_z_axis))
                    rot_angle = np.arccos(np.dot(gripper_x_axis.reshape((1,3)), reqd_z_axis))
                    if abs(rad2deg(rot_angle)) > 20 or abs(rad2deg(rot_angle)) < -20:
                        continue


                pre_grasp_ik_result = robot.solve_ik_collision_free_in_base(obj_pre_grasp_trans, 30)

                if pre_grasp_ik_result == None:
                        continue
                
                return obj_pre_grasp_trans, pre_grasp_ik_result, obj_grasp_trans, jaw_width, obj_grasp_trans_obframe
        # if not solution, then return None
        return None, None, None, None, None
    #Move to starting position
    robot.openGripper()
    obj_pre_grasp_trans, pre_grasp_ik_result, obj_grasp_trans, gripper_width, obj_grasp_trans_obframe = find_grasping_point_for_push(planner, object_pose, object_pose_table, obj_dim, grasp_dist_len, given_grasps)
    print(obj_pre_grasp_trans, obj_grasp_trans)
    # raw_input("This is what it is")
    # done
    if pre_grasp_ik_result == None: # can't find any solution then return false.
            return False, None, None

    plan = robot.planto_pose(obj_pre_grasp_trans)
    print("***")
    robot.display_trajectory(plan)
    # donenomore

    robot.execute_plan(plan)
    while True:

        pre_grasp_joint_angles = robot.get_ik_from_jointtrajectory(plan.joint_trajectory)
        moveit_robot_state = copy.deepcopy(pre_grasp_joint_angles)
        grasp_plan, fraction = robot.planFollowEndEffectorTrajectory(moveit_robot_state, [obj_grasp_trans], 0.01, 0)
        robot.display_trajectory(grasp_plan)
        ret = robot.execute_plan(grasp_plan)
        break
        va = raw_input("Going to grasp!")
        if int(va) != 0:
            break
        if int(va) == 0:
            done
    robot.closeGripper(gripper_width - 0.025)

    robot.attachManipulatedObject(object_name + "_collision")
    return True, obj_grasp_trans_obframe, gripper_width

# in hand pose estimation is currently optional. This function is used to estimate the object
# pose in hand.
# input: initialized estimated pose. If there is not initialize guess pose in hand then give None
# output: issuccess, preditect object pose in hand
def in_hand_pose_estimation(tf_helper, robot, guess_pose = None):

    object_pose_in_hand = None
    return True, object_pose_in_hand

# get init grasps will return a set of init grasp of the object
# return: issuccess, list of target grasps

def move_forward(robot):
    waypoints = []
    scale = 5
    direc = -1
    no_steps = 1
    wpose = robot.get_current_pose()

    for no in range(no_steps):
        wpose.position.x += scale * 0.1 * direc  # First move up (z)
        waypoints.append(copy.deepcopy(wpose))

    # wpose.position.x -= scale * 0.1 * dir  # Second move forward/backwards in (x)
    # waypoints.append(copy.deepcopy(wpose))
    # waypoints.append(copy.deepcopy(wpose))

    (plan, fraction) = robot.group.compute_cartesian_path(
        waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
    )
    print(fraction, plan)
    robot.display_trajectory(plan)
    ret = robot.execute_plan(plan)
    print(ret)


def gotowp(robot, wp):
    print("Going to : ", wp)
    plan = robot.planto_pose(wp)
    robot.display_trajectory(plan)
    # raw_input("Moving to WP1")
    robot.execute_plan(plan)
    # raw_input("Moved to WP1")

def move_to_push(robot, waypoint):
    plan = robot.planto_pose(waypoint)
    robot.display_trajectory(plan)
    # done
    robot.execute_plan(plan)
    # raw_input("Moved to WP1")
    return plan

def push_action(robot, moveit_robot_state, waypoints, cartesian=True):

    # moveit
    if not cartesian:
        for waypoint in waypoints:
            plan = robot.planto_pose(waypoint)

    #cartesian
    else:
        wposes = []
        for waypoint in waypoints:
            wp = Pose()
            waypoint_eef_frame = copy.deepcopy(waypoint)
            wp.position.x = waypoint_eef_frame[0][0]
            wp.position.y = waypoint_eef_frame[0][1]
            wp.position.z = waypoint_eef_frame[0][2]
            wp.orientation.x = waypoint_eef_frame[1][0]
            wp.orientation.y = waypoint_eef_frame[1][1]
            wp.orientation.z = waypoint_eef_frame[1][2]
            wp.orientation.w = waypoint_eef_frame[1][3]
            wposes.append(wp)
        print(wposes)
        # print(moveit_robot_state)
        plan, fraction = robot.planFollowEndEffectorTrajectory(moveit_robot_state, waypoints, 0.01, 5.0)

        # (plan, fraction) = robot.group.compute_cartesian_path(
        #     wposes, 0.01, 0.0  # waypoints to follow  # eef_step
        # )
    # return
    robot.display_trajectory(plan)
    # raw_input("Moving to WP2")
    robot.execute_plan(plan)
    # raw_input("Moved to WP2")


def push_with_waypoints(robot, waypoints, target_object_name):

    # obj_pre_grasp_trans = waypoints[0]
    # obj_grasp_trans = waypoints[1]
    # plan = robot.planto_pose(obj_pre_grasp_trans)
    # print("***")
    # robot.display_trajectory(plan)
    # robot.execute_plan(plan)
    # robot.removeCollisionObject(target_object_name)
    # while True:

    #     pre_grasp_joint_angles = robot.get_ik_from_jointtrajectory(plan.joint_trajectory)
    #     moveit_robot_state = copy.deepcopy(pre_grasp_joint_angles)
    #     grasp_plan, fraction = robot.planFollowEndEffectorTrajectory(moveit_robot_state, [obj_grasp_trans], 0.01)
    #     robot.display_trajectory(grasp_plan)
    #     ret = robot.execute_plan(grasp_plan)
    #     break
    # return 
    print(waypoints)
    plan = move_to_push(robot, waypoints[0])
    pre_grasp_joint_angles = robot.get_ik_from_jointtrajectory(plan.joint_trajectory)
    moveit_robot_state = copy.deepcopy(pre_grasp_joint_angles)

    robot.removeCollisionObject(target_object_name)
    raw_input("Pushing!")
    time.sleep(2.0)
    push_action(robot, moveit_robot_state, waypoints[1:])

def follow_waypoints(robot, waypoints):

    wposes = []
    # wpose = robot.get_current_pose()
    for waypoint in waypoints:
        wp = Pose()
        wp.position.x = waypoint[0][0]
        wp.position.y = waypoint[0][1]
        wp.position.z = waypoint[0][2]
        wp.orientation.x = waypoint[1][0]
        wp.orientation.y = waypoint[1][1]
        wp.orientation.z = waypoint[1][2]
        wp.orientation.w = waypoint[1][3]
        wposes.append(wp)


    
    
    # wp2 = Pose()
    # wp2.position.x = waypoints[1][0]
    # wp2.position.y = waypoints[1][1]
    # wp2.position.z = waypoints[1][2]
    # wp2.orientation = wpose.orientation

    # print(waypoints[0])
    plan = robot.planto_pose(waypoints[0])
    robot.display_trajectory(plan)
    raw_input("Moving to WP1")
    robot.execute_plan(plan)
    raw_input("Moved to WP1")

    # (plan, fraction) = robot.group.compute_cartesian_path(
    #     wposes[1:], 0.01, 0.0  # waypoints to follow  # eef_step
    # )
    # raw_input("Moving to WP2")
    # print(fraction, plan)
    # robot.display_trajectory(plan)
    # ret = robot.execute_plan(plan)
    # raw_input("Moved to WP2")
    # print(ret)
    plan = robot.planto_pose(waypoints[1])
    robot.display_trajectory(plan)
    raw_input("Moving to WP2")
    robot.execute_plan(plan)
    raw_input("Moved to WP2")


def add_table(robot, tf_helper):
    """
    add the table into the planning scene
    """
    # get table pose
    table_transform = tf_helper.getTransform('/base_link', '/Table')
    # add it to planning scene
    robot.addCollisionTable("table", table_transform[0][0], table_transform[0][1], table_transform[0][2]+.001, \
        table_transform[1][0], table_transform[1][1], table_transform[1][2], table_transform[1][3], \
        0.7, 1.5, 0.09) #.7 1.5, .06 

def add_object(robot, tf_helper,object_name,object_path=None):
    """
    add the object into the planning scene
    """
    this_dir, filename = os.path.split(os.path.realpath(__file__)) 
    object_path = os.path.join(os.path.split(this_dir)[0], "objects", object_name + ".stl") 
    # add the object into the planning scene 
    current_object_transform = tf_helper.getTransform('/base_link', '/' + object_name)# get object pose
    robot.addCollisionObject(object_name + "_collision", current_object_transform, object_path, size_scale = .15)# add it to planning scene



# pickup is the action to move the gripper up in the base_link frame
def pickup(robot, tf_helper, height, object_name):
    """Pick up object"""

    target_transform = tf_helper.getTransform('/base_link', '/gripper_link')
    target_transform[0][2] += height
    print(target_transform)

    robot.switchController('my_cartesian_motion_controller', 'arm_controller')

    while not rospy.is_shutdown():
        if robot.moveToFrame(target_transform, True):
            break
        rospy.sleep(0.01)

    robot.switchController('arm_controller', 'my_cartesian_motion_controller')


    # while True:
    #     pre_grasp_joint_angles = robot.get_ik_from_jointtrajectory(plan.joint_trajectory)
    #     moveit_robot_state = copy.deepcopy(pre_grasp_joint_angles)
    #     grasp_plan, fraction = robot.planFollowEndEffectorTrajectory(moveit_robot_state, [target_transform], 0.01)
    #     print(grasp_plan)
    #     robot.display_trajectory(grasp_plan)
    #     ret = robot.execute_plan(grasp_plan)
    #     print(ret)
    #     va = raw_input("Going to grasp!")
    #     if int(va) != 0:
    #         break

    # robot.detachManipulatedObject(object_name + "_collision")

    return True, target_transform
