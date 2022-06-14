#!/usr/bin/env python
import os
import rospy
import numpy as np
from tf_util import TF_Helper, PandaPosMax_t_PosMat, transformProduct, getMatrixFromQuaternionAndTrans, getTransformFromPoseMat, align_vectors, getQuaternionFromAxisAngle, matrixfromQuaternion, pose_to_array

from rail_segmentation.srv import SearchTable
from scipy.spatial.transform import Rotation as R
from scipy.special import softmax
from std_srvs.srv import Empty
from rail_manipulation_msgs.srv import SegmentObjects 
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d



import numpy as np
from ctypes import * # convert float to uint32

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
np.set_printoptions(suppress=True)


def convertCloudFromRosToOpen3d(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    field_names.remove("rgb")
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = ['x', 'y', 'z']))
    # print(field_names)
    # print(list(pc2.read_points(ros_cloud)))

    # Check empty
    o3d_cloud = o3d.geometry.PointCloud()

    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set o3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # combine
        o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        o3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

    # prn
    return o3d_cloud

def o3d_icp(model_pcl_np, scene_pcl_np_filtered, initial_guess, solver_params): 
    """Performs ICP with Open3d's Generalized-ICP implementation.
    Input: 
        - model_pcl_np: Nx3 np.float32 array, model pointcloud 
        - scene_pcl_np_filtered: Nx3 np.float32 array, model pointcloud
        - initial_guess: Drake RigidTransform object used as initial guess to correspondence. 
        - solver_params: dictionary containing solver parameters.
    Output:
        - X_MS: Estimated relative transform between model and scene pointclouds
    """

    reg_p2p = o3d.registration.registration_icp(
        model_pcl_np, scene_pcl_np_filtered, 
        solver_params["max_correspondence_distance"], initial_guess,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.registration.ICPConvergenceCriteria(
            relative_fitness=solver_params["relative_fitness"],
            relative_rmse=solver_params["relative_rmse"],
            max_iteration=solver_params["max_iteration"])
    )


    return reg_p2p.transformation


def segment_tabletop(tf_helper):
    rospy.wait_for_service('table_searcher/segment_objects')
    tableSearcher = rospy.ServiceProxy('table_searcher/segment_objects', SegmentObjects)
    try:
        tableresult = tableSearcher()
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))
        return False, None 
    print("Done with table and objects segment.")
    # for seg_obj in tableresult.segmented_objects.objects:
    #     print(seg_obj.orientation,  seg_obj.centroid, seg_obj.center)
    return tableresult.segmented_objects.objects




def detect_table_and_placement(tf_helper, robot=None):
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

    if robot:
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



def get_pose_and_publish(object_name, segmented_object, tf_helper, robot=None):
    initial_guess = np.eye(4)
    initial_guess[:3, 3] = [segmented_object.center.x, segmented_object.center.y, segmented_object.center.z]
    solver_params = {
        "max_correspondence_distance": 1.0,
        "relative_fitness": 1e-6,
        "relative_rmse": 1e-6,
        "max_iteration": 30
    }

    target_pcl = convertCloudFromRosToOpen3d(segmented_object.point_cloud)

    #TODO change!
    source_pcl = o3d.io.read_point_cloud('/home/olorin/Documents/models/screwdriver_blender/model.ply')

    pose_4x4 = o3d_icp(source_pcl, target_pcl, initial_guess, solver_params)
    target_transform = getTransformFromPoseMat(pose_4x4)
    



    tf_helper.pubTransform("/computed_"+object_name, target_transform)
    if robot:
        robot.addCollisionObject(object_name + "_collision", target_transform, "/home/olorin/Desktop/CRI/MacgyverProject/Coppelia/coppeliasim-macgyver/catkin_ws/src/macgyver_actions/objects/" + object_name + ".stl")




if __name__ == '__main__':
    rospy.init_node('perceptioner')
    tf_helper = TF_Helper()
    print("inited!")
    result, manipulation_trans_on_table = detect_table_and_placement(tf_helper, None)
    print "table detection and placement analysis"
    if result:
        print "---SUCCESS---"
    else:
        print "---FAILURE---"
        exit()


    seg_objs = segment_tabletop(tf_helper)
    print("Number of segmented objects : ", len(seg_objs))

    # print(seg_objs[-1].center)


    solver_params = {
        "max_correspondence_distance": 1.0,
        "relative_fitness": 1e-6,
        "relative_rmse": 1e-6,
        "max_iteration": 30
    }


    for idx, seg_obj in enumerate(seg_objs):
        bounding_box_msg = seg_obj.bounding_volume
        position, orientation = pose_to_array(bounding_box_msg.pose.pose) 
        print(position)
        dimensions = bounding_box_msg.dimensions
        raw_input("added")



    # initial_guess = np.eye(4)

    # target_pcl = convertCloudFromRosToOpen3d(seg_objs[-1].point_cloud)
    # source_pcl = o3d.io.read_point_cloud('/home/olorin/Documents/models/screwdriver_blender/model.ply')
    # print(type(target_pcl), type(source_pcl))
    # output = o3d_icp(source_pcl, target_pcl, initial_guess, solver_params)
    # print(output)
    # print(R.from_dcm(output[:3, :3]).as_euler('xyz'))

