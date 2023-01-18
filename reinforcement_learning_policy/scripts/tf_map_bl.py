#!/usr/bin/env python

"""
@Author: Miro
@Date: 10/12/2022
@Version: 1.0
@Objective: Transformation from map to base link coordination's
@TODO:
"""

import roslib
roslib.load_manifest('reinforcement_learning_policy')
from geometry_msgs.msg import PointStamped, Pose2D, PoseStamped
import rospy
import math
import tf
import geometry_msgs.msg
from reinforcement_learning_policy.srv import *


def tf_frame(req):
    if req.from_fr.header.frame_id == "/base_link":
        from_frame = "/base_link"
        to_frame = "/map"
    elif req.from_fr.header.frame_id == "/map":
        from_frame = "/map"
        to_frame = "/base_link"
    else:
        return
    listener, out_trans = tf.TransformListener(), None
    try:
        listener.waitForTransform(to_frame, from_frame, rospy.Time(0), rospy.Duration(5))
        out_trans = listener.transformPoint(to_frame, req.from_fr)
    except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.loginfo("Error during the position transformation")
    out_frame = from_frame + "->" + to_frame
    rospy.loginfo(out_frame)
    rospy.loginfo("(%.2f, %.2f. %.2f) -----> (%.2f, %.2f, %.2f)",
                  req.from_fr.point.x, req.from_fr.point.y, req.from_fr.point.z,
                  out_trans.point.x, out_trans.point.y, out_trans.point.z)

    return map_bl_serviceResponse(out_trans)


def main():
    rospy.init_node('tf_map_bl')
    rospy.Service('/tf_map_bl', map_bl_service, tf_frame)
    rospy.spin()


if __name__ == '__main__':
    main()
