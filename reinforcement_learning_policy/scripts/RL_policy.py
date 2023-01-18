#!/usr/bin/env python3

"""
@Author: Miro
@Date: 10/12/2022
@Version: 1.0
@Objective: Creation of RL policy
@TODO:
"""

from geometry_msgs.msg import PointStamped, PoseStamped
from test_drone import *

dir_weights = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/weights/"
dir_trajectory = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/trajectory/"
plot_dir = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/plots/"
goal_check = 0.216 * 1.5
min_clip_value = 0.25


class SavePolicy:
    def __init__(self):
        self.input_policy = None
        self.goal_x = 0
        self.goal_y = 0


policies = SavePolicy()


def callback_input(data):
    input_train.goal.x = data.point.x
    input_train.goal.y = data.point.y
    input_train.t = 1


def callback_goal(data):
    policies.goal_x = data.pose.position.x
    policies.goal_y = data.pose.position.y


def pub_goal(msg):
    plot(msg.data[0].data)
    new_scale = 0.05
    result = np.array(list(msg.data[0].data)).reshape(80, 80)
    x, y = np.unravel_index(result.argmax(), result.shape)
    rob_pos = np.array((80 / 2, 80 / 2))
    print([x, y])
    print([-x * new_scale, y * new_scale])
    x = -(x - rob_pos[0]) * new_scale
    y = (y - rob_pos[1]) * new_scale
    _, current_x, current_y = get_processed_input()
    action_x, action_y, new_point = setting_sub_goal(current_x, current_y, [[x, y]])

    point_smp_tf = tf_map_bl_fun("/base_link", 0, new_point)
    new_point.x = point_smp_tf.point.x
    new_point.y = point_smp_tf.point.y
    new_point.theta = np.arctan2((new_point.y - current_y), (new_point.x - current_x))

    print([x, y])
    print(new_point)

    try:
        sp = rospy.ServiceProxy('robot_goal', NN)
        sp(new_point)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)

    plot(list(result), "fusion.png")


def input_policy_callback(msg):
    rospy.loginfo("input policy found")
    policies.input_policy = np.array(list(msg.data[0].data)).reshape(80, 80)


def plot(data, name="rl_policy.png"):
    plt.clf()
    grid_map = np.asarray(list(data)).reshape(80, 80)
    plt.pcolor(grid_map)
    plt.colorbar()
    plt.savefig(plot_dir + name)


def tf_map_bl_fun(frame, sequence, new_point):
    point_smp = PointStamped()
    point_smp.header.frame_id = frame
    point_smp.header.seq = sequence
    point_smp.point.x = new_point.x
    point_smp.point.y = new_point.y

    point_smp_tf = None
    try:
        prox_tf_map_bl = rospy.ServiceProxy('/tf_map_bl', map_bl_service)
        point_smp_tf = prox_tf_map_bl(point_smp)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)
    return point_smp_tf.to_fr


def call_model_policy(pub):
    current_state, current_x, current_y = get_processed_input()
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)

    action = policy_definition(tf_prev_state)
    action_x, action_y, new_point = setting_sub_goal(current_x, current_y, action)

    prova = PointStamped()
    prova.header.frame_id = "/map"
    prova.point.x = new_point.x
    prova.point.y = new_point.y

    rospy.loginfo("Output NN /map %f,%f", new_point.x, new_point.y)
    rospy.loginfo(action)
    rospy.loginfo([current_x, current_y])

    pub.publish(prova)
    return current_x, current_y, new_point


def policy_build(directory_weights, directory_trajectory):
    init_test(directory_weights, directory_trajectory)
    reduce_error_of_positioning()
    rospy.loginfo("*******")
    rospy.loginfo("RL policy")
    rospy.loginfo("*******\n")
    sequence = 0

    rospy.Subscriber("/user_input", PointStamped, callback_input)
    rospy.Subscriber("/goal", PoseStamped, callback_goal)
    pub = rospy.Publisher('/rl_input', PointStamped, queue_size=10)
    current_x, current_y, new_point = 0, 0, Point(0, 0, 0)
    while not rospy.is_shutdown():
        if input_train.t == 0: continue

        if sqrt(pow(policies.goal_x - current_x, 2) + pow(policies.goal_y - current_y, 2)) < 0.2 or input_train.t == 1:
            call_model_policy(pub)
            input_train.t = -1
            sequence += 1

        _, current_x, current_y = get_processed_input()


if __name__ == '__main__':
    rospy.init_node('RL_policy')
    policy_build(dir_weights, dir_trajectory)
