/*
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: send goal to mobile robot
@TODO:
*/

#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <drone/NN.h>
#include <tf/tf.h>



typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

bool moveAction(drone::NN::Request  &req, drone::NN::Response &res){
  //tell the action client that we want to spin a thread by default
  MoveBaseClient ac("move_base", true);
  ROS_INFO("Goal node: on");
  //wait for the action server to come up
  while(!ac.waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the move_base action server to come up");}
  move_base_msgs::MoveBaseGoal goal;

  goal.target_pose.header.frame_id = "map";
  goal.target_pose.header.stamp = ros::Time::now();
  goal.target_pose.pose.position.x = req.goal.x;
  goal.target_pose.pose.position.y = req.goal.y;
  goal.target_pose.pose.position.z = 0;
  goal.target_pose.pose.orientation = tf::createQuaternionMsgFromYaw(req.goal.theta);

  
  ROS_INFO("Sending goal %f, %f", req.goal.x, req.goal.y);
  ac.sendGoal(goal);
  ac.waitForResult();
  
  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
    ROS_INFO("Hooray, the base moved");
    res.new_processing = true;
  }
  else{
    ROS_INFO("The base failed to move for some reason");
    res.new_processing = false;
  }
  return true;
}

int main(int argc, char** argv){
  
  ros::init(argc, argv, "goal_sending");
  ros::NodeHandle n;
  ros::ServiceServer service = n.advertiseService("robot_goal", moveAction);
  ros::spin();

}
