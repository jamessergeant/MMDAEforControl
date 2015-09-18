#!/usr/bin/env python
import sys, argparse, rospy
import numpy as N
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rospy.numpy_msg import numpy_msg
import time
from std_msgs.msg import String,Bool,Float64MultiArray
from tf.transformations import euler_from_quaternion as efq
from node_classes import Model, SequenceInput, GoalManager
import untangle

class DAE_node:
    def __init__(self,settings):

        print("loading model...")

        self.dae = Model(settings.model)

        self.model = self.dae.model
        self.mean = self.dae.model.mean
        self.std = self.dae.model.std
        self.sequence = self.dae.model.sequence
        self.normalise = self.dae.model.normalise
        self.f = self.dae.model.function

        self.new_goal = False
        self.message_sent = False

        self.goals = GoalManager(settings)

        self.dim = list()
        self.inputs = list()

        self.laser = None
        self.goal = None
        self.command = None

        in_count = 0
        if self.model.mmdae_type[0]:
            self.laser = SequenceInput(self.model.dims[in_count],self.mean[in_count],self.std[in_count],self.sequence,self.normalise[in_count])
            in_count += 1
        if self.model.mmdae_type[1]:
            self.goal = SequenceInput(self.model.dims[in_count],self.mean[in_count],self.std[in_count],self.sequence,self.normalise[in_count])
            in_count += 1
        if self.model.mmdae_type[2]:
            self.command = SequenceInput(self.model.dims[in_count],self.mean[in_count],self.std[in_count],self.sequence,self.normalise[in_count])
            in_count += 1

        print("...done.")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.goal_pub = rospy.Publisher("/goal_reached", String, queue_size=10)
        self.acc_pub = rospy.Publisher("/accuracy",String, queue_size=10)
        self.data_pub = rospy.Publisher('/data', Float64MultiArray,queue_size=10)

        self.scan_sub = rospy.Subscriber("/base_scan", numpy_msg(LaserScan), self.laser_callback)
        self.goal_sub = rospy.Subscriber("/goal", String, self.goal_callback)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        self.stall_sub = rospy.Subscriber("/stall", Bool, self.stall_callback)

        self.twist_msg = Twist()
        self.string_msg = String()
        self.data_msg = Float64MultiArray()

        self.multiplier = 1.0

        print("\nAwaiting first goal!")

        time.sleep(2)

        self.string_msg.data = 's'
        self.goal_pub.publish(self.string_msg)
        self.string_msg.data = ''

    def stall_callback(self,stall):

        if stall.data:
            self.string_msg.data = str(self.last_distance)
            self.acc_pub.publish(self.string_msg)
            self.string_msg.data = 'c'
            self.goal_pub.publish(self.string_msg)
            print 'Collision!'
            rospy.signal_shutdown('Collision')
            sys.exit("Collision")

    def pose_callback(self,pose):

        if not self.model.mmdae_type[1]:
            return

        # Convert pose to x,y,theta (from quaternion orientation)
        quaternion = N.asarray((pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z, pose.pose.pose.orientation.w))
        pose = N.asarray([pose.pose.pose.position.x, pose.pose.pose.position.y,efq(quaternion)[2]])

        self.goals.update_pose(pose)

        if self.goals.next_goal_flag:

            self.string_msg.data = str(self.last_distance)
            self.acc_pub.publish(self.string_msg)

            self.string_msg.data = str(self.current_goal)
            self.goal_pub.publish(self.string_msg)

            self.multiplier = self.goals.last_distance

    def laser_callback(self,laser):

        if self.goal is not None:
            # Check to see if first goal received
            if not self.goals.first_goal:
                return
            # Shift goal sequence and add current relative goal
            self.goals.update_relative()

        # Process laser measurement
        if self.laser is not None:
            self.laser.update(laser.ranges)

        # Generate command sequence

        self.twist_msg.linear.x, self.twist_msg.angular.z = self.dae.process(self.laser,self.goal,self.command)

        # At waypoint, rotate toward next waypoint
        if self.goals.reorient:
            self.twist_msg.linear.x *= self.multiplier
            self.twist_msg.angular.z = N.sign(self.goals.relative_goal[1]) * 0.5
        #     data = N.asarray((1)).reshape(1)
        # else:
        #     data = N.asarray((0)).reshape(1)

        # Publish command
        self.cmd_pub.publish(self.twist_msg)

        # Print information
        sys.stdout.write('Goal:\tx = %2.1f, y = %2.1f\t\tTo Goal:\tr = %2.1f, theta = %2.1f\r' % (self.goals.current_goal[0],self.goals.current_goal[1],self.goals.relative_goal[0],self.goals.relative_goal[1]))
        sys.stdout.flush()


        # data = N.concatenate((data,self.laser*self.std[0] + self.mean[0]),axis=1)
        # data = N.concatenate((data,self.goal.reshape(self.goal.shape[1])*N.tile(self.std[1],5) + N.tile(self.mean[1],5)),axis=1)
        # data = N.concatenate((data,y[0,:905]*self.std[0] + self.mean[0]),axis=1)
        # data = N.concatenate((data,y[0,905:905+10]*N.tile(self.std[1],5) + N.tile(self.mean[1],5)),axis=1)
        # data = N.concatenate((data,y[0,905+10:905+10+10]*N.tile(self.std[2],5) + N.tile(self.mean[2],5)),axis=1)
        #
        # self.data_msg.data = list(data)
        # self.data_pub.publish(self.data_msg)

    def goal_callback(self,goal):

        if goal.data in ['s','c']:
            rospy.signal_shutdown('goal reached')
            print 'Goal Reached!'
            sys.exit("Goal Reached")

        # receive new goal
        self.goals.update_goal(N.asarray([float(goal.data.split(',')[0]),float(goal.data.split(',')[1])]))

        sys.stdout.write('                                                                                                                            \r')
        sys.stdout.flush()

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")

    parser.add_argument('xml',
                        help='Specifies the xml file containing settings')

    return parser

def main(settings):
    rospy.init_node('DAE_node', anonymous=True)
    daeNode = DAE_node(settings)
    rospy.spin()

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    settings = untangle.parse(args.xml)
    main(settings.settings)
