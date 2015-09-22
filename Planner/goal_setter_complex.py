#!/usr/bin/env python
import sys, argparse, rospy, cv2
import numpy as N
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rospy.numpy_msg import numpy_msg
from theano import function
from pylearn2.utils import serial
import string
import time
from matplotlib import pyplot as P
import matplotlib.cm as cm
from math import floor,sqrt,ceil
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion as efq
from math import atan2,sqrt,pi
import re
import random
from Dalgorithm import shortestPath
import os

class goal_setter:
  def __init__(self):
      self.goal_pub = rospy.Publisher("/goal", String, queue_size=10)
      self.goal_num_pub = rospy.Publisher("/goal_number", String, queue_size=10)
      self.subscriber = rospy.Subscriber("/goal_reached", String, self.goal_callback)
      self.goals = {1: [-18,18],2: [-3.5,19],3: [3,19],4: [18,19],5: [-12,13],6: [-3.5,13],8: [12,12],9: [-18,10],10: [18,12],11: [-18,5],12: [-3,6],13: [2.5,7],14: [-4,0],15: [0,0],16: [6.5,2],17: [18,2],18: [-10,-3],19: [-2.5,-2.5], 20: [-18,-4], 21: [-13,-10], 22: [4,-9], 23: [11,-11], 24: [18,-11], 25: [-18,-18], 26: [-7,-16], 27: [4,-15], 28:[18,-18], 29: [2,-3], 30:[3,14], 31: [6,14], 32: [6,11], 33:[2.5,10.5],7:[-12,18.5], 34:[18,0]}
      self.next_goal = {1: [5,9,7],2: [3,6,7],3: [2,4,30],4: [3,10],5: [1,6,9,7],6: [2,5,12],8: [10],9: [1,5,11,7],10: [4,8,17],11: [9,18,20],12: [6,13,14,15],13: [12,15,16,33],14: [12,15,18,19],15: [12,13,14,16,19,29],16: [13,15,17,29],17: [10,16,34],18: [11,14],19: [14,15,21,29],20: [11,21,25], 21: [19,20,26], 22:[34,26,27], 23:[24], 24:[34,28,23], 25:[20,26], 26: [21,22,25,27], 27: [22,26,28], 28: [24,27],29:[15,16,19], 30:[3,31], 31:[30,32], 32:[31,33],33:[13,32],7:[1,2,5,9],34: [17,24,22]}
      l = len(self.goals)
      self.createDict()
      self.path = []
      random.seed()
      while len(self.path) < 4:
          self.start = random.randint(1,l)
          self.end = self.start
          while self.end == self.start:
              self.end = random.randint(1,l)
          self.path = shortestPath(self.distances,self.start,self.end)
      self.current_goal = self.start
      print "Start: %i,\tEnd:%i" % (self.start,self.end)
      print "Path: ", self.path
      self.count = 1
      world = open(os.environ['MMDAE'] + 'Worlds/complex/world.world').read()
      text_file = open(os.environ['MMDAE'] + 'Worlds/complex/pos_world.world', "w")
      a = self.goals[self.path[0]]
      b = self.goals[self.path[1]]
      text_file.write(world % tuple(self.goals[self.start] + [180*atan2((b[1]-a[1]),(b[0]-a[0]))/pi]))
      text_file.close()
      self.pause = True
      self.msg = String()
      time.sleep(2)
      self.goal_callback(self.msg)
      self.msg.data = str(self.path)
      self.goal_num_pub.publish(self.msg)
      print 'Ready, start Stage ROS then DAE_node.py.'

  def createDict(self):
      self.distances = dict()
      for ii in self.goals:
          a = self.goals[ii]
          self.distances[ii] = dict()
          for jj in self.next_goal[ii]:
              b = self.goals[jj]
              self.distances[ii][jj] = sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

  def goal_callback(self,goal):
      if goal.data == 's':
          self.pause = False
      elif goal.data == 'c':
          self.msg.data = 'c'
          self.goal_pub.publish(self.msg)
          rospy.signal_shutdown('collision')
          sys.exit("Collision")

      if not self.pause:
          self.get_new_goal()
          self.msg.data = str(self.goals[self.current_goal][0]) + ',' + str(self.goals[self.current_goal][1])
          self.goal_pub.publish(self.msg)
          print "\nNext Waypoint: ", self.current_goal, ' @ ', self.goals[self.current_goal]

  def get_new_goal(self):
      if self.count >= len(self.path):
          self.count -=1
          self.msg.data = 's'
          self.goal_pub.publish(self.msg)
          rospy.signal_shutdown('goal reached')
          sys.exit("Goal Reached")
      self.current_goal = self.path[self.count]
      self.count +=1

def main():

  rospy.init_node('goal_setter', anonymous=True)
  GS = goal_setter()

  rospy.spin()


if __name__ == '__main__':
  main()
