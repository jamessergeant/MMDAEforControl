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

class goal_setter:
  def __init__(self):
      self.goal_pub = rospy.Publisher("/goal", String, queue_size=10)
      self.goal_num_pub = rospy.Publisher("/goal_number", String, queue_size=10)
      self.subscriber = rospy.Subscriber("/goal_reached", String, self.goal_callback)
      self.goals = {1: [-18.5,18.5], 2: [-11,18.5], 3: [-3.5,18.5], 4: [3.5,18.5], 5: [11,18.5], 6: [18.5,18.5], 7: [-18.5,10.5], 8: [-11,10.5], 9: [-3.5,10.5], 10: [3.5,10.5], 11: [11,10.5], 12: [18.5,10.5], 13: [-18.5,3.5], 14: [-11,3.5], 15: [-3.5,3.5], 16: [3.5,3.5], 17: [11,3.5], 18: [18.5,3.5], 19: [-18.5,-3.5], 20: [-11,-3.5], 21: [-3.5,-3.5], 22: [3.5,-3.5], 23: [11,-3.5], 24: [18.5,-3.5], 25: [-18.5,-10.5], 26: [-11,-10.5], 27: [-3.5,-10.5], 28: [3.5,-10.5], 29: [11,-10.5], 30: [18.5,-10.5], 31: [-18.5,-18.5], 32: [-11,-18.5], 33: [-3.5,-18.5], 34: [3.5,-18.5], 35: [11,-18.5], 36: [18.5,-18.5]}
      self.next_goal = {1: [2,7],2: [1,3,8],3: [2,4,9],4: [3,5,10],5: [4,6,11],6: [5,12],7:[1,8,13],8: [2,7,9,14],9: [3,8,10,15],10: [4,9,11,16],11: [5,10,12,17],12: [6,11,18],13: [7,14,19],14: [8,13,15,20],15: [9,14,16,21],16: [10,15,17,22],17: [11,16,18,23],18: [12,17,24],19: [13,20,25],20: [14,19,21,26], 21: [15,20,22,27], 22:[16,21,23,28], 23:[17,22,24,29], 24:[18,23,30], 25:[19,26,31], 26: [20,25,27,32], 27: [21,26,28,33], 28: [22,27,29,34],29:[23,28,30,35], 30:[24,29,36], 31:[25,32], 32:[26,31,33],33:[27,32,34],34: [28,33,35], 35: [29,34,36], 36:[30,35]}
      l = len(self.goals)
      self.createDict()
      self.path = []
      random.seed()
      while len(self.path) < 6:
          self.start = random.randint(1,l)
          self.end = self.start
          while self.end == self.start:
              self.end = random.randint(1,l)
        #   self.start = 10
        #   self.end = 22
          self.path = shortestPath(self.distances,self.start,self.end)
      self.current_goal = self.start
      print "Start: %i,\tEnd:%i" % (self.start,self.end)
      print "Path: ", self.path
      self.count = 1
      world = open('/home/james/Dropbox/University/BEB801/PlayerStage/worlds/grid/world.world').read()
      text_file = open('/home/james/Dropbox/University/BEB801/PlayerStage/worlds/grid/pos_world.world', "w")
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
    #   self.msg.data = str(self.current_goal)
    #   self.goal_num(self.msg)
      print 'Ready, start Stage and all_node.py and hit enter.'

  def createDict(self):
      self.distances = dict()
      for ii in self.goals:
          a = self.goals[ii]
          self.distances[ii] = dict()
          for jj in self.next_goal[ii]:
              b = self.goals[jj]
              self.distances[ii][jj] = sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

  def goal_callback(self,goal):
    #   print "setter goal"
      if goal.data == 's':
          self.pause = False
      elif goal.data == 'c':
          self.msg.data = 'c'
          self.goal_pub.publish(self.msg)
          rospy.signal_shutdown('collision')
          sys.exit("Collision")
    #   self.msg.data = str(self.current_goal)
    #   self.goal_num_pub.publish(self.msg)

      if not self.pause:
          self.get_new_goal()
          self.msg.data = str(self.goals[self.current_goal][0]) + ',' + str(self.goals[self.current_goal][1])
          self.goal_pub.publish(self.msg)
          print "\nNext Waypoint: ", self.current_goal, ' @ ', self.goals[self.current_goal]

  def get_new_goal(self):
    #   print "new_goal"
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
