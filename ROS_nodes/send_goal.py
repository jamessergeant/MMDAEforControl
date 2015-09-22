
import sys, argparse, rospy
from std_msgs.msg import String
import time
import re


class send_goal:

  def __init__(self):

    self.cmd_pub = rospy.Publisher("/goal", String, queue_size=10)
    self.msg = String()
    self.string_check = re.compile('.*,.*')

  def send_msg(self):
    self.cmd_pub.publish(self.msg)

def main():

  rospy.init_node('send_goal', anonymous=True)

  GOAL = send_goal()

  while GOAL.msg.data != 's':

    # Obtain goal from user
    GOAL.msg.data = raw_input("Enter a x,y goal (or q to quit): ")

    # If quitting, shutdown the DAE_node by sending 's'
    if GOAL.msg.data == 'q':
        GOAL.msg.data = 's'

    # Else, check format of input
    elif GOAL.string_check.match(GOAL.msg.data) is None:
        print "Invalid goal"
        continue

    # Send goal
    GOAL.send_msg()

if __name__ == '__main__':
  main()
