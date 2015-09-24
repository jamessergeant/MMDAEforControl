# Multimodal Deep Autoencoders for Control of Mobile Robots
*James Sergeant, Research Assistant, Australian Centre for Robotic Vision, Queensland University of Technology, james.sergeant@qut.edu.au*

###### PLEASE NOTE THAT THIS REPOSITORY IS CURRENTLY UNDERGOING AN UPDATE, PLEASE CHECK BACK AFTER 30 SEPTEMBER 2015

This repository contains the software related to two papers under review:

"[Multimodal Deep Autoencoders for Control of Mobile Robots](http://bit.ly/acraMMDAE "Multimodal Deep Autoencoders for Control of Mobile Robots")", submitted 31 Aug 2015 to the Australasian Conference on Robotics and Automation 2015.

"[Learned Goal-oriented Navigation for Mobile Robots Using Multimodal Deep Autoencoders](http://bit.ly/icraMMDAE "Learned Goal-oriented Navigation for Mobile Robots Using Multimodal Deep Autoencoders")", submitted 15 September 2015 to the International Conference on Robotics and Automation 2016.

###### Dependencies
This project relies on:
* A [forked version of PyLearn2](https://github.com/jamessergeant/pylearn2.git "Forked PyLearn2") as well as its dependencies including Theano.
* A [forked version of the Stage ROS node](https://github.com/jamessergeant/stage_ros.git "Forked Stage ROS node") which publishes collisions as a Bool message on topic /stall.
* ROS (currently only tested on ROS Jade).

It also relies on [Dijkstra's algorithm for shortest paths, David Eppstein, UC Irvine, 4 April 2002](http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228) and [Priority dictionary using binary heaps, David Eppstein, UC Irvine, 8 Mar 2002](http://code.activestate.com/recipes/117228-priority-dictionary/) which are included.

###### Datasets
If performing the training stages, obtain the [datasets](http://bit.ly/MMDAEdata "MMDAE datasets") (521.14 Mb) and store in the Datasets folder.

###### Additional Information
This repository has only ever been tested on Ubuntu 14.04 and ROS jade and may not operate correctly on other flavours of Linux or versions of ROS.

Please note that the content is currently being edited for ease of use and will be used in the continued development of a fully learned goal-based navigation system based on various machine learning methods.

Feel free to contact James Sergeant at the email above if you are having any issues operating the system.

## Instructions

### Setup
1. Ensure the dependencies are installed.
2. Run `. ./setup.sh` from the repository's main directory. This will set the necessary environment variables.

### Training

The training stage can be bypassed by obtaining the [pretrained DAEs](http://linkdoesntexistyet.com "Pretrained DAEs") (FILE SIZE Mb). Place file in the DAEs folder and run `python $MMDAEdaes/extract_daes.py`.

#### Unsupervised Learning (RBMs)
1. Obtain the [datasets](http://bit.ly/MMDAEdata "MMDAE datasets") (551.2 Mb). Place file in Datasets folder and run `python $MMDAEdata/extract_datasets.py`.
2. From the command line, run:
  * `python Training/trainLaser.py`
  * `python Training/trainCommand.py`
  * `python Training/trainGoal.py` (ICRA only)
3. The trained RBMs will be available in the RBMs folder and can be assessed with a variety of PyLearn2 tools.

#### Fine-Tune Training (Deep Autoencoders)
1. Obtain the [datasets](http://bit.ly/MMDAEdata "MMDAE datasets") (551.2 Mb). Place file in Datasets folder and run `python $MMDAEdata/extract_datasets.py`.
2. From the command line, run:
  * `python Training/trainLCMMAE.py`
  * `python Training/trainLCSMAE.py`
  * `python Training/trainGCSMAE.py` (ICRA only)
3. The trained DAEs will be available in the DAEs folder and can be assessed with a variety of PyLearn2 tools.

### Operation

#### In Simulation
1. Start `roscore`.
2. For ACRA:
  * `rosrun stage_ros stageros $MMDAE/Worlds/complex/pos_world.world` (edit the initial position in the pos_world.world file as desired)
  * `python ROS_nodes/DAE_node.py ROS_nodes/acra.xml`
3. For ICRA:
  * `python Planner/goal_setter_grid.py`, wait until ready
  * `rosrun stage_ros stageros $MMDAE/Worlds/grid/pos_world.world`
  * `python ROS_nodes/DAE_node.py ROS_nodes/icra.xml`

#### On a Pioneer P3-DX
1. Connect to the Pioneer's roscore.
2. Determine the laser sensor topic (e.g. /scan). This system currently only accepts 181 range measurements for a 180&deg; field of view.
3. Ensure the robot is located in a safe starting pose. Engage drive system.
4. From the command line, run (as appropriate):
  * `python ROS_nodes/DAE_node.py ROS_nodes/acra.xml -t /scan`
  * `python ROS_nodes/DAE_node.py ROS_nodes/icra.xml -t /scan`
  To use alternate models, ensure the model pkl files are in the DAEs folder and edit the acra.xml or icra.xml files as appropriate.
5. For the ICRA system:
  * the ROS module map_server must be publishing a map of the area
  * the ROS module AMCL must be publishing the robot's position on the topic /amcl_pose relative to the world frame
  * before sending the first goal, ensure the robot is suitably localised within the environment (this can be visualised in RVIZ)
  * to provide goals to the system, publish the x,y coordinates of the goal (in the world frame) as a comma delimited string on the /goal topic
