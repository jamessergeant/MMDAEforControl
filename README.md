# Multimodal Deep Autoencoders for Control of Mobile Robots
*James Sergeant, Research Assistant, Australian Centre for Robotic Vision, Queensland University of Technology, james.sergeant@qut.edu.au*

This repository contains the software related to two papers under review:

"[Multimodal Deep Autoencoders for Control of Mobile Robots](http://bit.ly/acraMMDAE "Multimodal Deep Autoencoders for Control of Mobile Robots")", submitted 31 Aug 2015 to the Australasian Conference on Robotics and Automation 2015.

"[Learned Goal-oriented Navigation for Mobile Robots Using Multimodal Deep Autoencoders](http://bit.ly/icraMMDAE "Learned Goal-oriented Navigation for Mobile Robots Using Multimodal Deep Autoencoders")", submitted 15 September 2015 to the International Conference on Robotics and Automation 2016.

###### Dependencies
This project relies on:
* A [forked version of PyLearn2](https://github.com/jamessergeant/pylearn2.git "Forked PyLearn2") as well as its dependencies including Theano.
* A [forked version of the Stage ROS node](https://github.com/jamessergeant/stage_ros.git "Forked Stage ROS node") which publishes collisions as a Bool message on topic /stall.
* ROS (currently only tested on jade) and Player Stage.

###### Datasets
If performing the training stages, obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb) and store in the Datasets folder.

###### Additional Information
This repository has only ever been tested on Ubuntu 14.04 and ROS jade and may not operate correctly on other flavours of Linux or versions of ROS.

Please note that the content is currently being edited for ease of use and will be used in the continued development of a fully learned goal-based navigation system based on various machine learning methods.

Feel free to contact James Sergeant at the email above if you are having any issues operating the system.

## Instructions

### Setup
1. Run `source setup.sh` from the repository's main directory. This will set the necessary environment variables.

### Training

#### Unsupervised Learning (RBMs)
1. Obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb) and place in Datasets folder.
2. From the command line, run `python trainRBMs.py`. Note this may take some time to complete.
3. The trained RBMs will be available in the RBMs folder and can be assessed with a variety of PyLearn2 tools.

#### Fine-Tune Training (Deep Autoencoders)
1. Obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb) and place in Datasets folder.
2. From the command line, run `python trainDAEs.py`.
3. The trained DAEs will be available in the DAEs folder and can be assessed with a variety of PyLearn2 tools.

### Operation

#### In Simulation
1. Start `roscore`.
2. If operating with the Player Stage simulator, use `rosrun stage_ros stage ros worlds/complex/world.world`. Other environments are available.
3. From the command line, run `python operate.py TYPE` where TYPE should be replaced with one of the following:
* For ACRA: MMAE-ZCI, MMAE-PGCI, SMAE
* For ICRA: ICRA
4. The system can be paused at anytime by pressing Enter.

#### On a Pioneer P3-DX
1. Connect to the Pioneer's roscore.
2. Determine the laser scanners topic (e.g. /scan). This system currently only accepts 181 range measurements for a 180&deg; field of view.
3. Ensure the robot is located in a safe starting pose. Engage drive system.
4. From the command line, run `python operate.py MMAE-ZCI -t /scan` where other models (MMAE-PGCI, SMAE) can be used and /scan is the laser sensor topic. This loads the pretrained models supplied with the repository. To use alternate models, use `-m` and supply the filepath of the model.
5. The system can be paused at anytime by pressing Enter.
