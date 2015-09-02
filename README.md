# Multimodal Deep Autoencoders for Control of Mobile Robots
*James Sergeant, Research Assistant, Australian Centre for Robotic Vision, Queensland University of Technology, james.sergeant@qut.edu.au*

This repository contains the software related to the paper "Multimodal Deep Autoencoders for Control of Mobile Robots" submitted 31 Aug 2015 to the Australasian Conference on Robotics and Automation 2015. This paper can be viewed at http://bit.ly/MMDAEpaper.

###### Dependencies
* This project relies on a forked version of PyLearn2 available at https://github.com/jamessergeant/pylearn2.git as well as it's dependencies including Theano.
* ROS and Player Stage.

###### Datasets
If performing the training stages, obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb) and store in the Datasets folder.

###### Additional Information
Please note that the content is currently being edited for ease of use and will be used in the continued development of an end-to-end goal-based navigation system based on various machine learning methods.

Feel free to contact James Sergeant at the email above if you are having any issues operating the system.

## Paper Abstract:
Robot navigation systems are typically engineered to suit certain platforms, sensing suites and environment types. In order to deploy a robot in an environment where its existing navigation system is insufficient, the system must be modified manually, often at significant cost. In this paper we address this problem, proposing a system based on multimodal deep autoencoders that enables a robot to learn how to navigate by observing a dataset of sensor input and motor commands collected while being teleoperated by a human. Low-level features and cross modal correlations are learned and used in initialising two different architectures with three operating modes. During operation, these systems exploit the learned correlations in generating suitable control signals based only on the sensor information. In simulation these systems are shown to run indefinitely in simple environments, are capable of generalising to more complex environments and found to be robust to significant amounts of sensor noise. The system has been deployed on a real robot and operated in a simple corridor for over 30 minutes without collision.

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
3. From the command line, run `python operate.py MMAE-ZCI` where other models (MMAE-PGCI, SMAE) can be used.
4. The system can be paused at anytime by pressing Enter.

#### On a Pioneer P3-DX
1. Connect to the Pioneer's roscore.
2. Determine the laser scanners topic (e.g. /scan). This system currently only accepts 181 range measurements for a 180&deg; field of view.
3. Ensure the robot is located in a safe starting pose. Engage drive system.
4. From the command line, run `python operate.py MMAE-ZCI -t /scan` where other models (MMAE-PGCI, SMAE) can be used and /scan is the laser sensor topic. This loads the pretrained models supplied with the repository. To use alternate models, use `-m` and supply the filepath of the model.
5. The system can be paused at anytime by pressing Enter.
