# MMDAEforControl
James Sergeant
Research Assistant
Australian Centre for Robotic Vision
Queensland University of Technology
james.sergeant@qut.edu.au

DEPENDENCIES: This project relies on a forked version of PyLearn2 available at https://github.com/jamessergeant/pylearn2.git as well as it's dependencies. Other dependencies include ROS and Player Stage.

DATASETS: If performing the training stages, obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb)

Please note that the content is currently being edited for ease of use and will be used in the continued development of an end-to-end goal-based navigation system based on various machine learning methods.

Feel free to contact James Sergeant at the email above if you are having any issues operating the system.

This repository contains the software related to the paper "Multimodal Deep Autoencoders for Control of Mobile Robots" submitted 31 Aug 2015 to the Australasian Conference on Robotics and Automation 2015.

## Paper Abstract:
Robot navigation systems are typically engineered to suit certain platforms, sensing suites and environment types. In order to deploy a robot in an environment where its existing navigation system is insufficient, the system must be modified manually, often at significant cost. In this paper we address this problem, proposing a system based on multimodal deep autoencoders that enables a robot to learn how to navigate by observing a dataset of sensor input and motor commands collected while being teleoperated by a human. Low-level features and cross modal correlations are learned and used in initialising two different architectures with three operating modes. During operation, these systems exploit the learned correlations in generating suitable control signals based only on the sensor information. In simulation these systems are shown to run indefinitely in simple environments, are capable of generalising to more complex environments and found to be robust to significant amounts of sensor noise. The system has been deployed on a real robot and operated in a simple corridor for over 30 minutes without collision.

## Instructions

NOTE: If intending to use the existing pretrained model, avoid downloading the Datasets folder.

### Training

#### Unsupervised Learning (RBMs)
1. Obtain the datasets from http://bit.ly/MMDAEdata (521.14 Mb)
2. INCOMPLETE

#### Fine-Tune Training (Deep Autoencoders)
1. INCOMPLETE

### Operation
1. Start `roscore`
2. If operating on a Pioneer P3-DX, ensure the robot is located in a safe starting pose. Engage drive system.
3. If operating with the Player Stage simulator, use `rosrun stage_ros stage ros $MMDAEforControl/simulator/worlds/complex/world.world`
