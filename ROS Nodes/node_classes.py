import os
from pylearn2.utils import serial
import re
import numpy as N
from pylearn2.config import yaml_parse
import string
from theano import function
from math import sqrt,pi,atan2


class Model:

    def __init__(self,model):

        self.filename = model['filename']
        try:
            try:
                self.model = serial.load(self.filename)
            except:
                self.filename = os.environ['MMDAEdaes'] + self.filename
                self.model = serial.load(self.filename)
        except Exception as e:
            print("error loading {}:".format(self.filename))
            print(e)
            return False

        if not hasattr(self.model, 'sequence'):
            self.__extract_sequence()

        if not hasattr(self.model, 'mean') or not hasattr(self.model, 'std') or not hasattr(self.model, 'normalise'):
            self.__calc_mean_std()

        if not hasattr(self.model, 'function'):
            self.__create_function()

        if not hasattr(self.model, 'mmdae_type'):
            self.__determine_type()

        self.__clean()

    def __determine_type(self):
        self.model.dims = [comp.dim for comp in self.model.get_input_space().components]

        in_dims = self.model.dims
        self.model.mmdae_type = list()

        laser = True if self.model.sequence*181 in in_dims else False

        in_dims_total = N.asarray(in_dims).sum().item()
        out_dims = self.model.get_output_space().dim

        goal = True if in_dims.count(self.model.sequence*2) == 2 or (in_dims.count(self.model.sequence*2) == 1 and in_dims_total != out_dims) else False

        command = in_dims_total == out_dims

        self.model.mmdae_type = [laser,goal,command]

        serial.save(self.filename,self.model)

    def __extract_sequence(self):

        self.model.sequence = 1

        sequence = self.model.dataset_yaml_src.split('sequence:')

        if len(sequence) > 1:
            self.model.sequence = int(sequence[1].split(',')[0])

        serial.save(self.filename,self.model)

    def __calc_mean_std(self):

        self.__load_data()

        self.model.mean = list()
        self.model.std = list()

        for d,n in zip(self.data,self.model.normalise):
            if n == 1:
                self.model.mean.append(d[self.start:self.stop,:].mean())
                self.model.std.append(d[self.start:self.stop,:].std())
            elif n == 2:
                self.model.mean.append(d[self.start:self.stop,:].mean(axis=0))
                self.model.std.append(d[self.start:self.stop,:].std(axis=0))

        assert len(self.model.mean) == len(self.model.std)

        serial.save(self.filename,self.model)

    def __load_data(self):

        self.model.normalise = self.model.dataset_yaml_src.split('normalise: [')[1]
        self.model.normalise = self.model.normalise.split(']')[0]
        self.model.normalise = re.split(',|\n| ', self.model.normalise)
        self.model.normalise = [int(norm) for norm in self.model.normalise if len(norm) > 0]

        datalist = self.model.dataset_yaml_src.split('datasets: [')[1]
        datalist = datalist.split(']')[0]
        datalist = re.split(',|\n| ', datalist)
        datalist = [data for data in datalist if len(data) > 1]

        self.data = list()

        for data in datalist:
            d = data.split('Datasets/')[1]
            self.data.append(N.load(os.environ['MMDAEdata'] + d))

        # find start and stop used for training
        self.start = 0
        self.stop = self.data[0].shape[0]
        s = self.model.dataset_yaml_src.split('start: ')
        if len(s) > 1:
            s = s[1].split(',')
            self.start = int(float(s[0]))

        s = self.model.dataset_yaml_src.split('stop: ')
        if len(s) > 1:
            s = re.split(',|}',s[1])
            self.stop = int(float(s[0]))

    def __clean(self):

        if hasattr(self,'data'):
            del self.data
        if hasattr(self,'start'):
            del self.start
        if hasattr(self,'end'):
            del self.end

    def __create_function(self):

        X = self.model.get_input_space().make_theano_batch()
        Y = self.model.fprop(X)

        self.model.function = function(X,Y)

    def process(self,laser,goal,command):

        inputs = list()

        if self.model.mmdae_type[0]:
            inputs.append(laser)
        if self.model.mmdae_type[1]:
            inputs.append(goal)
        if self.model.mmdae_type[2]:
            inputs.append(command)

        if len(inputs) == 3:
            self.y = self.model.function(inputs[0].data,inputs[1].data,inputs[2].data)
        elif len(inputs) == 2:
            self.y = self.model.function(inputs[0].data,inputs[1].data)
        elif len(inputs) == 1:
            self.y = self.model.function(inputs[0].data)

        return self.y[0][-2:] * self.model.std[-1] + self.model.mean[-1]

class SequenceInput:

    def __init__(self,length,mean=0,std=1,sequence=1,normalise=0):
        self.data = N.zeros((1,length))
        self.sequence = sequence
        self.normalise = normalise
        self.length = length
        self.mean = mean
        self.std = std

    def update(self,new_data):

        data = new_data.copy()

        if len(data.shape) < 2:
            data = data.reshape(1,data.shape[0])

        if self.normalise != 0:
            data -= self.mean
            data /= self.std

        self.data = N.concatenate((self.data[:,int(self.length/self.sequence):],data),axis=1)

class GoalManager:

    def __init__(self,settings):
        self.current_goal = N.zeros((2))
        self.current_pose = N.zeros((3))
        self.relative_goal = N.zeros((2))
        try:
            self.tolerance = float(settings.goals['tolerance'])
        except:
            self.tolerance = 2.0
        try:
            self.aim_next = settings.goals['aim_next'] == 'True'
        except:
            self.aim_next = False
        self.first_goal = False
        self.reset()

    def reset(self):
        self.last_distance = self.tolerance
        self.within_goal = False
        self.closest = False
        self.next_goal_flag = False
        self.reorient = self.aim_next

    def update_pose(self,pose):
        if type(pose) == type([]):
            pose = N.asarray(pose)
        self.current_pose = pose

    def update_relative(self):

        distance = sqrt((self.current_pose[0]-self.current_goal[0])**2 + (self.current_pose[1]-self.current_goal[1])**2)
        angle = atan2(self.current_goal[1]-self.current_pose[1],self.current_goal[0]-self.current_pose[0]) - self.current_pose[2]
        angle = angle if angle < pi else angle - 2*pi
        angle = angle if angle > -pi else angle + 2*pi

        self.relative_goal = N.asarray((distance,angle))

        if distance < self.tolerance:
            self.within_goal = True
        else:
            self.within_goal = False

        if self.within_goal and (distance > self.last_distance or distance < 0.5*self.tolerance):
            self.closest = True

        if self.within_goal and self.closest:
            self.next_goal_flag = True

        self.last_distance = distance if distance < self.last_distance else self.last_distance

        if self.reorient and abs(angle) < N.deg2rad(5).item():
            self.reorient = False


    def update_goal(self,goal):

        if type(goal) == type([]):
            goal = N.asarray(goal)
        self.current_goal = goal
        if not self.first_goal:
            self.first_goal = True
        self.reset()
