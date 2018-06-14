######### IMPORTING PACKAGES

import random
import operator
import math
from deap import algorithms
from deap import base
from deap import creator 
from deap import tools
from deap import gp
from numpy import *
import os
import os.path
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import scipy.misc
import tensorflow as tf
from caffe_classes import class_names
from graphics_deap import *

######### INITIALISING GLOBAL VARIABLES

global img_dim
img_dim = 200
global n_stim
global stim_mat
global fc7_stim_mat
global n_ea 
n_ea = 200
global nRuns
global mRate
global cRate
nRuns= 200
mRate=0.25
cRate=0.5
global treesize_min
global treesize_max
treesize_min = 1
treesize_max = 10
global tourn_size
tourn_size = 20

######### CREATING REQUIRED FUNCTIONS

def sig_mod(x):
  return ((1 / (1 + math.exp(-x)))-0.5)*2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def protectedDiv(left, right):
    try:
        return sig_mod(left / right)
    except ZeroDivisionError:
        return 1

def protectedMult(left, right):
    return (left * right)

def protectedAdd(left, right):
    return (left + right)/2.

def protectedSubt(left, right):
    return ((left - right)+1.)/2.

def evalDummy(individual): 
    FusedIm = eval(str(individual).replace('\'',''),{'__builtins__':None},dispatch)
    FusedIm = np.array(FusedIm)
    FusedIm[FusedIm<15] = 0
    FusedIm[FusedIm>15] = 20 
    FusedIm[FusedIm==0] = 255
    FusedIm[FusedIm==20] = 255
    FusedIm[FusedIm==15] = 0
    im_inst = np.zeros([np.shape(FusedIm)[0],np.shape(FusedIm)[0],3])
    im_inst[:,:,0] = FusedIm
    im_inst[:,:,1] = FusedIm
    im_inst[:,:,2] = FusedIm
    dim_inst = np.shape(im_inst)[0]
    im_inst = scipy.misc.imresize(im_inst,227*1./dim_inst*1.)
    fc7_inst1 = sess.run(fc7_read, feed_dict = {x:[im_inst,im_inst]})
    fc7_inst = fc7_inst1[0,:]
    img_sim = zeros([n_stim,1])
    fc7_sim = zeros([n_stim,1])
    for i in range(n_stim): 
        img_sim[i,0] = (np.sum((np.reshape(FusedIm,[1,img_dim*img_dim])-stim_mat[i,:])**2))**0.5
        fc7_sim[i,0] = (np.sum((fc7_inst-fc7_stim_mat[i,:])**2))**0.5
    #evaluator = np.min(img_sim) + np.min(fc7_sim) # can add novelty term here
    poke_ind = np.random.randint(n_stim)
    #evaluator = img_sim[poke_ind,:]*1./56. + fc7_sim[poke_ind,:] # can add novelty term here
    #evaluator = fc7_sim[poke_ind,:] # can add novelty term here
    evaluator = fc7_sim[poke_ind,:] + 1./2.*250./0.015*1./(1.*len(str(individual))) # can add novelty term here, one-half the influence of string length
    #evaluator = 1./(1.*len(str(individual)))
    return evaluator,

def main():

    #random.seed(319)

    pop = toolbox.population(n=n_ea)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cRate, mRate, nRuns, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

######## REGISTERING REQUIRED FUNCTIONS FOR GRAPHICS UNIT

dispatch ={'Tx':Tx,'Ty':Ty,'R':R,'Sx':Sx,'Sy':Sy,'SF':SF,'TF':TF,'OCCL':OCCL,'P':P,'protectedAdd':protectedAdd,'protectedSubt':protectedSubt,
'protectedMult':protectedMult,'protectedDiv':protectedDiv,'Circle':Circle}

######## EA STUF

# PRIMITIVES

pset = gp.PrimitiveSetTyped("main", [], str) 
pset.addPrimitive(Tx, [str, float], str)
pset.addPrimitive(Ty, [str, float], str)
pset.addPrimitive(R, [str, float], str)
pset.addPrimitive(Sx, [str, float], str)
pset.addPrimitive(Sy, [str, float], str)
pset.addPrimitive(SF, [str, str], str)
pset.addPrimitive(TF, [str, str], str)
pset.addPrimitive(OCCL, [str, str], str)

#pset.addPrimitive(protectedAdd, [float,float], float)
#pset.addPrimitive(protectedSubt, [float,float], float)
pset.addPrimitive(protectedMult, [float,float], float)
#pset.addPrimitive(protectedDiv, [float,float], float)

# TERMINALS

pset.addTerminal("P(2)",str)
pset.addTerminal("P(3)",str)
pset.addTerminal("P(4)",str)
pset.addTerminal("P(5)",str)
pset.addTerminal("P(6)",str)
pset.addTerminal("Circle()",str)

for i in np.linspace(0,1,50):
    pset.addTerminal(i,float)

# EA initialisation

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=treesize_min, max_=treesize_max)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalDummy)
toolbox.register("select", tools.selTournament,tournsize=tourn_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=treesize_min, max_=treesize_max)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("bloatcont", gp.staticLimit, key=operator.attrgetter('height'), max_value=17)

######### MAIN EXECUTION

if __name__ == "__main__":

    ## NN intialisation - NN source: https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]

    if os.path.isfile("bvlc_alexnet.npy"):
    	net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    	print('Model intialised succesfully')
    else:
    	print('Model not found. Beginning file download with urllib2...')
    	url = 'https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
    	urllib.urlretrieve(url, 'bvlc_alexnet.npy') 
    	print('Model succesfully downloaded')

    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

    x = tf.placeholder(tf.float32, (None,) + xdim)

    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7_read = tf.nn.xw_plus_b(fc6, fc7W, fc7b)
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    prob = tf.nn.softmax(fc8)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    ## Initialise pokemon images and FC7 activations

    input_dir = [file for file in os.listdir('stimuli/pokemon-images-processed') if file.endswith('.png')]
    n_stim = len(input_dir)

    stim_mat = np.zeros([n_stim,img_dim*img_dim])
    fc7_stim_mat = np.zeros([n_stim,4096])
    for i in range(n_stim):
        im_inst = scipy.misc.imread('stimuli/pokemon-images-processed/'+input_dir[0])
        dim_inst = np.shape(im_inst)[0]
        im_inst = scipy.misc.imresize(im_inst,img_dim*1./dim_inst*1.)
        stim_mat[i,:] = np.reshape(im_inst,[1,img_dim*img_dim])
        im_inst = scipy.misc.imread('stimuli/pokemon-images-processed/'+input_dir[0],mode='RGB')
        dim_inst = np.shape(im_inst)[0]
        im_inst = (scipy.misc.imresize(im_inst,227*1./dim_inst*1.)).astype('float32')
        im_inst = im_inst - mean(im_inst)
        fc7_inst = sess.run(fc7_read, feed_dict = {x:[im_inst,im_inst]})
        fc7_stim_mat[i,:] = fc7_inst[0,:]

    print('Done with pokemon intialisation')

    ## RUN THE EA AND OUPUT STATS AND IMAGES

    pop, log, hof = main()
    for i in pop:
        #print(i)
        FusedIm = eval(str(i).replace('\'',''),{'__builtins__':None},dispatch)
        FusedIm = np.array(FusedIm)
        FusedIm[FusedIm<15] = 0
        FusedIm[FusedIm>15] = 20 
        FusedIm[FusedIm==0] = 255
        FusedIm[FusedIm==20] = 255
        FusedIm[FusedIm==15] = 0
        FusedIm = Image.fromarray(FusedIm)
        FusedIm.show()
