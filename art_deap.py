######### IMPORTING PACKAGES

import random
import operator
import math
from deap import algorithms
from deap.algorithms import *
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
import pdb

######### INITIALISING GLOBAL VARIABLES

global img_dim
img_dim = 200
global n_stim
global stim_mat
global fc7_stim_mat
global conv2_stim_mat
global conv5_stim_mat
global n_ea 
n_ea = 300
global nRuns
global noise_inj
global noise_injector
noise_injector = 0
noise_inj = 25
global mRate
global cRate
nRuns= 200
mRate=0.25
cRate=0.5
global init_treesize_min
global init_treesize_max
init_treesize_min = 3
init_treesize_max = 8
global mut_treesize_min
global mut_treesize_max
mut_treesize_min = 0
mut_treesize_max = 3
global tourn_size
tourn_size = 3

######### CREATING REQUIRED FUNCTIONS

def sig_mod(x):
  return ((1 / (1 + math.exp(-x)))-0.5)*2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def pD(left, right):
    try:
        return sig_mod(left / right)
    except ZeroDivisionError:
        return 1

def pM(left, right):
    return (left * right)

def pA(left, right):
    return (left + right)/2.

def pS(left, right):
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
    #poke_ind = np.random.randint(n_stim)
    poke_ind = 21
    evaluator = (1./14211.)*img_sim[poke_ind,:] + (1./220.)*fc7_sim[poke_ind,:] + 0.25*(1./0.0027)*1./(1.*len(str(individual)))
    return evaluator,

def evalDum(offspring): 
    count = 0
    in_offspring = np.zeros([len(offspring),img_dim,img_dim])
    conv2_offspring = np.zeros([len(offspring),200704])
    conv5_offspring = np.zeros([len(offspring),43264])
    fc7_offspring = np.zeros([len(offspring),4096])
    evaluator_in = np.zeros([len(offspring)])
    evaluator_conv2 = np.zeros([len(offspring)])
    evaluator_conv5 = np.zeros([len(offspring)])
    evaluator_fc7 = np.zeros([len(offspring)])
    evaluator_len = np.zeros([len(offspring)])
    evaluator_pop_in = np.zeros([len(offspring)])
    evaluator_pop_conv2 = np.zeros([len(offspring)])
    evaluator_pop_conv5 = np.zeros([len(offspring)])
    evaluator_pop_fc7 = np.zeros([len(offspring)])
    empty_flag = np.ones([len(offspring)])
    for ind in offspring:
        FusedIm = eval(str(ind).replace('\'',''),{'__builtins__':None},dispatch)
        FusedIm = np.array(FusedIm)
        FusedIm[FusedIm<15] = 0
        FusedIm[FusedIm>15] = 20 
        FusedIm[FusedIm==0] = 255
        FusedIm[FusedIm==20] = 255
        FusedIm[FusedIm==15] = 0
        in_offspring[count,:,:] = FusedIm
        if np.sum(FusedIm) == 0:
            empty_flag[count] = 0
        im_inst = np.zeros([np.shape(FusedIm)[0],np.shape(FusedIm)[0],3])
        im_inst[:,:,0] = FusedIm
        im_inst[:,:,1] = FusedIm
        im_inst[:,:,2] = FusedIm
        dim_inst = np.shape(im_inst)[0]
        im_inst = scipy.misc.imresize(im_inst,227*1./dim_inst*1.)
        im_inst = im_inst - mean(im_inst)
        fc7_inst1 = sess.run(fc7_read, feed_dict = {x:[im_inst,im_inst]})
        fc7_inst = fc7_inst1[0,:]
        fc7_offspring[count,:] = fc7_inst
        conv2_inst1 = sess.run(conv2_in, feed_dict = {x:[im_inst,im_inst]})
        conv2_inst = conv2_inst1[0,:].flatten()
        conv2_offspring[count,:] = conv2_inst
        conv5_inst1 = sess.run(conv5_in, feed_dict = {x:[im_inst,im_inst]})
        conv5_inst = conv5_inst1[0,:].flatten()
        conv5_offspring[count,:] = conv5_inst
        img_sim = zeros([n_stim,1])
        conv2_sim = zeros([n_stim,1])
        conv5_sim = zeros([n_stim,1])
        fc7_sim = zeros([n_stim,1])
        for i in range(n_stim): 
            img_sim[i,0] = (np.sum((np.reshape(FusedIm,[1,img_dim*img_dim])-stim_mat[i,:])**2))**0.5
            conv2_sim[i,0] = (np.sum((conv2_inst-conv2_stim_mat[i,:])**2))**0.5
            conv5_sim[i,0] = (np.sum((conv5_inst-conv5_stim_mat[i,:])**2))**0.5
            fc7_sim[i,0] = (np.sum((fc7_inst-fc7_stim_mat[i,:])**2))**0.5
        #evaluator = np.min(img_sim) + np.min(fc7_sim) # can add novelty term here
        poke_ind = np.random.randint(n_stim)
        #poke_ind = 21
        evaluator_in[count] = img_sim[poke_ind,:]
        evaluator_conv2[count] = conv2_sim[poke_ind,:]
        evaluator_conv5[count] = conv5_sim[poke_ind,:]
        evaluator_fc7[count] = fc7_sim[poke_ind,:]
        evaluator_len[count] = 1./(1.*len(str(ind)))
        #evaluator.append((1./14211.)*img_sim[poke_ind,:] + (1./220.)*fc7_sim[poke_ind,:] + 0.25*(1./0.0027)*1./(1.*len(str(individual))))
        count = count + 1
    count = 0
    for ind in offspring:
        for i in range(len(offspring)):
            if count != i:
                evaluator_pop_in[count] = evaluator_pop_in[count] + (np.sum((np.reshape(in_offspring[count,:,:],[1,img_dim*img_dim])-np.reshape(in_offspring[i,:,:],[1,img_dim*img_dim]))**2))**0.5
                evaluator_pop_conv2[count] = evaluator_pop_conv2[count] + (np.sum((conv2_offspring[count,:]-conv2_offspring[i,:])**2))**0.5
                evaluator_pop_conv5[count] = evaluator_pop_conv5[count] + (np.sum((conv5_offspring[count,:]-conv5_offspring[i,:])**2))**0.5
                evaluator_pop_fc7[count] = evaluator_pop_fc7[count] + (np.sum((fc7_offspring[count,:]-fc7_offspring[i,:])**2))**0.5
        count = count + 1

    #pdb.set_trace()

    evaluator_in = evaluator_in/np.std(evaluator_in)
    evaluator_conv2 = evaluator_conv2/np.std(evaluator_conv2)
    evaluator_conv5 = evaluator_conv2/np.std(evaluator_conv5)
    evaluator_fc7 = evaluator_fc7/np.std(evaluator_fc7)
    evaluator_len = evaluator_len/np.std(evaluator_len)
    evaluator_pop_in = 1./(1.*evaluator_pop_in/(1.*(len(offspring)-1)))
    evaluator_pop_in = evaluator_pop_in/np.std(evaluator_pop_in)
    evaluator_pop_conv2 = 1./(1.*evaluator_pop_conv2/(1.*(len(offspring)-1)))
    evaluator_pop_conv2 = evaluator_pop_conv2/np.std(evaluator_pop_conv2)
    evaluator_pop_conv5 = 1./(1.*evaluator_pop_conv5/(1.*(len(offspring)-1)))
    evaluator_pop_conv5 = evaluator_pop_conv5/np.std(evaluator_pop_conv5)
    evaluator_pop_fc7 = 1./(1.*evaluator_pop_fc7/(1.*(len(offspring)-1)))
    evaluator_pop_fc7 = evaluator_pop_fc7/np.std(evaluator_pop_fc7)

    #evaluator1 = 1./4.*(evaluator_in + evaluator_conv2 + evaluator_conv5 + evaluator_fc7) 
    #+ 1./4.*(evaluator_pop_in + evaluator_pop_conv2 + evaluator_pop_conv5 + evaluator_pop_fc7)
    #+ 2.*evaluator_len # mixing fitnesses

    #evaluator1 = 1./2.*(evaluator_in + evaluator_fc7) 
    #+ 1./2.*(evaluator_pop_in + evaluator_pop_fc7)
    #+ evaluator_len # mixing fitnesses
    evaluator1 = 0.*evaluator_in

    #evaluator1 = evaluator_in + evaluator_fc7 + evaluator_len + 2*evaluator_pop_in + 2*evaluator_pop_fc7 # mixing fitnesses
    #evaluator = evaluator.tolist()
    
    evaluator = []
    for i in range(len(offspring)):
        dum_hs = np.random.random(1)[0]
        if dum_hs < 0.25:
            dum_hs1 = np.random.random(1)[0]
            if dum_hs1 < 0.33:
                evaluator1[i] = evaluator_in[i]
            elif dum_hs1 < 0.66:
                evaluator1[i] = evaluator_conv5[i]
            else:
                evaluator1[i] = evaluator_fc7[i]
        elif dum_hs < 0.75:
            dum_hs1 = np.random.random(1)[0]
            if dum_hs1 < 0.33:
                evaluator1[i] = evaluator_pop_in[i]
            elif dum_hs1 < 0.66:
                evaluator1[i] = evaluator_pop_conv5[i]
            else:
                evaluator1[i] = evaluator_pop_fc7[i]
        else:
            evaluator1[i] = evaluator_len[i]
        if empty_flag[i] == 0:
            evaluator1[i] = 10.
        evaluator.append((np.array([evaluator1[i]]),))

    #pdb.set_trace()

    return evaluator

def eaSimple1(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    fitnesses = evalDum(invalid_ind)
    #pdb.set_trace()
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        #pdb.set_trace()
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = evalDum(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if noise_injector == 1:
            if gen % noise_inj == 0:
                pop2 = toolbox.population(n=len(offspring)/4)
                invalid_ind2 = [ind for ind in pop2 if not ind.fitness.valid]
                fitnesses2 = evalDum(invalid_ind2)
                for ind, fit in zip(invalid_ind2, fitnesses2):
                    ind.fitness.values = fit
                bothpops = offspring + pop2
                offspring = bothpops
                del pop2, bothpops

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

    return population, logbook

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
    pop, log = eaSimple1(pop, toolbox, cRate, mRate, nRuns, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

######## REGISTERING REQUIRED FUNCTIONS FOR GRAPHICS UNIT

dispatch ={'Tx':Tx,'Ty':Ty,'R0':R0,'Sx':Sx,'Sy':Sy,'SF':SF,'TF':TF,'OC':OC,'P':P,'pA':pA,'pS':pS,
'pM':pM,'pD':pD,'C':C}

######## EA STUF

# PRIMITIVES

pset = gp.PrimitiveSetTyped("main", [], str) 
pset.addPrimitive(Tx, [str, float], str)
pset.addPrimitive(Ty, [str, float], str)
pset.addPrimitive(R0, [str, float], str)
pset.addPrimitive(Sx, [str, float], str)
pset.addPrimitive(Sy, [str, float], str)
pset.addPrimitive(SF, [str, str], str)
pset.addPrimitive(TF, [str, str], str)
pset.addPrimitive(OC, [str, str], str)

pset.addPrimitive(pA, [float,float], float)
pset.addPrimitive(pS, [float,float], float)
#pset.addPrimitive(pM, [float,float], float)
#pset.addPrimitive(pD, [float,float], float)

# TERMINALS

pset.addTerminal("P(2)",str)
pset.addTerminal("P(3)",str)
pset.addTerminal("P(4)",str)
pset.addTerminal("P(5)",str)
pset.addTerminal("P(6)",str)
pset.addTerminal("C()",str)

for i in np.linspace(0,1,50):
    pset.addTerminal(i,float)

# EA initialisation

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=init_treesize_min, max_=init_treesize_max)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalDummy)
#toolbox.register("")
toolbox.register("select", tools.selTournament,tournsize=tourn_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=mut_treesize_min, max_=mut_treesize_max)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

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
    conv2_stim_mat = np.zeros([n_stim,200704])
    conv5_stim_mat = np.zeros([n_stim,43264])
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
        conv2_inst = sess.run(conv2_in, feed_dict = {x:[im_inst,im_inst]})
        conv2_stim_mat[i,:] = conv2_inst[0,:].flatten()
        conv5_inst = sess.run(conv5_in, feed_dict = {x:[im_inst,im_inst]})
        conv5_stim_mat[i,:] = conv5_inst[0,:].flatten()

    print('Done with pokemon intialisation')

    ## RUN THE EA AND OUPUT STATS AND IMAGES

    pop, log, hof = main()
    count = 0
    for i in pop:
        count = count + 1
        #print(i)
        FusedIm = eval(str(i).replace('\'',''),{'__builtins__':None},dispatch)
        FusedIm = np.array(FusedIm)
        FusedIm[FusedIm<15] = 0
        FusedIm[FusedIm>15] = 20 
        FusedIm[FusedIm==0] = 255
        FusedIm[FusedIm==20] = 255
        FusedIm[FusedIm==15] = 0
        str_h = 'run_full/'+str(count)+'.png'
        scipy.misc.imsave(str_h,FusedIm)
