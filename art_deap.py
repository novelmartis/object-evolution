import random
import numpy as np
import operator


from deap import algorithms
from deap import base
from deap import creator 
from deap import tools
from deap import gp

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def Tx(a,b):
	return ('Tx('+a+','+str(b)+')')

def Ty(a,b):
	return ('Ty('+a+','+str(b)+')')

def R(a,b):
	return ('R('+a+','+str(b)+')')

def Sx(a,b):
	return ('Sx('+a+','+str(b)+')')

def Sy(a,b):
	return ('Sy('+a+','+str(b)+')')

def F(a,b):
	return ('F('+a+','+str(b)+')')

def SF(a,b):
	return ('SF('+a+','+b+')')

def TF(a,b):
	return ('SF('+a+','+b+')')

def OCCL(a,b):
	return ('SF('+a+','+b+')')



pset = gp.PrimitiveSetTyped("main", [], str) 
pset.addPrimitive(Tx, [str, float], str)
pset.addPrimitive(Ty, [str, float], str)
pset.addPrimitive(R, [str, float], str)
pset.addPrimitive(Sx, [str, float], str)
pset.addPrimitive(Sy, [str, float], str)
pset.addPrimitive(F, [str, float], str)



pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)

#object combinations
pset.addPrimitive(SF, [str, str], str)
pset.addPrimitive(TF, [str, str], str)
pset.addPrimitive(OCCL, [str, str], str)

pset.addTerminal("P2",str)
pset.addTerminal("P3",str)
pset.addTerminal("P4",str)
pset.addTerminal("P5",str)
pset.addTerminal("P6",str)
pset.addTerminal("P20",str)
'''
for i in range (50):
    pset.addTerminal(random.random(),float)
'''
#np.linspace(0,1,50)
for i in np.linspace(0,1,50):
	pset.addTerminal(i,float)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("compile", gp.compile, pset=pset)

def evalDummy(individual): # call to drawing script
    return 1,

toolbox.register("evaluate", evalDummy)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
'''
expr = gp.genHalfAndHalf(pset, min_=1, max_=5)
tree = gp.PrimitiveTree(expr)
str(tree)
'''
def main():
    random.seed(319)

    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 2, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
	pop, log, hof = main()
	for i in pop:
		print(i)
