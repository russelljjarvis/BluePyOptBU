"""Optimisation class"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=R0912, R0914


import random
import logging
import functools
import numpy

import deap
import deap.base
import deap.algorithms
import deap.tools

from . import algorithms
from . import tools

import bluepyopt.optimisations

logger = logging.getLogger('__main__')

# TODO decide which variables go in constructor, which ones go in 'run' function
# TODO abstract the algorithm by creating a class for every algorithm, that way
# settings of the algorithm can be stored in objects of these classes


class WeightedSumFitness(deap.base.Fitness):

    """Fitness that compares by weighted sum"""

    def __init__(self, values=(), obj_size=None):
        self.weights = [-1.0] * obj_size if obj_size is not None else [-1]

        super(WeightedSumFitness, self).__init__(values)

    @property
    def weighted_sum(self):
        """Weighted sum of wvalues"""
        return sum(self.wvalues)

    @property
    def sum(self):
        """Weighted sum of values"""
        return sum(self.values)

    @property
    def norm(self):
        """Frobenius norm of values"""
        return numpy.linalg.norm(self.values)

    def __le__(self, other):
        return self.weighted_sum <= other.weighted_sum

    def __lt__(self, other):
        return self.weighted_sum < other.weighted_sum

    def __deepcopy__(self, _):
        """Override deepcopy"""

        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class WSListIndividual(list):

    """Individual consisting of list with weighted sum field"""

    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.fitness = WeightedSumFitness(obj_size=kwargs['obj_size'])
        del kwargs['obj_size']
        super(WSListIndividual, self).__init__(*args, **kwargs)


class DEAPOptimisation(bluepyopt.optimisations.Optimisation):

    """DEAP Optimisation class"""

    def get_trans_list(self,param_dict):
        trans_list = []
        for i,k in enumerate(list(param_dict.keys())):
            trans_list.append(k)
        return trans_list

    def setnparams(self, nparams = 10, provided_dict = None):
        from neuronunit.optimization import optimization_management
        self.params = optimization_management.create_subset(nparams = nparams,provided_dict = provided_dict)
        self.nparams = len(self.params)
        self.td = self.get_trans_list(self.params)
        return self.params, self.td


    def __init__(self, error_criterion = None, evaluator = None,
                 selection = 'selIBEA',
                 benchmark = False,
                 seed=1,
                 offspring_size=15,
                 elite_size=0,
                 eta=10,
                 mutpb=1.0,
                 cxpb=1.0,
                 map_function=None,
                 backend=None,
                 nparams = 10,
                 provided_dict= {}):
        """Constructor"""

        super(DEAPOptimisation, self).__init__()
        self.selection = selection
        self.benchmark = benchmark
        self.error_criterion = error_criterion
        self.seed = seed
        self.offspring_size = offspring_size
        self.elite_size = elite_size
        self.eta = eta
        self.cxpb = cxpb
        self.mutpb = mutpb
        #self.CPU_NUM = CPU_NUM
        self.backend = backend
        # Create a DEAP toolbox
        self.toolbox = deap.base.Toolbox()
        self.setnparams(nparams = nparams, provided_dict = provided_dict)

        self.setup_deap()



    def set_evaluate(self):
        if self.benchmark == True:
            self.toolbox.register("evaluate", benchmarks.zdt1)
        else:
            #from neuronunit.optimization.optimization_management import OptMan
            #optimization_management = OptMan()
            from neuronunit.optimization import optimization_management
            self.toolbox.register("evaluate", optimization_management.evaluate)

    def setup_deap(self):
        """Set up optimisation"""

        # Number of objectives
        #OBJ_SIZE = len(self.evaluator.objectives)

        # Set random seed
        random.seed(self.seed)

        # Eta parameter of crossover / mutation parameters
        # Basically defines how much they 'spread' solution around
        # The lower this value, the more spread
        ETA = self.eta

        # Number of parameters
        # Bounds for the parameters
        IND_SIZE = len(list(self.params.values()))
        OBJ_SIZE = len(self.error_criterion)
        import numpy as np
        LOWER = [ np.min(self.params[v]) for v in self.td ]
        UPPER = [ np.max(self.params[v]) for v in self.td ]
        for index, i in enumerate(UPPER):
            if i == LOWER[index]:
                LOWER[index]-=2.0
                i+=2.0

        import pdb; pdb.set_trace()
        # Define a function that will uniformly pick an individual
        def uniform(lower_list, upper_list, dimensions):
            """Fill array """

            if hasattr(lower_list, '__iter__'):
                return [random.uniform(lower, upper) for lower, upper in
                        zip(lower_list, upper_list)]
            else:
                return [random.uniform(lower_list, upper_list)
                        for _ in range(dimensions)]

        # Register the 'uniform' function
        self.toolbox.register("uniformparams", uniform, LOWER, UPPER, IND_SIZE)

        # Register the individual format
        # An indiviual is create by WSListIndividual and parameters
        # are initially
        # picked by 'uniform'
        self.toolbox.register(
            "Individual",
            deap.tools.initIterate,
            functools.partial(WSListIndividual, obj_size=OBJ_SIZE),
            self.toolbox.uniformparams)

        # Register the population format. It is a list of individuals
        self.toolbox.register(
            "population",
            deap.tools.initRepeat,
            list,
            self.toolbox.Individual)

        # Register the evaluation function for the individuals
        def custom_code(invalid_ind):
            from neuronunit.optimization.optimization_management import evaluate, map_wrapper, update_deap_pop

            #from neuronunit.optimization.optimization_management.OptMan import evaluate, map_wrapper, update_deap_pop
            #optimization_management = OptMan()
            #from neuronunit.optimization.optimization_management import evaluate, map_wrapper, update_deap_pop
            if self.backend is not None:
                return_package = list(update_deap_pop(invalid_ind, self.error_criterion, td = self.td, backend = self.backend))
            else:
                return_package = list(update_deap_pop(invalid_ind, self.error_criterion, td = self.td))
            invalid_dtc = [i[0] for i in return_package ]
            invalid_ind = [i[1] for i in return_package ]
            fitnesses = list(map(evaluate,invalid_dtc))
            return fitnesses

        self.toolbox.register("evaluate", custom_code)
        # Register the mate operator
        self.toolbox.register(
            "mate",
            deap.tools.cxSimulatedBinaryBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER)

        # Register the mutation operator
        self.toolbox.register(
            "mutate",
            deap.tools.mutPolynomialBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER,
            indpb=0.5)

        # Register the variate operator
        self.toolbox.register("variate", deap.algorithms.varAnd)

        # Register the selector (picks parents from population)
        if self.selection == str('ngsa'):
            self.toolbox.register("select", tools.selNSGA2)
        elif self.selection == str('selIBEA'):
            self.toolbox.register("select", tools.selIBEA)

    def run(self,
            max_ngen=25,
            offspring_size=None,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=0):
        """Run optimisation"""
        # Allow run function to override offspring_size
        # TODO probably in the future this should not be an object field anymore
        # keeping for backward compatibility
        if offspring_size is None:
            offspring_size = self.offspring_size

        # Generate the population object
        pop = self.toolbox.population(n=offspring_size)
        hof = deap.tools.HallOfFame(10)
        hof.update(pop)
        init_value = hof
        stats = deap.tools.Statistics(key=lambda ind: ind.fitness.sum)
        import numpy
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        pop, log, history, gen_vs_hof = algorithms.eaAlphaMuPlusLambdaCheckpoint(
            pop,
            self.toolbox,
            offspring_size,
            self.cxpb,
            self.mutpb,
            max_ngen,
            stats=stats,
            halloffame=hof,
            nelite=self.elite_size,
            cp_frequency=cp_frequency,
            continue_cp=continue_cp,
            cp_filename=cp_filename,
            selection = self.selection)

        # insert the initial HOF value back in.
        gen_vs_hof.insert(0,init_value)
        return pop, hof, log, history, self.td, gen_vs_hof


class IBEADEAPOptimisation(DEAPOptimisation):

    """IBEA DEAP class"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super(IBEADEAPOptimisation, self).__init__(*args, **kwargs)
