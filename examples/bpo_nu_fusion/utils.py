import dask
def dask_map_function(eval_,invalid_ind):
    results = []
    for x in invalid_ind:
        y = dask.delayed(eval_)(x)
        results.append(y)
    fitnesses = dask.compute(*results)
    return fitnesses
from sciunit.scores import ZScore
#from sciunit import TestSuite
from sciunit.scores.collections import ScoreArray
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from neuronunit.optimisation.optimization_management import dtc_to_rheo, switch_logic,active_values
from neuronunit.tests.base import AMPL, DELAY, DURATION
import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms
import sciunit
import numpy as np
def hof_to_euclid(hof,MODEL_PARAMS,target):
    lengths = {}
    tv = 1
    cnt = 0
    constellation0 = hof[0]
    constellation1 = hof[1]
    subset = list(MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    if len(MODEL_PARAMS)==1:
        
        ax = plt.subplot()
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))

            x = [h[cnt] for h in hof]
            y = [np.sum(h.fitness.values()) for h in hof]
            ax.set_xlim(v[0],v[1])
            ax.set_xlabel(k)
            tgene = tg[cnt]
            yg = 0

        ax.scatter(x, y, c='b', marker='o',label='samples')
        ax.scatter(tgene, yg, c='r', marker='*',label='target')
        ax.legend()

        plt.show()
    
    
    if len(MODEL_PARAMS)==2:
        
        ax = plt.subplot()
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))
                
            if cnt==0:
                tgenex = tg[cnt]
                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0],v[1])
                ax.set_xlabel(k)
            if cnt==1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0],v[1])
                ax.set_ylabel(k)
            cnt+=1

        ax.scatter(x, y, c='r', marker='o',label='samples',s=5)
        ax.scatter(tgenex, tgeney, c='b', marker='*',label='target',s=11)

        ax.legend()

        plt.sgow()
    #print(len(MODEL_PARAMS))
    if len(MODEL_PARAMS)==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))
        
            if cnt==0:
                tgenex = tg[cnt]

                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0],v[1])
                ax.set_xlabel(k)
            if cnt==1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0],v[1])
                ax.set_ylabel(k)
            if cnt==2:
                tgenez = tg[cnt]

                z = [h[cnt] for h in hof]
                ax.set_zlim(v[0],v[1])
                ax.set_zlabel(k)

            cnt+=1
        ax.scatter(x, y, z, c='r', marker='o')
        ax.scatter(tgenex, tgeney,tgenez, c='b', marker='*',label='target',s=11)

        plt.show()
        

def initialise_test(v,rheobase):
    v = switch_logic([v])
    v = v[0]
    k = v.name
    if not hasattr(v,'params'):
        v.params = {}
    if not 'injected_square_current' in v.params.keys():    
        v.params['injected_square_current'] = {}
    if v.passive == False and v.active == True:
        keyed = v.params['injected_square_current']
        v.params = active_values(keyed,rheobase)
        v.params['injected_square_current']['delay'] = DELAY
        v.params['injected_square_current']['duration'] = DURATION
    if v.passive == True and v.active == False:

        v.params['injected_square_current']['amplitude'] =  -10*pq.pA
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION

    if v.name in str('RestingPotentialTest'):
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION
        v.params['injected_square_current']['amplitude'] = 0.0*pq.pA    
        
    return v
from sciunit.scores import ZScore
class NUFeature(object):
    def __init__(self,test,model):
        self.test = test
        self.model = model
    def calculate_score(self,responses):
        model = responses['model'].dtc_to_model()
        model.attrs = responses['params']
        self.test = initialise_test(self.test,responses['rheobase'])
        if "RheobaseTest" in str(self.test.name):
            self.test.score_type = ZScore
            prediction = {'value':responses['rheobase']}
            score_gene = self.test.compute_score(self.test.observation,prediction)
            #lns = np.abs(np.float(score_gene.log_norm_score))
            #return lns
        else:
            try:
                score_gene = self.test.judge(model)
            except:
                print(self.test.observation,self.test.name)
                #print(score_gene,'\n\n\n')

                return 100.0

        if not isinstance(type(score_gene),type(None)):
            if not isinstance(score_gene,sciunit.scores.InsufficientDataScore):
                if not isinstance(type(score_gene.log_norm_score),type(None)):
                    try:

                        lns = np.abs(np.float(score_gene.log_norm_score))
                    except:
                        # works 1/2 time that log_norm_score does not work
                        # more informative than nominal bad score 100
                        lns = np.abs(np.float(score_gene.raw))
                else:
                    # works 1/2 time that log_norm_score does not work
                    # more informative than nominal bad score 100

                    lns = np.abs(np.float(score_gene.raw))    
            else:
                print(prediction,self.test.observation)
                print(score_gene,'\n\n\n')
                lns = 100
        if lns==np.inf:
            lns = np.abs(np.float(score_gene.raw))
        #print(lns,self.test.name)
        return lns