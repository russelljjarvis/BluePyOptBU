import pickle
from neo.core import AnalogSignal
import sciunit
from sciunit.models import RunnableModel
import sciunit.capabilities as scap
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf

import l5pc_evaluator
import matplotlib.pyplot as plt

import json
PARAMS = json.load(open('config/params.json'))


class A(object):
    def __init__(self):
        self = self
    
    def set_stop_time(self,tstop):
        pass

class L5Model(RunnableModel,
                  cap.ReceivesSquareCurrent,
                  cap.ProducesActionPotentials,
                  cap.ProducesMembranePotential):
    def __init__(self):
        self.evaluator = l5pc_evaluator.create()
        self.evaluator.fitness_protocols.pop('bAP',None)
        self.evaluator.fitness_protocols.pop('Step3',None)
        self.evaluator.fitness_protocols.pop('Step2',None)
        self.evaluator.NU = None
        self.evaluator.NU = True
        self.run_params = {}
        self.default_params = pickle.load(open('test_params.p','rb'))
        self.test_params = self.default_params 
        #self.default_params = None

    def set_attrs(self,attrs=None):
        #print('these are parameters that can be modified.')
        not_fronzen = {k:v for k,v in self.evaluator.cell_model.params.items() if not v.frozen}
        cnt = 0
        for k,v in enumerate(self.default_params.keys()):
            print(k,v)

            cnt+=1
            if cnt%2==0:
                self.default_params[k] = self.default_params[k]+self.default_params[k]*0.059
            else:    
                self.default_params[k] = self.default_params[k]-self.default_params[k]*0.059
    def inject_square_current(self,current):
        protocol = self.evaluator.fitness_protocols['Step1']
        if 'injected_square_current' in current.keys():
            current = current['injected_square_current']
        protocol.stimuli[0].step_amplitude = float(current['amplitude'])/1000.0
        protocol.stimuli[0].step_delay = float(current['delay'])#/(1000.0*1000.0*1000.0)#*1000.0
        protocol.stimuli[0].step_duration = float(current['duration'])#/(1000.0*1000.0*1000.0)#*1000.0
        
        feature_outputs = self.evaluator.evaluate(self.test_params)

        self.vm = feature_outputs['neo_Step1.soma.v']
        self.vM = self.vm
        plt.plot(self.vm.times,self.vm)
        return feature_outputs['neo_Step1.soma.v']
    
    def get_spike_count(self):
        train = sf.get_spike_train(self.vm)
        return len(train)

    def get_membrane_potential(self):
        """Return the Vm passed into the class constructor."""
        
        return self.vm

    def get_APs(self):
        """Return the APs, if any, contained in the static waveform."""
        vm = self.vm 
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms
    def get_backend(self):
        a = A()
        return a
