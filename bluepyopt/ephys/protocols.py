"""Protocol classes"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=W0511

import collections

# TODO: maybe find a better name ? -> sweep ?
import logging
logger = logging.getLogger(__name__)

from . import locations
from . import simulators
import copy
from neo import AnalogSignal
import quantities as pq
import neuronunit.capabilities.spike_functions as sf


class Protocol(object):

    """Class representing a protocol (stimulus and recording)."""

    def __init__(self, name=None):
        """Constructor

        Args:
            name (str): name of the feature
        """

        self.name = name


class SequenceProtocol(Protocol):

    """A protocol consisting of a sequence of other protocols"""

    def __init__(self, name=None, protocols=None):
        """Constructor

        Args:
            name (str): name of this object
            protocols (list of Protocols): subprotocols this protocol
                consists of
        """
        super(SequenceProtocol, self).__init__(name)
        self.protocols = protocols

    def run(
            self,
            cell_model,
            param_values,
            sim=None,
            isolate=None,
            timeout=None):
        """Instantiate protocol"""

        responses = collections.OrderedDict({})

        for protocol in self.protocols:

            # Try/except added for backward compatibility
            try:
                response = protocol.run(
                    cell_model=cell_model,
                    param_values=param_values,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout)
            except TypeError as e:
                if "unexpected keyword" in str(e):
                    response = protocol.run(
                        cell_model=cell_model,
                        param_values=param_values,
                        sim=sim,
                        isolate=isolate)
                else:
                    raise

            key_intersect = set(
                response.keys()).intersection(set(responses.keys()))
            if len(key_intersect) != 0:
                raise Exception(
                    'SequenceProtocol: one of the protocols (%s) is trying to '
                    'add already existing keys to the response: %s' %
                    (protocol.name, key_intersect))

            responses.update(response)

        return responses

    def subprotocols(self):
        """Return subprotocols"""

        subprotocols = collections.OrderedDict({self.name: self})

        for protocol in self.protocols:
            subprotocols.update(protocol.subprotocols())

        return subprotocols

    def __str__(self):
        """String representation"""

        content = 'Sequence protocol %s:\n' % self.name

        content += '%d subprotocols:\n' % len(self.protocols)
        for protocol in self.protocols:
            content += '%s\n' % str(protocol)

        return content
from bluepyopt.parameters import Parameter
class SweepProtocol(Protocol):

    """Sweep protocol"""

    def __init__(
            self,
            name=None,
            stimuli=None,
            recordings=None,
            cvode_active=None):
        """Constructor

        Args:
            name (str): name of this object
            stimuli (list of Stimuli): Stimulus objects used in the protocol
            recordings (list of Recordings): Recording objects used in the
                protocol
            cvode_active (bool): whether to use variable time step
        """

        super(SweepProtocol, self).__init__(name)
        self.stimuli = stimuli
        self.recordings = recordings
        self.cvode_active = cvode_active

    @property
    def total_duration(self):
        """Total duration"""

        return max([stimulus.total_duration for stimulus in self.stimuli])

    def subprotocols(self):
        """Return subprotocols"""

        return collections.OrderedDict({self.name: self})

    def _run_func(self, cell_model, param_values, sim=None):
        """Run protocols"""

        try:

            ##
            # this new block could be a failure cause
            ##
            #try:
            cell_model.freeze(param_values)
            cell_model.instantiate(sim=sim)
            '''
            fix parameters in l5pc model
            #except:
                #import pdb
                #pdb.set_trace()
                
                replace = {}
                #cell_model.attrs = {}
                for k,v in param_values.items():
                    #cell_model.attrs[k] = 
                    replace[k] = Parameter(k,value=param_values[k])
                param_values = replace
                cell_model.params = replace
                #import pdb
                #pdb.set_trace()
                cell_model.freeze(param_values)
                for k,v in cell_model.params.items():
                    cell_model.params[k] = float(v)
 
                #print('passed')
                
                #pass
            '''
            try:
                self.instantiate(sim=sim, icell=cell_model.icell)
            except:
                pass
            try:
                ##
                # The defualt NEURON SIM RUN
                ##
                if not hasattr(cell_model,'NU'):
                    #self.cvode_active = False
                    sim.run(self.total_duration, cvode_active=self.cvode_active)
                

                else:
                    '''
                    if False:
                        # Faster way to achieve same thing
                        if cell_model.tests is not None:
                            dtc = cell_model.model_to_dtc()
                            scores = dtc.self_evaluate()
                            responses = {'name':'rheobase_inj',
                                        'model':cell_model.model_to_dtc(),
                                        'rheobase':cell_model.rheobase,
                                        'params':param_values,
                                        'scores':scores}
                    '''
                    # first populate the dtc by frozen default attributes
                    # then update with dynamic gene attributes as appropriate.
                    attrs = cell_model._backend.default_attrs
                    attrs.update(copy.copy(param_values))
                    assert attrs is not None
                    assert len(param_values)
                    dtc = cell_model.model_to_dtc(attrs=attrs) 
                    if hasattr(cell_model,'allen'):



                        from neuronunit.optimisation.optimization_management import three_step_protocol
                        if hasattr(cell_model,'seeded_current'):
                            dtc.seeded_current = cell_model.seeded_current
                            dtc.spk_count = cell_model.spk_count

                            dtc = three_step_protocol(dtc,known_current=cell_model.seeded_current)
                            vm = cell_model.inject_model()
                            if hasattr(dtc,'everything'):
                                #print(dtc.everything['Spikecount_1.5x'],'broken?')
                                #print(dtc.Spikecount_15,'broken no this one is right')
                                #dtc.everything['Spikecount_1.5x'] = dtc.Spikecount_15

                                responses = {'features':dtc.everything,'name':'rheobase_inj',
                                'dtc':dtc,'model':cell_model,'rheobase':cell_model.rheobase,'params':param_values}
                            else:
                                responses = {'name':'rheobase_inj','model':dtc,
                                'rheobase':cell_model.rheobase,'params':param_values}

                        else:
                            #print('also goes here why?')
                            dtc = three_step_protocol(dtc)

                            vm = cell_model.inject_model()
                            if hasattr(dtc,'everything'):
                                responses = {'features':dtc.everything,'name':'rheobase_inj',
                                'dtc':dtc,'model':cell_model,'rheobase':cell_model.rheobase,'params':param_values}
                            else:
                                responses = {'name':'rheobase_inj','model':dtc,
                                'rheobase':cell_model.rheobase,'params':param_values}
                    else:
                        vm = cell_model.inject_model()

                        responses = {'name':'rheobase_inj',
                        'response':vm,'model':dtc,
                        'rheobase':cell_model.rheobase,'params':param_values}

                    return responses


            except (RuntimeError, simulators.NrnSimulatorException):
                logger.debug(
                    'SweepProtocol: Running of parameter set {%s} generated '
                    'an exception, returning None in responses',
                    str(param_values))
                responses = {recording.name:
                             None for recording in self.recordings}
            else:
                responses = {
                    recording.name: recording.response
                    for recording in self.recordings}

            self.destroy(sim=sim)

            cell_model.destroy(sim=sim)

            cell_model.unfreeze(param_values.keys())
            return responses
        except BaseException:
            import sys
            import traceback
            raise Exception(
                "".join(
                    traceback.format_exception(*sys.exc_info())))

    def run(
            self,
            cell_model,
            param_values,
            sim=None,
            isolate=None,
            timeout=None):
        """Instantiate protocol"""

        if isolate is None:
            isolate = True
        '''    
        def _reduce_method(meth):
            """Overwrite reduce"""
            return (getattr, (meth.__self__, meth.__func__.__name__))

        import copyreg
        import types
        copyreg.pickle(types.MethodType, _reduce_method)
        '''
        if isolate:# and not cell_model.name in 'L5PC':
            '''
            responses = self._run_func(cell_model=cell_model,
                            param_values=param_values,
                            sim=sim)
            return responses
            '''
            def _reduce_method(meth):
                """Overwrite reduce"""
                return (getattr, (meth.__self__, meth.__func__.__name__))

            import copyreg
            import types
            copyreg.pickle(types.MethodType, _reduce_method)
            
            import pebble
            from concurrent.futures import TimeoutError
            
            if timeout is not None:
                if timeout < 0:
                    raise ValueError("timeout should be > 0")
            ###
            # Foriegn code
            ###
            #responses = self._run_func(cell_model=cell_model,param_values=param_values,sim=sim)
            #import pdb
            #pdb.set_trace()
            with pebble.ProcessPool(max_workers=1, max_tasks=1) as pool:
                tasks = pool.schedule(self._run_func, kwargs={
                    'cell_model': cell_model,
                    'param_values': param_values,
                    'sim': sim},
                    timeout=timeout)
                try:
                    responses = tasks.result()
                except TimeoutError:
                    logger.debug('SweepProtocol: task took longer than '
                                 'timeout, will return empty response '
                                 'for this recording')
                    responses = {recording.name:
                                 None for recording in self.recordings}
            
        else:
            responses = self._run_func(cell_model=cell_model,
                                       param_values=param_values,
                                       sim=sim)
        #time = [r.response[0] for r in responses.values() ]
        #vm = [ r.response[1] for r in responses.values() ]

        new_responses = {}
        for k,v in responses.items():
            if hasattr(v,'response'):
                time = v.response['time'].values#[r.response[0] for r in self.recording.repsonse ]
                vm = v.response['voltage'].values #[ r.response[1] for r in self.recording.repsonse ]
       
                new_responses['neo_'+str(k)] = AnalogSignal(vm,
                                        units=pq.mV,
                                        sampling_period=(time[1]-time[0])*pq.s)
                train_len = len(sf.get_spike_train(new_responses['neo_'+str(k)]))
                if train_len>0:
                    pass
 

        responses.update(new_responses)
        return responses

    def instantiate(self, sim=None, icell=None):
        """Instantiate"""

        for stimulus in self.stimuli:
            stimulus.instantiate(sim=sim, icell=icell)

        for recording in self.recordings:
            try:
                recording.instantiate(sim=sim, icell=icell)
            except locations.EPhysLocInstantiateException:
                logger.debug(
                    'SweepProtocol: Instantiating recording generated '
                    'location exception, will return empty response for '
                    'this recording')

    def destroy(self, sim=None):
        """Destroy protocol"""

        for stimulus in self.stimuli:
            stimulus.destroy(sim=sim)

        for recording in self.recordings:
            recording.destroy(sim=sim)

    def __str__(self):
        """String representation"""

        content = '%s:\n' % self.name

        content += '  stimuli:\n'
        for stimulus in self.stimuli:
            content += '    %s\n' % str(stimulus)

        content += '  recordings:\n'
        for recording in self.recordings:
            content += '    %s\n' % str(recording)

        return content


class StepProtocol(SweepProtocol):

    """Protocol consisting of step and holding current"""

    def __init__(
            self,
            name=None,
            step_stimulus=None,
            holding_stimulus=None,
            recordings=None,
            cvode_active=None):
        """Constructor

        Args:
            name (str): name of this object
            step_stimulus (list of Stimuli): Stimulus objects used in protocol
            recordings (list of Recordings): Recording objects used in the
                protocol
            cvode_active (bool): whether to use variable time step
        """

        super(StepProtocol, self).__init__(
            name,
            stimuli=[
                step_stimulus,
                holding_stimulus]
            if holding_stimulus is not None else [step_stimulus],
            recordings=recordings,
            cvode_active=cvode_active)

        self.step_stimulus = step_stimulus
        self.holding_stimulus = holding_stimulus

    @property
    def step_delay(self):
        """Time stimulus starts"""
        return self.step_stimulus.step_delay

    @property
    def step_duration(self):
        """Time stimulus starts"""
        return self.step_stimulus.step_duration
