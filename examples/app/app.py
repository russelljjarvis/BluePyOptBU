"""
BPO front end for reduced neural models
"""  	

import pandas as pd
import numpy as np                       
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px
import sklearn   
import os
# sense if running on heroku
if 'DYNO' in os.environ:
    heroku = False
else:
    heroku = True
import pickle
experimental_constraints = pickle.load(open("../data_driven/processed_multicellular_constraints.p","rb"))
import utils
import streamlit as st
import bluepyopt as bpop
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model



from neuronunit.optimisation.optimization_management import TSD

def instance_opt(experimental_constraints,MODEL_PARAMS,test_key,model_value,MU,NGEN):
  cell_evaluator, simple_cell, score_calc, test_names = utils.make_evaluator(
                                                        experimental_constraints,
                                                        MODEL_PARAMS,
                                                        test_key,
                                                        model=model_value)
  #cell_evaluator, simple_cell, score_calc = make_evaluator(cells,MODEL_PARAMS)

  #MU =10
  mut = 0.05
  cxp = 0.1
  optimisation = bpop.optimisations.DEAPOptimisation(
          evaluator=cell_evaluator,
          offspring_size = MU,
          map_function = utils.dask_map_function,
          selector_name=diversity,
          mutpb=mut,
          cxpb=cxp)

  final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)

  st.success("Model fitted to experiment {0}".format(test_key))

  best_ind = hall_of_fame[0]
  model = cell_evaluator.cell_model
  cell_evaluator.param_dict(best_ind)
  model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
  opt = model.model_to_dtc()
  vm,fig = inject_and_plot_model(opt,plotly=True)
  st.write(fig)
  #vm,fig = inject_and_plot_passive_model(opt,opt,plotly=True)
  #st.write(fig)

  #plt.show()
  st.markdown("""
  The optimal model parameterization is {0}    
  """.format(opt.attrs))

  st.markdown("""
  Model Performance Relative to fitting data {0}    
  """.format(sum(best_ind.fitness.values)/10*len(experimental_constraints)))
  # radio_value = st.sidebar.radtio("Target Number of Samples",[10,20,30])
  st.markdown("""
  This score is {0} worst score is {1}  
  """.format(sum(best_ind.fitness.values),10*len(experimental_constraints)))

if __name__ == "__main__":  
    st.title('Reduced Model Optimization')
    experimental_constraints.pop("Olfactory bulb (main) mitral cell")
    #API_TOKEN = st.text_input('Please Enter Your Fitbit API token:')
    test_key = st.sidebar.radio("\
      What experiments would you like to fit models to?"
		,tuple(experimental_constraints.keys()))
    #names = tuple(t.name for t in experimental_constraints[test_key].tests)
    #st.text(names)


    experimental_constraints = TSD(experimental_constraints[test_key])
    test_keys = st.sidebar.multiselect("\
     Are you interested in less than all of the features?"
		,tuple(experimental_constraints.keys()))

    #st.text()

    experimental_constraints = [ experimental_constraints[k] for k in test_keys ]
    model_value = st.sidebar.radio("\
		Which models would you like to optimize"
		,("ADEXP","IZHI","NEURONHH"))

    diversity = st.sidebar.radio("\
		Do you want diverse solutions or just the best solution?"
		,("NSGA2","IBEA"))

    readiness = st.sidebar.radio("\
		Ready to go?"
		,("No","Yes"))

    MU = st.sidebar.radio("\
		Population size is"
		,(10,50,100))
    NGEN = st.sidebar.radio("\
		Number of generations is"
		,(10,50,100))

    if readiness == "Yes":
      instance_opt(experimental_constraints,MODEL_PARAMS,test_key,model_value,MU,NGEN)
      '''

      Do you want to try again on a different model to see how that would look?
      '''

      another_go = st.sidebar.radio("\
		  Ready to go?"
		  ,("No","Yes"))
      if another_go:
        instance_opt(experimental_constraints,MODEL_PARAMS,test_key,model_value,MU,NGEN)

