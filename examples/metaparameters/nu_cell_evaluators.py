class NUFeatureAllen(object):
    def __init__(self,test,model):
        self.test = test
        self.model = model
    def calculate_score(self,responses):
        dtc = responses['model']
        dtc.attrs = responses['params']
        features = dtc.everything
        if self.test.name in features.keys():
            self.test.set_prediction(features[self.test.name])
            score_gene = self.test.feature_judge()

        else:
            lns = 0
            return lns

        if not isinstance(type(score_gene),type(None)):
            if not isinstance(score_gene,sciunit.scores.InsufficientDataScore):
                if not isinstance(type(score_gene.log_norm_score),type(None)):
                    try:

                        lns = np.abs(score_gene.log_norm_score)
                    except:
                        lns = np.abs(score_gene.raw)
                else:
                    lns = np.abs(score_gene.raw)    
            else:
                lns = 2
        if lns==np.inf:
            lns = 2
        print(lns)
        return lns
def make_allen_objectives():    
    objectives = []

    for tt in nu_tests:
        feature_name = '%s.%s' % (tt.name, tt.name)
        ft = NUFeatureAllen(tt,model)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            ft)
        objectives.append(objective)
    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives) 
    return score_calc, objectives
#if not allen:
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

class NUFeature(object):
    def __init__(self,test,model):
        self.test = test
        self.model = model
    def calculate_score(self,responses):
        model = responses['model'].dtc_to_model()
        model.attrs = responses['params']
        self.test = initialise_test(self.test,responses['rheobase'])
        if "Rheobase" in str(self.test.name):
            prediction = {'value':responses['rheobase']}
            score_gene = self.test.compute_score(self.test.observation,prediction)
        try:
            score_gene = self.test.judge(model)
        except:
            return 100.0

        if not isinstance(type(score_gene),type(None)):
            if not isinstance(score_gene,sciunit.scores.InsufficientDataScore):
                if not isinstance(type(score_gene.log_norm_score),type(None)):
                    try:

                        lns = np.abs(score_gene.log_norm_score)
                    except:
                        lns = np.abs(score_gene.raw)
                else:
                    lns = np.abs(score_gene.raw)    
            else:
                lns = 100
        if lns==np.inf:
            lns = 100
        return lns

def make_basic_objectives()
    objectives = []

    for tt in nu_tests:
        feature_name = '%s.%s' % (tt.name, tt.name)
        ft = NUFeature(tt,model)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            ft)
        objectives.append(objective)

    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives) 
    return score_calc, objectives