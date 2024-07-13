from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.estimation import MarginalOutcomeEstimator, IPW, Standardization, StratifiedStandardization
from causallib.estimation import AIPW, PropensityFeatureStandardization, WeightedStandardization

class DifferenceInMeans(object):
    
    def __init__(self):
        self.model_class = MarginalOutcomeEstimator
        self.learner = None
        self.fit_y = True
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def get_effect(self, data):
        X_train, X_val, Z_train, Z_val, Y_train, Y_val = data
        if self.fit_y:
            self.model.fit(X_train, Z_train, Y_train)
        else:
            self.model.fit(X_train, Z_train)
        if self.outcome_y:
            train_outcomes = self.model.estimate_population_outcome(X_train, Z_train, Y_train)
            val_outcomes = self.model.estimate_population_outcome(X_val, Z_val, Y_val)
        else:
            train_outcomes = self.model.estimate_population_outcome(X_train, Z_train)
            val_outcomes = self.model.estimate_population_outcome(X_val, Z_val)
        train_effect = self.model.estimate_effect(train_outcomes[1], train_outcomes[0])["diff"]
        val_effect = self.model.estimate_effect(val_outcomes[1], val_outcomes[0])["diff"]
        return train_effect, val_effect
    

class IPSW(DifferenceInMeans):

    def __init__(self):
        super().__init__()
        self.model_class = IPW
        self.learner = LogisticRegression(solver="liblinear")
        self.fit_y = False
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)


class OutcomeImputation(DifferenceInMeans):
    
    def __init__(self):
        super().__init__()
        self.model_class = Standardization
        self.learner = LinearRegression()
        self.fit_y = True
        self.outcome_y = False
        self.model = self.model_class(learner=self.learner)


class StratifiedOutcomeImputation(DifferenceInMeans):
    
    def __init__(self):
        super().__init__()
        self.model_class = StratifiedStandardization
        self.learner = LinearRegression()
        self.fit_y = True
        self.outcome_y = False
        self.model = self.model_class(learner=self.learner)


class DoublyRobust(DifferenceInMeans):
    
    def __init__(self):
        super().__init__()
        self.model_class = AIPW
        self.fit_y = True
        self.outcome_y = True
        self.outcome_imputation = OutcomeImputation().model
        self.ipw = IPSW().model
        self.model = self.model_class(self.outcome_imputation, self.ipw)


class DoublyRobustIPFeature(DoublyRobust):
    
    def __init__(self):
        super().__init__()
        self.model_class = PropensityFeatureStandardization
        self.outcome_y = False
        self.model = self.model_class(self.outcome_imputation, self.ipw)


class DoublyRobustImportance(DoublyRobust):
    
    def __init__(self):
        super().__init__()
        self.model_class = WeightedStandardization
        self.outcome_y = False
        self.model = self.model_class(self.outcome_imputation, self.ipw)
