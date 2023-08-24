import tpot2
import sklearn
import sklearn.datasets


#my_analysis.py

if __name__ == "__main__":

    est = tpot2.TPOTEstimator(  population_size=3,
                                generations=1,
                                scorers=['roc_auc_ovr'], #scorers can be a list of strings or a list of scorers. These get evaluated during cross validation. 
                                scorers_weights=[1],
                                classification=True,
                                n_jobs=3, 
                                early_stop=5, #how many generations with no improvement to stop after
                                
                                #List of other objective functions. All objective functions take in an untrained GraphPipeline and return a score or a list of scores
                                other_objective_functions= [ ],
                                
                                #List of weights for the other objective functions. Must be the same length as other_objective_functions. By default, bigger is better is set to True. 
                                other_objective_functions_weights=[],
                                verbose=6)

    scorer = sklearn.metrics.get_scorer('roc_auc_ovo')
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.70, test_size=0.30)
    est.fit(X_train, y_train)
    print(scorer(est, X_test, y_test))