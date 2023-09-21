import tpot2
import sklearn
import sklearn.datasets
import numpy as np
from functools import partial
import os
import dill as pickle
import openml



def SeeClassDistribution(a):
    return {x:a.count(x) for x in set(a)}

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, preprocess=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)


        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)


            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


# put everyhting in herer so no refits
def SelectionIndividualScores(est,X,y,X_select,y_select,classification):
    # fit model
    est.fit(X,y)
    scores = []

    if classification:
        scores = list(est.predict(X_select) == y_select)
        scores.append(sum(scores) / len(scores))

    else:
        # regression: wanna minimize the distance between them
        scores = list(np.absolute(y_select - est.predict(X_select)))
        scores.append(sum(scores))


    return scores
    # return est.predict(X_select).flatten()

if __name__ == "__main__":

    # stuff we need to know before hand
    classification = True

    # get the data first
    print('Getting data')
    X, y, X_test, y_test = load_task(359990)
    print('Data gathered')

    print('X:',X.shape,'|','y:',y.shape)
    print('X_test:',X_test.shape,'|','y_test:',y_test.shape)


    # split data according to traning and testing type
    split_pro = 0.95
    if classification:
        X_train, X_select, y_train, y_select = sklearn.model_selection.train_test_split(X, y, train_size=split_pro, test_size=1-split_pro, stratify=y, random_state=17)
        # see class distribution
        print('y_train:',SeeClassDistribution(list(y_train)))
        print('y_select:',SeeClassDistribution(list(y_select)))
    else:
        X_train, X_select, y_train, y_select = sklearn.model_selection.train_test_split(X, y, train_size=split_pro, test_size=1-split_pro, random_state=17)

    print('X_train:',X_train.shape,'|','y_train:',y_train.shape)
    print('X_select:',X_select.shape,'|','y_select:',y_select.shape)

    # create custom objective function

    select_objective = partial(SelectionIndividualScores,X=X_train,y=y_train,X_select=X_select,y_select=y_select,classification=classification)
    select_objective.__name__ = 'scorer'

    exit(0)

    print('\nBEGIN EVOLUTION')
    est = tpot2.TPOTEstimator(  population_size=25,
                                generations=50,
                                classification=classification,
                                n_jobs=10,

                                # selection scheme stuff
                                survival_selector=None,
                                parent_selector=tpot2.selectors.lexicase_selection,

                                # variation stuff
                                crossover_probability=0,
                                crossover_then_mutate_probability=0,
                                mutate_then_crossover_probability=0,

                                #score stuff
                                scorers=[],
                                scorers_weights=[],


                                #List of other objective functions, names, and weigths. All objective functions take in an untrained GraphPipeline and return a score or a list of scores
                                other_objective_functions=[select_objective],
                                # other_objective_functions_weights=[-1] * (len(X_select)+1), # minimize
                                other_objective_functions_weights=[1] * (len(X_select)+1), # maxmimize
                                objective_function_names = ['s-'+str(i) for i in range(len(y_select)+1)],

                                verbose=6,
                                root_config_dict="classifiers",
                                # root_config_dict="regressors",
                                inner_config_dict= ["arithmetic_transformer","transformers","selectors","passthrough","feature_set_selector"],
                                leaf_config_dict=["arithmetic_transformer","transformers","selectors","passthrough","feature_set_selector"],
                                max_size=7,
                                memory_limit = "20GB")

    scorer = sklearn.metrics.get_scorer('roc_auc_ovo')
    est.fit(X_train,y_train)
    print('final_scorer:',scorer(est, X_test, y_test))