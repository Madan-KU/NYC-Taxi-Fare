import yaml
import optuna
import logging
from sklearn.model_selection import cross_val_score

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


def objective(trial, model_class, X,y,cv=5):
    """Objective function for hyperparameter optimization."""

    # Defining Search Space,
    # log=True for search in logarithmic space and not linear space
    if model_class=='SVR':
        params={
            'kernel': trial.suggest_categorical('kernel',['linear', 'poly', 'rbf']),
            'C': trial.suggest_float('C', 1e-5, 1, log=True)
        }

    elif model_class =='GradientBoostingRegressor':
        params={
            'n_estimators':trial.suggest_int('n_estimators',10,200),
            'learning_rate': trial.suggest_float('learning_rate',1e-5,1,log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16)
        }

    elif model_class == 'Ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1, log=True)
        }
        
    elif model_class == 'Lasso':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1, log=True)
        }
        
    elif model_class == 'DecisionTreeRegressor':
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16)
        }
        
    else:
        raise ValueError("Invalid model_class provided.")
    
    model = globals()[model_class](**params)

    return cross_val_score(model, X, y, cv=cv, scoring='r2').mean()



def hyperparameter_tuning(model_class, X, y, n_trials=100):
    """Hyperparameter tuning using Optuna's Tree-structured Parzen Estimator (TPE)
    Returns:
    - A dictionary containing the best hyperparameters.
    """

    # Define study object
    sampler = optuna.samplers.TPESampler()
    study_name = "STUDY_" + model_class

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///optuna_hyperparameter_tuning/optuna_db.sqlite3",  
                                study_name=study_name,
                                sampler=sampler,
                                load_if_exists=True)  

    # Optimize the study with the objective function.
    study.optimize(lambda trial: objective(trial, model_class, X, y), n_trials=n_trials)

    return study.best_params


 
def update_yaml_params(model_name,best_params,yaml_path):   

    try:
        with open(yaml_path, 'r') as file:
            yaml_data=yaml.safe_load(file)

        yaml_data['model'][model_name]['params']=best_params

        with open (yaml_path, 'w') as file:
            yaml.safe_dump(yaml_data, file)
        
        logging.info(f"Updated parameters for {model_name} in {yaml_path}")
    
    except Exception as e:
        logging.error(f"Failed to update parameters for {model_name} in {yaml_path}. Error: {e}")
 


def main():

    configure_logger()
    global config
    config = read_config('parameters.yaml')
    yaml_path= 'parameters.yaml'
    model_class = list(config['model'].keys())

    X, filename = read_data(config['data']['transformed']['X'])
    y, filename = read_data(config['data']['transformed']['y'])
    y=y.squeeze() 


    for model_name in model_class:
        best_params=hyperparameter_tuning(model_name, X, y)
        print(best_params)
        update_yaml_params(model_name,best_params,yaml_path)


if __name__ == "__main__":
    main()