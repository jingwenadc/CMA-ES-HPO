import time
import pickle
import numpy as np
from datetime import datetime
from cnn_cpu import objective_function, data_loader, CNN_MAX_EVALS, CNN_LR, CNN_DROPOUT, CNN_LAYERS, CNN_NEURONS
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from functools import partial

def bo(search_space, max_feval, objective_function, train_loader, val_loader, test_loader):
    objective_with_loaders = partial(objective_function, train_loader, val_loader, test_loader)
    raw_result = gp_minimize(objective_with_loaders, search_space, n_calls=max_feval, random_state=42)
    
    
    best_accuracy = raw_result['fun']
    best_params =  raw_result['x']
    results = [x + [f] for x, f in zip(raw_result['x_iters'], -raw_result['func_vals'])]

    return results, best_params, best_accuracy


if __name__ == "__main__":
    print("\n\n\n\n\n========================= Running BO on CNN =========================")
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Timestamp: {job_timestamp}')
    
    # BO Setup
    # Define the search space for the hyperparameters
    search_space = [
        Real(CNN_LR[0], CNN_LR[1], name='learning_rate'),
        Real(CNN_DROPOUT[0], CNN_DROPOUT[1], name='dropout'),
        Integer(CNN_LAYERS[0], CNN_LAYERS[1], name='num_layers'),
        Integer(CNN_NEURONS[0], CNN_NEURONS[1], name='num_neurons')
    ]
    train_loader, val_loader, test_loader = data_loader()

    start_time = time.time()
    results, best_params, best_accuracy = bo(search_space, CNN_MAX_EVALS, objective_function, train_loader, val_loader, test_loader)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken by CNN + BO: {elapsed_time:.2f} seconds")

    # Save variables to a file
    filename = f"results_bo_{job_timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
        pickle.dump(best_params, f)
        pickle.dump(best_accuracy, f)

    print("################## Complete BO on CNN ##################")