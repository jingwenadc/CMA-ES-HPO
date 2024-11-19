import math
import numpy as np
import itertools
import time
import pickle
from datetime import datetime
from cnn import objective_function, CNN_NUM_RUNS, CNN_LR, CNN_DROPOUT, CNN_LAYERS, CNN_NEURONS


# python my_script.py >> output.txt

def random_search(param_combinations, objective_function):
  results = []
  best_params = None
  best_accuracy = 0
  for param in param_combinations:
    # print(param)
    res = -objective_function(param)
    
    if res > best_accuracy:
      best_accuracy = res
      best_params = param

    param = list(param)
    param.append(res)
    results.append(param)
  return results, best_params, best_accuracy


if __name__ == "__main__":
    print("\n\n\n\n\n========================= Running Random Search on CNN =========================")
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Timestamp: {job_timestamp}')
    # Parameters
    num_points = math.floor(CNN_NUM_RUNS**(1/4)) # Number of random points

    # Generate the points
    learning_rates_list_CNN_RS = np.random.uniform(CNN_LR[0], CNN_LR[1], num_points)
    n_dropout_list_CNN_RS = np.random.uniform(CNN_DROPOUT[0], CNN_DROPOUT[1], num_points)
    n_layers_list_CNN_RS = np.random.uniform(CNN_LAYERS[0], CNN_LAYERS[1], num_points)
    n_neurons_list_CNN_RS = np.random.uniform(CNN_NEURONS[0], CNN_NEURONS[1], num_points)
    param_combinations_CNN_RS = list(itertools.product(learning_rates_list_CNN_RS, n_dropout_list_CNN_RS, n_layers_list_CNN_RS, n_neurons_list_CNN_RS))
    print(f'Number of evaluations: {len(param_combinations_CNN_RS)} \nExample parameter sets: {param_combinations_CNN_RS[:5]}')

    start_time_CNN_RS = time.time()
    results_CNN_RS, best_params_CNN_RS, best_accuracy_CNN_RS = random_search(param_combinations_CNN_RS, objective_function)
    end_time_CNN_RS = time.time()

    elapsed_time_CNN_RS = end_time_CNN_RS - start_time_CNN_RS
    print(f"Time taken by CNN + random search: {elapsed_time_CNN_RS:.2f} seconds")

    # Save variables to a file
    filename = f"results_CNN_RS_{job_timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results_CNN_RS, f)
        pickle.dump(best_params_CNN_RS, f)
        pickle.dump(best_accuracy_CNN_RS, f)

    print("################## Complete Random Search on CNN ##################")

