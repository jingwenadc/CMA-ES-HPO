import math
import numpy as np
import itertools
import time
import pickle
from datetime import datetime
from cnn_cpu import objective_function, data_loader, CNN_NUM_RUNS, CNN_LR, CNN_DROPOUT, CNN_LAYERS, CNN_NEURONS


def grid_search(param_combinations, objective_function):
  results = []
  best_params = None
  best_accuracy = 0
  for param in param_combinations:
    # print(param)
    res = -objective_function(param, train_loader, val_loader, test_loader)

    if res > best_accuracy:
      best_accuracy = res
      best_params = param

    param = list(param)
    param.append(res)
    results.append(param)
  return results, best_params, best_accuracy



if __name__ == "__main__":
    print("\n\n\n\n\n========================= Running Grid Search on CNN =========================")
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Timestamp: {job_timestamp}')
    # Parameters
    num_points = math.floor(CNN_NUM_RUNS**(1/4)) # Number of random points

    # Generate the points
    learning_rates_list = np.linspace(CNN_LR[0], CNN_LR[1], num_points)
    n_dropout_list = np.linspace(CNN_DROPOUT[0], CNN_DROPOUT[1], num_points)
    n_layers_list = np.linspace(CNN_LAYERS[0], CNN_LAYERS[1], num_points, dtype=int)
    n_neurons_list = np.linspace(CNN_NEURONS[0], CNN_NEURONS[1], num_points, dtype=int)
    param_combinations = list(itertools.product(learning_rates_list, n_dropout_list, n_layers_list, n_neurons_list))
    print(f'Number of evaluations: {len(param_combinations)} \nExample parameter sets: {param_combinations[:5]}')


    train_loader, val_loader, test_loader = data_loader()

    start_time = time.time()
    # results, best_params, best_accuracy = grid_search(param_combinations, objective_function)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken by CNN + grid search: {elapsed_time:.2f} seconds == {elapsed_time/3600:.2f} hours")

    # Save variables to a file
    # filename = f"results_{job_timestamp}.pkl"
    # with open(filename, 'wb') as f:
    #     pickle.dump(results, f)
    #     pickle.dump(best_params, f)
    #     pickle.dump(best_accuracy, f)

    print("################## Complete Grid Search on CNN ##################")
