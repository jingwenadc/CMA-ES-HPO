from cma import CMAEvolutionStrategy
import time
import pickle
import numpy as np
from datetime import datetime
from cnn_cpu import objective_function, data_loader, CNN_MAX_EVALS, CNN_LR, CNN_DROPOUT, CNN_LAYERS, CNN_NEURONS


def cma_es(initial_guess, sigma, bounds, max_feval, objective_function, train_loader, val_loader, test_loader):
    
    es = CMAEvolutionStrategy(initial_guess, sigma, {'bounds': bounds, 'maxfevals': max_feval})
    
    results = []
    best_params = None
    best_accuracy = 0
    
    while not es.stop():
        params = es.ask()
        f_values = [objective_function(param, train_loader, val_loader, test_loader) for param in params] 
        es.tell(params, f_values)  # Pass results back to optimizer

        res = -min(f_values)


        # Find the index of the minimum value
        min_index = f_values.index(-res)
        if res > best_accuracy:
            best_accuracy = res
            best_params = params[min_index]
        
        results = [list(np.append(params[i], -f_values[i])) for i in range(len(params))]

    return results, best_params, best_accuracy


if __name__ == "__main__":
    print("\n\n\n\n\n========================= Running CMA-ES on CNN =========================")
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Timestamp: {job_timestamp}')
    
    # CMA-ES Setup
    initial_guess = [0.001, 0.2, 3, 128]  # [learning_rate, dropout, num_layers, num_neurons]
    sigma = 0.5
    max_feval = CNN_MAX_EVALS

    bounds = [[CNN_LR[0], CNN_DROPOUT[0], CNN_LAYERS[0], CNN_NEURONS[0]],  # Lower bounds: learning_rate, dropout, num_layers, num_neurons
            [CNN_LR[1], CNN_DROPOUT[1], CNN_LAYERS[1], CNN_NEURONS[1]]]  # Upper bounds: learning_rate, dropout, num_layers, num_neurons
    
    
    # es = CMAEvolutionStrategy(initial_guess, sigma, {'bounds': bounds})
    # es.optimize(objective_function, iterations=2)
    # # Best hyperparameters
    # best_params = es.result.xbest
    # print("Best Parameters:", best_params)
    # es.result

    train_loader, val_loader, test_loader = data_loader()

    start_time = time.time()
    results, best_params, best_accuracy = cma_es(initial_guess, sigma, bounds, max_feval, objective_function, train_loader, val_loader, test_loader)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken by CNN + CMA-ES: {elapsed_time:.2f} seconds")

    # Save variables to a file
    filename = f"results_cmaes_{job_timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
        pickle.dump(best_params, f)
        pickle.dump(best_accuracy, f)

    print("################## Complete CMA-ES on CNN ##################")