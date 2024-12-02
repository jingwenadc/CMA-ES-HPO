# CNN Hyperparameter Optimization with dCMA-ES, CMA-ES, and Bayesian Optimization

## Project Description
This project implements hyperparameter optimization for a Convolutional Neural Network (CNN) using three different optimization techniques: Discrete Covariance Matrix Adaptation Evolution Strategy (dCMA-ES), Continuous CMA-ES, and Bayesian Optimization. The goal is to enhance the performance of a CNN model on the CIFAR-10 dataset with limited computing resources (CPU-only).


## Optimization Techniques
# Continuous CMA-ES
Optimizes continuous parameters such as learning rate and dropout rate within specified bounds.
# Discrete CMA-ES
Focuses on discrete parameter spaces by rounding values to predefined sets.
# Bayesian Optimization
Utilizes Gaussian processes to model the objective function and find optimal parameters efficiently.

## Results
The best hyperparameters found using each technique are documented in the results directory. For instance:
1. Continuous CMA-ES: Achieved a test accuracy of 71.61% with parameters: learning_rate = 0.000696, num_hidden = 256, dropout = 0.286.
2. Discrete CMA-ES: Achieved a test accuracy of 70.51% with parameters: learning_rate = 0.001, num_hidden = 128, dropout = 0.
3. Bayesian Optimization: Achieved a test accuracy of 72.88% with parameters: learning_rate = 0.001445, num_hidden = 350, dropout = 0.018.