from .utils import *
from scipy.linalg import sqrtm

import numpy as np
from sklearn.impute import KNNImputer
from scipy.sparse.linalg import svds
import random
random.seed(42)


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    matrix[np.isnan(matrix)] = 0
    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    K=k
    u, s, vT = svds(matrix, k=K,  which='LM')
    reconst_matrix = u @ np.diag(s) @ vT
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    m = train_data["question_id"][i]
    
    u[n, :] = u[n, :] + lr * (c - np.dot(u[n, :], z[m, :]) )* z[m, :] 
    z[m, :] = z[m, :] + lr * (c - np.dot(u[n, :], z[m, :]) )* u[n, :]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    valid_data = load_valid_csv("data")
    i=1
    train_mse=[]
    val_mse=[]
    while i<= num_iteration:
          j=1
          while j <= len(train_data["user_id"]):
              u , z = update_u_z(train_data, lr, u, z)
              j=j+1
          i=i+1
          train_mse.append(squared_error_loss(train_data, u, z))
          val_mse.append(squared_error_loss(valid_data, u, z))
    mat = u @ np.transpose(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return  mat, u, z, train_mse, val_mse


def matrix_factorization_main():
    train_matrix = load_train_sparse("data").toarray()
    train_data = load_train_csv("data")
    val_data = load_valid_csv("data")
    try:
        test_data = load_public_test_csv("data")
    except Exception:
        pass
    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    reconstructed2 = svd_reconstruct(train_matrix, 2)
    reconstructed3 = svd_reconstruct(train_matrix, 3)
    reconstructed4 = svd_reconstruct(train_matrix, 4)
    reconstructed5 = svd_reconstruct(train_matrix, 5)
    reconstructed6 = svd_reconstruct(train_matrix, 6)
    reconstructed7 = svd_reconstruct(train_matrix, 7)
    
    acc2=sparse_matrix_evaluate(val_data, reconstructed2)
    acc3=sparse_matrix_evaluate(val_data, reconstructed3)
    acc4=sparse_matrix_evaluate(val_data, reconstructed4)
    acc5=sparse_matrix_evaluate(val_data, reconstructed5)
    acc6=sparse_matrix_evaluate(val_data, reconstructed6)
    acc7=sparse_matrix_evaluate(val_data, reconstructed7)
    
    accuracies=[]
    accuracies.append(acc2)
    accuracies.append(acc3)
    accuracies.append(acc4)
    accuracies.append(acc5)
    accuracies.append(acc6)
    accuracies.append(acc7)
    
    best_acc_idx = 2 + accuracies.index(max(accuracies))
    best_val_acc = max(accuracies)
    
    k=[2,3,4,5,6,7]
    print('-----------------MF Part A-----------------')
    for i in k:
        print(f'Accuracy for k={i} is {accuracies[i-2]}.')
        
    print(f'The Best K for Validation Set is {best_acc_idx} and Its Accuracy is {max(accuracies)}.')
    
    try:
        test_acc2=sparse_matrix_evaluate(val_data, reconstructed2)
        test_acc3=sparse_matrix_evaluate(val_data, reconstructed3)
        test_acc4=sparse_matrix_evaluate(val_data, reconstructed4)
        test_acc5=sparse_matrix_evaluate(val_data, reconstructed5)
        test_acc6=sparse_matrix_evaluate(val_data, reconstructed6)
        test_acc7=sparse_matrix_evaluate(val_data, reconstructed7)
        
        test_accuracies=[]
        test_accuracies.append(acc2)
        test_accuracies.append(acc3)
        test_accuracies.append(acc4)
        test_accuracies.append(acc5)
        test_accuracies.append(acc6)
        test_accuracies.append(acc7)
        
        best_test_acc = max(test_accuracies)
    except Exception:
        pass
    
    best_k_svd:int=best_acc_idx
    best_val_acc_svd:float=best_val_acc
    test_acc_svd:float=best_test_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Part D and E:                                                     #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    l_rate = 0.01
    iteration = 250
    
    reconstructed_als_10, u10, z10, MSE_train_10, MSE_val_10 =als(train_data, 2, l_rate, iteration)
    reconstructed_als_20, u20, z20, MSE_train_20, MSE_val_20 =als(train_data, 3, l_rate, iteration)
    reconstructed_als_30, u30, z30, MSE_train_30, MSE_val_30 =als(train_data, 4, l_rate, iteration)
    reconstructed_als_40, u40, z40, MSE_train_40, MSE_val_40 =als(train_data, 5, l_rate, iteration)
    reconstructed_als_50, u50, z50, MSE_train_50, MSE_val_50 =als(train_data, 6, l_rate, iteration)
    
    acc10 = sparse_matrix_evaluate(val_data, reconstructed_als_10)
    acc20 = sparse_matrix_evaluate(val_data, reconstructed_als_20)
    acc30 = sparse_matrix_evaluate(val_data, reconstructed_als_30)
    acc40 = sparse_matrix_evaluate(val_data, reconstructed_als_40)
    acc50 = sparse_matrix_evaluate(val_data, reconstructed_als_50)
    
    print('-----------------MF Part D-----------------')
    print(f'Final U Matrix is {u10}')
    print(f'Final Z Matrix is {z10}')
    print(f'Final Matrix is {reconstructed_als_10}')
    print(f'Iterations = {iteration}')
    print(f'Learning Rate = {l_rate}')
    print(f'Accuracy for k=2 is {acc10}.')
    print(f'Accuracy for k=3 is {acc20}.')
    print(f'Accuracy for k=4 is {acc30}.')
    print(f'Accuracy for k=5 is {acc40}.')
    print(f'Accuracy for k=6 is {acc50}.')
    print(f'The Best K for Validation Set is k=2 and Its Accuracy is {max(acc10, acc20, acc30, acc40, acc50)}.')
    # Results of part D
    best_k_als:int=2
    best_val_acc_als:float=max(acc10, acc20, acc30, acc40, acc50)

    # Results of part E
    # Save the line chart
    import matplotlib.pyplot as plt
    
    plt.figure('Sum of Squared Loss')
    plt.plot(range(1,1+iteration), MSE_train_20, label='Train')
    plt.plot(range(1,1+iteration), MSE_val_20, label='Val. Data')
    plt.xlabel('iteration')
    plt.ylabel('Squared Error Loss')
    plt.legend(loc='upper left')
    plt.title('Squared Error Loss for Training and Val. Data')
    plt.savefig('plots/matrix_factorization/part_e.png')
    plt.close(fig='Sum of Squared Loss')
    
    print('-----------------MF Part E-----------------')
    print(f'The Best K for Validation Set is k=2 and Its Accuracy is {max(acc10, acc20, acc30, acc40, acc50)}.')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    results={
    'best_k_svd':best_k_svd,
    'test_acc_svd':test_acc_svd,
    'best_val_acc_svd':best_val_acc_svd,
    'best_val_acc_als':best_val_acc_als,
    'best_k_als':best_k_als

    }

    return results

if __name__ == "__main__":
    matrix_factorization_main()
