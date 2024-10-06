from sklearn.impute import KNNImputer
from .utils import *
#####################################################################
# TODO:                                                             #
# Import packages you need here                                     #
#####################################################################
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    completed_mat = imputer.fit_transform(matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################    
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    m=np.transpose(matrix)
    imputer = KNNImputer(n_neighbors=k)
    completed_mat = np.transpose(imputer.fit_transform(m))
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    
    try:
        test_data = load_public_test_csv("data")
    except Exception:
        pass
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Part B&C:                                                         #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_user().                                                       #
    #####################################################################
    
    # Part B.
    print('-----------------KNN Part B-----------------')
    k=[1,6,11,16,21,26]
    str_k=['1','6','11','16','21','26']
    accuracies=[]
    for i in k:
        accuracies.append(knn_impute_by_user(sparse_matrix, val_data, i))
    
    plt.figure('KNNImputer by Students')
    plt.plot(str_k,accuracies,'o')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy for Different K Values (Impute by Students)')
    plt.savefig('plots/knn/KNNImputer by Students.png')
    plt.close(fig='KNNImputer by Students')
    # Part C.
    print('-----------------KNN Part C-----------------')
    best_acc_idx = accuracies.index(max(accuracies))
    best_k = k[best_acc_idx]
    print(f'The Best K is {best_k} and Its Accuracy is {max(accuracies)}.')
    
    pass
    user_best_k:float = best_k                    # :float means that this variable should be a float number
    user_test_acc:float = None
    user_valid_acc:list = accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    
    #####################################################################
    # Part D:                                                           #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_item().                                                        #
    #####################################################################
    print('-----------------KNN Part D-----------------')
    
    accuracies=[]
    for i in k:
        accuracies.append(knn_impute_by_item(sparse_matrix, val_data, i))
    
    plt.figure('KNNImputer by Item')
    plt.plot(str_k,accuracies,'o')
    plt.close(fig='KNNImputer by Students')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy for Different K Values (Impute by Item)')
    plt.savefig('plots/knn/KNNImputer by Item.png')
    plt.close(fig='KNNImputer by Item')
    
    best_acc_idx = accuracies.index(max(accuracies))
    best_k = k[best_acc_idx]
    print(f'The Best K is {best_k} and Its Accuracy is {max(accuracies)}.')
    pass
    question_best_k:float = best_k
    question_test_acc:float = None
    question_valid_acc:list = accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
    'user_best_k':user_best_k,
    'user_test_acc':user_test_acc,
    'user_valid_accs':user_valid_acc,
    'question_best_k':question_best_k,
    'question_test_acc':question_test_acc,
    'question_valid_acc':question_valid_acc,
    }
    
    
    return results

if __name__ == "__main__":
    knn_main()
