import numpy as np

#####################################################################
# TODO:                                                             #                                                          
# Import packages you need here                                     #
#####################################################################
from sklearn.impute import KNNImputer
from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

print('-----------------Ensemble-----------------')
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  




#####################################################################
# Define and implement functions here                               #
#####################################################################
def knn_impute_by_item(matrix, valid_data, k):
    """ 
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    #m=np.transpose(matrix)
    imputer = KNNImputer(n_neighbors=k)
    completed_mat = imputer.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    print(f"Validation Accuracy k={k}: {acc}")

    return completed_mat

#####################################################################
#                       END OF YOUR CODE                            #
##################################################################### 




def ensemble_main():
    #####################################################################
    # Compute the finall validation and test accuracy                   #
    #####################################################################
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    
    try:
        test_data = load_public_test_csv("data")
    except Exception:
        pass
    
    method1_output = np.array([])
    method2_output = np.array([])
    method3_output = np.array([])
    
    random.seed(42)
    bootstrap_set1 = random.choices(sparse_matrix, k = min(sparse_matrix.shape))
    bootstrap_set2 = random.choices(sparse_matrix, k = min(sparse_matrix.shape))
    bootstrap_set3 = random.choices(sparse_matrix, k = min(sparse_matrix.shape))
    
    
    method1_output = knn_impute_by_item(bootstrap_set1, val_data, 6)
    method2_output = knn_impute_by_item(bootstrap_set2, val_data, 11)
    method3_output = knn_impute_by_item(bootstrap_set3, val_data, 26)
    
    final_output = np.mean( np.array([method1_output, method2_output, method3_output]), axis=0 )
    
    for i in range(min(final_output.shape)):
        for j in range(max(final_output.shape)):
            if final_output[i][j] > 0.5:
                final_output[i][j] = 1
            
    
    final_acc = sparse_matrix_evaluate(val_data, final_output)
    
    print(f'Final Accuracy of Ensemble Approach is {final_acc}.')
    
    try:
        method1_output_test = knn_impute_by_item(bootstrap_set1, test_data, 6)
        method2_output_test = knn_impute_by_item(bootstrap_set2, test_data, 11)
        method3_output_test = knn_impute_by_item(bootstrap_set3, test_data, 26)
        final_output_test = np.mean( np.array([method1_output_test, method2_output_test, method3_output_test]), axis=0 )
        
        for i in range(min(final_output_test.shape)):
            for j in range(max(final_output_test.shape)):
                if final_output_test[i][j] > 0.5:
                    final_output_test[i][j] = 1

        final_acc_test = sparse_matrix_evaluate(test_data, final_output_test)
    except Exception:
        pass
    
    val_acc_ensemble:float = final_acc
    try:
        test_acc_ensemble:float = final_acc_test
    except :
        test_acc_ensemble:float = None
    method1_output_matrix:np.array = method1_output
    method2_output_matrix:np.array = method2_output
    method3_output_matrix:np.array = method3_output
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results={
    'val_acc_ensemble':val_acc_ensemble,
    'test_acc_ensemble':test_acc_ensemble,
    'method1_output_matrix':method1_output_matrix,
    'method2_output_matrix':method2_output_matrix,
    'method3_output_matrix':method3_output_matrix
    }

    return results


if __name__ == "__main__":
    ensemble_main()

