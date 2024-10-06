from .utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: 2D sparse matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0    
    
    for n, j in enumerate(data["question_id"]):
        i = data["user_id"][n]
        correct = data["is_correct"][n]
        log_lklihood = log_lklihood + correct * np.log(sigmoid(theta[i] - beta[j])) + (1-correct) * np.log((1-sigmoid(theta[i] - beta[j])))
            
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for n, j in enumerate(data["question_id"]):
        i = data["user_id"][n]
        theta[i] = theta[i] + lr * (data["is_correct"][n] - sigmoid(theta[i] - beta[j]))
        beta[j] = beta[j] + lr * (sigmoid(theta[i] - beta[j]) - data["is_correct"][n])    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    
    
    theta = 0.5 * np.ones(542)
    beta = 0.5 * np.ones(1774)
    
    train_acc_lst = []
    val_acc_lst = []
    neg_lld_lst_train = []
    neg_lld_lst_val = []
   
    for i in range(iterations):
    #####################################################################
    # TODO:Complete the code                                            #
    ##################################################################### 
        neg_lld_lst_train.append(neg_log_likelihood(data, theta, beta))
        neg_lld_lst_val.append(neg_log_likelihood(val_data, theta, beta))
        
        theta, beta = update_theta_beta(data, lr, theta, beta)
        
        train_acc_lst.append(evaluate(data, theta, beta))
        val_acc_lst.append(evaluate(val_data, theta, beta)) 
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################       
    
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst_train, neg_lld_lst_val


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def item_response_main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    try:
        test_data = load_public_test_csv("data")
    except Exception:
        pass

    num_iterations = 250
    lr = 0.01

    #####################################################################
    # Part B:                                                           #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    theta, beta, val_acc_lst, neg_lld_lst_train, neg_lld_lst_val = irt(train_data, val_data, lr, num_iterations)
    learned_theta:np.array = theta
    learned_beta:np.array = beta
    
    plt.figure('Negative log likelihood')    
    plt.plot(neg_lld_lst_train, label = 'Train')   
    plt.plot(neg_lld_lst_val, label = 'Val. Data')
    plt.xlabel('Iterations')
    plt.ylabel('Negative log likelihood')
    plt.legend(loc = 'upper right')
    plt.title('Negative log likelihoods of Train and Val. Data Part B')
    plt.savefig('plots/IRT/Log Likelihoods of Train and Valid.png')
    plt.close(fig='Negative log likelihood')
    
    print('-----------------Item Response Part B-----------------')
    print(f'Iterations = {num_iterations}')
    print(f'Learning Rate = {lr}')

    val_acc_list:list = val_acc_lst
    neg_lld_lst:list = neg_lld_lst_train
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part C:                                                           #
    # Best Results                                                      #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    print('-----------------Item Response Part C-----------------')
    print(f'The Best Accuracy for Val. Set is {max(val_acc_lst)} and log likelihood is {min(neg_lld_lst_val)}.')
    final_validation_acc = max(val_acc_lst)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Plots                                                             #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    
    questions=[474, 1740, 1599, 821,  386]
    theta=np.sort(theta)
    probs=[]
    
    for j in questions:
        prob=[]
        correct = train_data["is_correct"][j]
        for i in range(542):
            prob.append(correct * sigmoid(theta[i] - beta[j]) + (1-correct) * (1-sigmoid(theta[i] - beta[j])))
       
        probs.append(prob)
        
    plt.figure('Probabilities')    
    plt.plot(theta,probs[0], label = f' question {questions[0]} ' )
    plt.plot(theta,probs[1], label = f' question {questions[1]} ' )
    plt.plot(theta,probs[2], label = f' question {questions[2]} ' )
    plt.plot(theta,probs[3], label = f' question {questions[3]} ' )
    plt.plot(theta,probs[4], label = f' question {questions[4]} ' )
    plt.xlabel('theta')
    plt.ylabel('p(c_ij)')
    plt.legend(loc='lower right')
    plt.title('p(c_ij = 1 | theta)')
    plt.savefig('plots/IRT/Probabilities of Different Qs.png')
    plt.close(fig='Probabilities')
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


    results = {
        'lr':lr,
        'num_iterations':num_iterations,
        'theta':learned_theta,
        'beta':learned_beta,
        'val_acc_list':val_acc_list,
        'neg_lld_lst':neg_lld_lst,
        'final_validation_acc':final_validation_acc,
        }
    return results

if __name__ == "__main__":
    item_response_main()
