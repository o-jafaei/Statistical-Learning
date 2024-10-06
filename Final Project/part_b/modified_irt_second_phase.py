from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    
    return np.exp(z) / (1 + np.exp(z))


def neg_log_likelihood(data, theta, beta, a, gamma):
    """ 
    :param data: 2D sparse matrix
    :param theta: Vector
    :param beta: Vector
    :param a: Vector
    :param gamma: float
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0    
    
    for n, j in enumerate(data["question_id"]):
        i = data["user_id"][n]
        c = data["is_correct"][n]
        log_lklihood = log_lklihood + c * np.log(gamma + (1-gamma)*(sigmoid(a[j]*(theta[i] - beta[j]))))  + (1-c) * np.log((1-gamma) * (1-sigmoid(a[j]*(theta[i] - beta[j]))))
            
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, a, gamma):
    """ 
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param a: Vector
    :param gamma: float
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for n, j in enumerate(data["question_id"]):
        i = data["user_id"][n]
        c = data["is_correct"][n]
        x = a[j] * (theta[i] - beta[j])
        y = sigmoid(x)
        p1 = (c*(1-gamma))/(gamma+(1-gamma)*y) - ((1-c)/(1-y))
        p2 = y*(1-y)
        theta[i] = theta[i] + lr * (p1 * p2 * (a[j]))
        beta[j] = beta[j] + lr * (p1 * p2 * (-a[j]))
        a[j] = a[j] + lr * (p1 * p2 * (theta[i] - beta[j]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ 
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
    a = 0.67 * np.ones(1774)
    gamma = 0 
    
    train_acc_lst = []
    val_acc_lst = []
    neg_lld_lst_train = []
    neg_lld_lst_val = []
   
    for i in range(iterations):
    #####################################################################
    # TODO:Complete the code                                            #
    ##################################################################### 
        neg_lld_lst_train.append(neg_log_likelihood(data, theta, beta, a, gamma))
        neg_lld_lst_val.append(neg_log_likelihood(val_data, theta, beta, a, gamma))
        
        theta, beta = update_theta_beta(data, lr, theta, beta, a, gamma)
        
        train_acc_lst.append(evaluate(data, theta, beta, a, gamma))
        val_acc_lst.append(evaluate(val_data, theta, beta, a, gamma)) 
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################       
    
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, a, gamma, val_acc_lst, neg_lld_lst_train, neg_lld_lst_val


def evaluate(data, theta, beta, a, gamma):
    """ 
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = a[q]*(theta[u] - beta[q]).sum()
        p_a = (gamma + (1-gamma))*sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def item_response_main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    try:
        test_data = load_public_test_csv("../data")
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
    theta, beta, a, gamma, val_acc_lst, neg_lld_lst_train, neg_lld_lst_val = irt(train_data, val_data, lr, num_iterations)
    learned_theta:np.array = theta
    learned_beta:np.array = beta
    
    plt.figure('Negative log likelihood')    
    plt.plot(neg_lld_lst_train, label = 'Train')   
    plt.plot(neg_lld_lst_val, label = 'Val. Data')
    plt.xlabel('Iterations')
    plt.ylabel('Negative log likelihood')
    plt.legend(loc = 'upper right')
    plt.title('Negative log likelihoods of Train and Val. Data Part B')
    plt.savefig('../plots/Part_B/Log Likelihoods of Train and Valid.png')
    plt.close(fig='Negative log likelihood')
    
    
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
    print('-----------------Modified IRT 2nd Phase-----------------')
    print(f'Iterations = {num_iterations}')
    print(f'Learning Rate = {lr}')
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
        c = train_data["is_correct"][j]
        for i in range(542):
            prob.append(c * (gamma + (1-gamma)) * sigmoid(a[j]*(theta[i] - beta[j])) + (1-c) * (1- gamma) * (1-sigmoid(a[j]*(theta[i] - beta[j]))))
       
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
    plt.savefig('../plots/Part_B/Probabilities of Different Qs.png')
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
