# LMS regression 
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv
import pandas as pd
#-----------------train data process------------------------------------
train_data_path = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\hw-2\concrete-2\concrete\train.csv"
train_data = pd.read_csv(train_data_path, header=None).values


test_data_path = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\hw-2\concrete-2\concrete\test.csv"
test_data = pd.read_csv(test_data_path, header=None).values


m = len(train_data)   
d = len(train_data[0]) - 1


# print(myList_train)
def loss_func(w, dataset):
    loss = 0.5*sum([ (row[-1]-np.inner(w,row[0:7]))**2 for row in dataset ])
    return loss

def grad(w, dataset):
    grad = []
    for j in range(d):
        grad.append(-sum([ (row[-1]-np.inner(w, row[0:7]))*row[j] for row in dataset]))
    return grad

def batch_grad(eps, rate, w, dataset):
    loss =[]
    while np.linalg.norm(grad(w, dataset)) >= eps:
        loss.append(loss_func(w, dataset))
        w = w - [rate*x for x in grad(w, dataset)]       
    return [w, loss]
#---------------------------------batch GD---------------------------------
# =============================================================================

# [ww, loss_v] = batch_grad(0.000001, 0.01, np.zeros(d), train_data)
# print(ww)
# print(loss_func(ww, train_data))
# print(loss_func(ww, test_data))
# plot.plot(loss_v)
# plot.ylabel('Loss function value')
# plot.xlabel('Iterations')
# plot.title('tolerance= 10^-6')
# plot.show()
# =============================================================================
# pi-- permutation of the dataset
def sgd_single(eps, rate, w, dataset, pi):
    flag = 0
    loss_vec =[]
    for x in pi:
        if np.linalg.norm(sgd_grad(w, pi[x], dataset)) <= eps:
            flag = 1
            return [w, loss_vec, flag]
        loss_vec.append(loss_func(w, dataset))
        w = w - [rate*x for x in sgd_grad(w, pi[x] ,dataset)]     
    return [w, loss_vec, flag]

# sample_idx: evaluate grad at sample index
# grad approximation at sampel_idx
def sgd_grad(w, sample_idx, dataset):
    s_grad = []
    for j in range(d):
        s_grad.append(-(dataset[sample_idx][-1]-np.inner(w, dataset[sample_idx][0:7]) )*dataset[sample_idx][j])
    return s_grad

def shuffle_sgd(eps, rate, w, dataset, N_epoch ):
    loss_all =[]
    for i in range(N_epoch):
        pi = np.random.permutation(m)
        [w, loss_vec, flag] = sgd_single(eps, rate, w, dataset, pi)
        if flag == 1:
            return [w, loss_all]
        loss_all = loss_all + loss_vec
    return [w, loss_all]

#----------------------------------SGD---------------------------------------
# [ww, loss_all] = shuffle_sgd(0.000001, 0.002, np.zeros(d), train_data, 20000)
# print(ww)
# print(loss_func(ww, train_data))
# print(loss_func(ww, test_data))
# plot.plot(loss_all)
# plot.ylabel('Loss function value')
# plot.xlabel('Iterations')
# plot.title('tolerance= 10^-6, # passings =20000 ')
# plot.show()

#---------------analytical solution--------------------------------------------
# =============================================================================
data_list = [row[0:7] for row in train_data]
label_list = [row[-1] for row in train_data]
data_mat = np.array(data_list)
label_mat = np.array(label_list)
X = data_mat.transpose()


a = inv(np.matmul(X, X.transpose()))
b = np.matmul(a, X)
c =np.matmul(b, label_mat)
print(c)
# =============================================================================







        
    
    



