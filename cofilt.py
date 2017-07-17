import numpy as np
import h5py
from scipy.sparse import csr_matrix
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import argparse
import os

# logging and argument parse
logging.basicConfig(level = logging.DEBUG,
        format = '%(asctime)s %(filename)s[line:%(lineno)d] '
                + '%(levelname)s %(message)s')
parse = argparse.ArgumentParser(
	description='Using matrix factorization method to solve' +
		' the collaborative filtering problem.' )
parse.add_argument('input', type=str, 
	help='input data file with csv format')
parse.add_argument('--solver', type=str, choices=['sgd','als', 'adam'],
	default='sgd', help='optimizing method')
parse.add_argument('--weights', type=str, default=None,
        help='weights which can be loaded to finetune the model.') 
parse.add_argument('--max-epochs', default=200, type=int, 
	help='maximum epochs for training')
parse.add_argument('--factor', default=20, type=int, dest='k',
	help='dimension of latent feature space')
parse.add_argument('--weight-decay', default=0.1, type=float,
	help='value controls the regularization strength')
parse.add_argument('--learning-rate', default=0.005, type=float, dest='alpha',
	help='learning rate for SGD, simple ignore it when using ALS')
parse.add_argument('--momentum', default=0, type=float,
        help='momentum for SGD, simple ignore it when using ALS')
parse.add_argument('--dense', action='store_true', default=False,
        help='active to use dense matrix representation. WARNING THAT \
        there may be MEMORY ERROR for large matrix.')
parse.add_argument('--no-tikhonov', action='store_false', default=True,
        dest='tikhonov', help='active to forbid tikhonov regularizer \
         (default ture). For more details about tikhonov regularizer, see: \
         Y. Zhou, D. W., R. S., R. P. (2010). Large-Scale Parallel \
         Collaborative Filtering for the Netflix Prize. Lecture Notes \
         in Computer Science, 5034, 337-347.')
parse.add_argument('--early-stop', default=1e-4, type=float, dest='epsilon', 
        help='finish training when loss decrease less than epsilon')
parse.add_argument('--save-steps', default=None, type=int,
        help='value controls how many steps to store weights, \
        leave None to store weights when finished training')
parse.add_argument('--show-steps', default=10, type=int, 
	help='value controls how many steps to show training info')
parse.add_argument('--test-steps', default=20, type=int,
	help='value controls how many steps to show testing info')
parse.add_argument('--show-fig', action='store_true', default=False,
	help='active to show loss figure when finished training')
parse.add_argument('--log-file', default=None, type=str,
	help='provide filename for writing log, leave none when no need')
args = parse.parse_args()
data_file = args.input
solver = args.solver
max_epochs = args.max_epochs
if args.save_steps == None: save_steps = max_epochs
k = args.k
weight_decay = args.weight_decay
alpha = args.alpha
momentum = args.momentum
epsilon = args.epsilon
show_steps = args.show_steps
test_steps = args.test_steps
show_fig = args.show_fig
log_file = args.log_file
dense = args.dense
tikhonov = args.tikhonov
weights = args.weights
if solver == 'adam': 
    beta_1 = 0.9
    beta_2 = 0.999
    m_value = 1e-8

# read data
start = datetime.now()

df = pd.read_csv(data_file, 
         dtype={'user_uin':np.int32, 'biz_uin': np.int32})
n_users = df.user_uin.max() + 1
n_movies = df.biz_uin.max() + 1

print ('Max user ID: ' + str(n_users) + ' Max movie ID: ' + str(n_movies))
end = datetime.now()
print ('Escape time (loading data): ' + str(end - start))
# split train data and test data
train_data, test_data = train_test_split(df, test_size=0.3)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# construct rating matrix
if dense:
    R = np.zeros((n_users, n_movies))
    for line in train_data.itertuples():
        R[line[1]][line[2]] = line[3]
    T = np.zeros((n_users, n_movies))
    for line in test_data.itertuples():
        T[line[1]][line[2]] = line[3]
else:
    rows = train_data.user_uin
    cols = train_data.biz_uin
    data = np.ones(len(rows))
    R = csr_matrix((data, (rows, cols)))
    rows = test_data.user_uin
    cols = test_data.biz_uin
    data = np.ones(len(rows))
    T = csr_matrix((data, (rows, cols)))
    del rows, cols, data

# initialization
if weights is not None:
    f = h5py.File(weights, 'r')
    U = f['U'][:]    
    M = f['M'][:]
    f.close()
    logging.info('loaded model from ' + weights)
else:
    U = np.random.randn(k, n_users)     # user latent matrix
    M = np.random.randn(k, n_movies)    # movie latent matrix

if dense:
    RI = R.copy()			    # index matrix
    RI[RI > 0] = 1
    TI = T.copy()
    TI[TI > 0] = 1


def sparse_rmse(A, U, M):
    '''
        calculate the rmse for U.T * M and A, here A is sparse matrix
    '''
    s = 0
    for row in xrange(A.shape[0]):
        rd = A.getrow(row).data - np.dot(U[:,row].T, M[:,A.getrow(row).nonzero()[1]])
        s += np.sum(rd ** 2)
    return np.sqrt(float(s) / A.nnz)


# training
start = datetime.now()
train_loss = list()
test_loss = list()
u_idx, m_idx = R.nonzero()
last_loss = 0                       # loss of last epoch, for early stop
last_v = 0 
if solver == 'adam': 
    last_m = 0
    t = 1
for itr in range(max_epochs):
    for ii in xrange(R.nnz):
        ui = u_idx[ii]
        mj = m_idx[ii]
        if solver == 'sgd':		           # SGD optimizer 
            if dense: delta = R[ui][mj] - np.dot(U[:,ui].T, M[:,mj])
            else: delta = R.data[ii] - np.dot(U[:,ui].T, M[:,mj]) 
            v = momentum * last_v + alpha * \
                         (-delta * M[:,mj] + weight_decay * U[:,ui])
            U[:,ui] -= v
            v = momentum * last_v + alpha * \
                         (-delta * U[:,ui] + weight_decay * M[:,mj])
            M[:,mj] -= v
            last_v = v
        elif solver == 'adam':
            if dense: delta = R[ui][mj] - np.dot(U[:,ui].T, M[:,mj])
            else: delta = R.data[ii] - np.dot(U[:,ui].T, M[:,mj]) 
            g = -delta * M[:,mj] + weight_decay * U[:,ui]
            m = beta_1 * last_m + (1 - beta_1) * g
            v = beta_2 * last_v + (1 - beta_2) * g**2
            a = alpha * np.sqrt(1-beta_2**t)/(1-beta_1**t)
            U[:,ui] -= a * m / (np.sqrt(v)+m_value)
            g = -delta * U[:,ui] + weight_decay * M[:,mj]
            m = beta_1 * last_m + (1 - beta_1) * g
            v = beta_2 * last_v + (1 - beta_2) * g**2
            M[:,mj] -= a * m / (np.sqrt(v)+m_value)
            last_v = v
            last_m = m
            t += 1
        elif solver == 'als':	                   # ALS optimizer
            # fix M, solve Ui => AUi = b
            if dense: M_Ui = M[:,R[ui,:]>0]        # movies that ui rated
            else: M_Ui = M[:,R.getrow(ui).nonzero()[1]]
            regularizer = weight_decay * np.eye(k)
            if  tikhonov: regularizer *= len(M_Ui)
            A_Ui = np.dot(M_Ui, M_Ui.T) + regularizer              # A
            if dense: b_Ui = np.dot(M_Ui, R[ui,R[ui,:]>0].T)       # b
            else: b_Ui = np.dot(M_Ui, R.getrow(ui).data.T)
            #U[:,ui] = np.linalg.solve(A_Ui, b_Ui)  # update Ui
            U[:,ui] = np.dot(np.linalg.inv(A_Ui), b_Ui)
            # fix U, solve Mj => AMj = b
            if dense: U_Mj = U[:,R[:,mj]>0]	   # users who rated mj 
            else: U_Mj = U[:,R.getcol(mj).nonzero()[0]]
            regularizer = weight_decay * np.eye(k)
            if  tikhonov: regularizer *= len(U_Mj)
            A_Mj = np.dot(U_Mj, U_Mj.T) + regularizer               # A
            if dense: b_Mj = np.dot(U_Mj, R[R[:,mj]>0,mj])          # b
            else: b_Mj = np.dot(U_Mj, R.getcol(mj).data)   
            #M[:,mj] = np.linalg.solve(A_Mj, b_Mj)  # update Mj
            M[:,mj] = np.dot(np.linalg.inv(A_Mj), b_Mj)
    if dense: 
        train_rmse = np.sqrt(np.sum((RI*(R - np.dot(U.T, M)))**2)/len(R[R>0]))
        test_rmse = np.sqrt(np.sum((TI*(T - np.dot(U.T, M)))**2) / len(T[T>0]))
    else: 
        train_rmse = sparse_rmse(R, U, M)
        test_rmse = sparse_rmse(T, U, M)
    train_loss.append( train_rmse )
    test_loss.append( test_rmse )
    if abs(train_rmse-last_loss) < epsilon: break  # early stop
    last_loss = train_rmse
    if itr % show_steps == 0:
        logging.info('it ' + str(itr) + ' rmse(train) = %.3f' % train_rmse)
    if itr % test_steps == 0:
        logging.info('it ' + str(itr) + ' rmse(TEST) = %.3f' % test_rmse)
    if (itr + 1) % save_steps == 0:
        model_name = 'weights-k%d-it%d.hdf5' % (k, itr) 
        f = h5py.File(model_name, 'w')
        f.create_dataset('U', data = U)
        f.create_dataset('M', data = M)
        f.close()
        logging.info('saved model ' + model_name)
logging.info('it ' + str(itr) + ' rmse(train) = %.3f' % train_rmse)
logging.info('it ' + str(itr) + ' rmse(TEST) = %.3f' % test_rmse)
end = datetime.now()
print ('Escape time (training): ' + str(end - start))

if show_fig:
    import matplotlib.pyplot as plt
    plt.plot(range(max_epochs), train_loss, marker='o', color='r', label='train')
    plt.plot(range(max_epochs), test_loss, marker='v', color='b', label='test')
    plt.title(solver.upper() + ' Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
if log_file != None:
    pd.DataFrame({'train-rmse':train_loss,'test-rmse':test_loss}).to_csv(log_file)
