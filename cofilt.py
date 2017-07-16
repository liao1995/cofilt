import numpy as np
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
parse.add_argument('--solver', type=str, choices=['sgd','als'],
	default='sgd', help='optimizing method')
parse.add_argument('--max-epochs', default=100, type=int, 
	help='maximum epochs for training')
parse.add_argument('--factor', default=20, type=int, dest='k',
	help='dimension of latent feature space')
parse.add_argument('--weight-decay', default=0.1, type=float,
	help='value controls the regularization strength')
parse.add_argument('--learning-rate', default=0.01, type=float, dest='alpha',
	help='learning rate for SGD, simple ignore it when using ALS')
parse.add_argument('--early-stop', default=1e-4, type=float, dest='epsilon', 
        help='finish training when loss decrease less than epsilon')
parse.add_argument('--show-steps', default=10, type=int, 
	help='value controls how many steps to show training info')
parse.add_argument('--test-steps', default=20, type=int,
	help='value controls how many steps to show testing info')
parse.add_argument('--show-fig', default=True, type=bool,
	help='show loss figure or not when completed training')
parse.add_argument('--log-file', default=None, type=str,
	help='provide filename for writing log, leave none when no need')
args = parse.parse_args()
data_file = args.input
solver = args.solver
max_epochs = args.max_epochs
k = args.k
weight_decay = args.weight_decay
alpha = args.alpha
epsilon = args.epsilon
show_steps = args.show_steps
test_steps = args.test_steps
show_fig = args.show_fig
log_file = args.log_file

# read data
start = datetime.now()
#data_file = os.path.join('ml-latest-small', 'ratings.csv')
df = pd.read_csv(data_file, 
        dtype={'userId':np.int32, 'movieId': np.int64, 
            'rating': np.float32, 'timestamp': np.int64})
n_users = df.userId.max()
n_movies = df.movieId.max()
print ('Max user ID: ' + str(n_users) + ' Max movie ID: ' + str(n_movies))
end = datetime.now()
print ('Escape time (loading data): ' + str(end - start))
# split train data and test data
train_data, test_data = train_test_split(df, test_size=0.3)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# construct rating matrix
R = np.zeros((n_users, n_movies))
for line in train_data.itertuples():
    R[line[1]-1][line[2]-1] = line[3]
T = np.zeros((n_users, n_movies))
for line in test_data.itertuples():
    T[line[1]-1][line[2]-1] = line[3]

# initialization
U = np.random.randn(k, n_users)     # user latent matrix
M = np.random.randn(k, n_movies)    # movie latent matrix
RI = R.copy()			    # index matrix
RI[RI > 0] = 1
TI = T.copy()
TI[TI > 0] = 1

# training
start = datetime.now()
train_loss = list()
test_loss = list()
u_idx, m_idx = R.nonzero()
last_loss = 0                       # loss of last epoch, for early stop
for itr in range(max_epochs):
    for ui, mj in zip(u_idx, m_idx):
        if solver == 'sgd':		           # SGD optimizer 
            delta = R[ui][mj] - np.dot(U[:,ui].T, M[:,mj])
            U[:,ui] += alpha * (delta * M[:,mj] - weight_decay * U[:,ui]) 
            M[:,mj] += alpha * (delta * U[:,ui] - weight_decay * M[:,mj])
        elif solver == 'als':	                   # ALS optimizer
            # fix M, solve Ui => AUi = b
            M_Ui = M[:,R[ui,:]>0] 	           # movies that ui rated
            A_Ui = np.dot(M_Ui, M_Ui.T) + weight_decay * np.eye(k) # A
            b_Ui = np.dot(M_Ui, R[ui,R[ui,:]>0].T)                 # b
            #U[:,ui] = np.linalg.solve(A_Ui, b_Ui)  # update Ui
            U[:,ui] = np.dot(np.linalg.pinv(A_Ui), b_Ui)
            # fix U, solve Mj => AMj = b
            U_Mj = U[:,R[:,mj]>0]	           # users who rated mj 
            A_Mj = np.dot(U_Mj, U_Mj.T) + weight_decay * np.eye(k) # A
            b_Mj = np.dot(U_Mj, R[R[:,mj]>0,mj])                   # b   
            #M[:,mj] = np.linalg.solve(A_Mj, b_Mj)  # update Mj
            M[:,mj] = np.dot(np.linalg.pinv(A_Mj), b_Mj)
    train_rmse = np.sqrt(np.sum((RI*(R - np.dot(U.T, M)))**2)/len(R[R>0]))
    train_loss.append( train_rmse )
    test_rmse = np.sqrt(np.sum((TI*(T - np.dot(U.T, M)))**2) / len(T[T>0]))
    test_loss.append( test_rmse )
    if abs(train_rmse-last_loss) < epsilon: break  # early stop
    last_loss = train_rmse
    if itr % show_steps == 0:
        logging.info('it ' + str(itr) + ' rmse(train) = %.3f' % train_rmse)
    if itr % test_steps == 0:
        logging.info('it ' + str(itr) + ' rmse(TEST) = %.3f' % test_rmse)
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
