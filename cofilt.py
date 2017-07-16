import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os

logging.basicConfig(level = logging.DEBUG,
        format = '%(asctime)s %(filename)s[line:%(lineno)d] '
                + '%(levelname)s %(message)s')
start = datetime.now()

# read data
data_file = os.path.join('ml-latest-small', 'ratings.csv')
df = pd.read_csv(data_file, 
        dtype={'userId':np.int32, 'movieId': np.int64, 
            'rating': np.float32, 'timestamp': np.int64})
n_users = df.userId.max()
n_movies = df.movieId.max()
print ('Max user ID: ' + str(n_users) + ' Max movie ID: ' + str(n_movies))
end = datetime.now()
print ('Escape time (loading data): ' + str(end - start))

# construct rating matrix
R = np.zeros((n_users, n_movies))
for line in df.itertuples():
    R[line[1]-1][line[2]-1] = line[3]

# hyper-parameters
weight_decay = 0.1      # regularization strength
k = 20                  # dimension of latent feature space
max_iters = 100         # maximum iterations
alpha = 0.005           # learning rate
show_iters = 10         # how many iterations to show info

# initialization
U = np.random.randn(k, n_users)     # user latent matrix
M = np.random.randn(k, n_movies)    # movie latent matrix

start = datetime.now()
# training
u_idx, m_idx = R.nonzero()
for itr in range(max_iters+1):
    for ui, mi in zip(u_idx, m_idx):
        delta = R[ui][mi] - np.dot(U[:,ui].T, M[:,mi])
        U[:,ui] += alpha * (delta * M[:,mi] - weight_decay * U[:,ui]) 
        M[:,mi] += alpha * (delta * U[:,ui] - weight_decay * M[:,mi])
    rmse = np.sqrt(np.sum((R - np.dot(U.T, M))**2)) / len(R[R>0])
    if itr % show_iters == 0:
        logging.info('iter ' + str(itr) + ' rmse = ' + str(rmse))
end = datetime.now()
print ('Escape time (training): ' + str(end - start))
