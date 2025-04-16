
from mpi4py import MPI
import pandas as pd
import plotly.express as px
import numpy as np
from sksparse.cholmod import cholesky
import plotly.graph_objects as go
from scipy.sparse import diags, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
import math
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import inv

# Inputs
"""
"Graduation" smoothing technique based on the methodology proposed by: 
    Eilers, P.H., 2003. A perfect smoother. Analytical chemistry, 75(14), pp.3631-3636. 
after 
    Whittaker, E.T., 1922. On a new method of graduation. Proceedings of the Edinburgh Mathematical Society, 41, pp.63-75.


The code also includes finding the optimal value for coefficient 'lambda', in parallel. 


INPUTS:
    INPUT_xy.csv:     is the input data file for smoothing purpose. 
                      It has two columns with headers, x and y. 
                      First column is the x (can be non-uniformly sampled). 
                      The second column is the y (to be smoothed).
                      The file is in the same folder as the main code.
    
    diff_order:       differentiation order.
    lambdas:          range of the lambdas over which the cross validation standard error is calculated.


OUTPUTS:
    mpi_results.txt:  the cross validation standar error for every lamdas set as input
"""


diff_order = 2
lambdas = np.logspace(1, 6, 24)



def calculate_CVSE_for_lambda(lambda_, diff_order, w, y):
    """
    Calculate the cross validation standard error of smoothed data for a given lambda_.
    """
    m = len(y)
    rows = []
    cols = []
    data = []

    for i in range(m - diff_order):
        for j in range(diff_order + 1):
            rows.append(i)
            cols.append(i + j)
            data.append((-1)**(diff_order - j) * math.comb(diff_order, j))

    D = csc_matrix((data, (rows, cols)), shape=(m - diff_order, m))
    W = diags(w, 0, format='csc')
    A = (W + lambda_ * (D.T @ D))
    wy = w * y
    factor = cholesky(A)
    z = factor(wy)
       
    Ainv=inv(A)
    Ainv_W=Ainv.dot(W)
    Ainv_W_Arr=Ainv_W.toarray()
    Ainv_W_Arr_Diag=Ainv_W_Arr.diagonal()
    
    CV_SE = 0
    non_zero_no = 0

    for ii in range(0, len(y)):
        if y[ii] != 0:
            non_zero_no += 1
            CV_SE += ((y[ii] - z[ii]) / (1 - Ainv_W_Arr_Diag[ii]))**2
    CV_SE = np.sqrt((CV_SE / non_zero_no))

    return lambda_, CV_SE

data = pd.read_csv('INPUT_xy.csv')

data.set_index('X', inplace=True)
data.index = (data.index - data.index[0])
max_X = data.index.max()
new_index = pd.RangeIndex(start=data.index[0], stop=int(max_X) + 1, step=1)
new_data = pd.DataFrame(index=new_index, columns=['y'])
new_data['y'] = 0.0
new_data.index.name='X'
for X in new_data.index:
    if X in data.index:
        new_data.at[X, 'y'] = data.loc[data.index == X, 'y'].values[0]
new_data['weights'] = 0.0
for X in new_data.index:
    if new_data.at[X, 'y']!=0:
        new_data.at[X, 'weights'] = 1
        
w = new_data['weights'].to_numpy()
y = new_data['y'].to_numpy()


# Parallel computation of cross validation standard error for different lambdas
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Distribute lambdas among processes
results = []
for i in range(rank, len(lambdas), size):
    lambda_ = lambdas[i]
    result = calculate_CVSE_for_lambda(lambda_, diff_order, w, y)
    results.append(result)

# Collect results at root process
all_results = comm.gather(results, root=0)

# print all results
if rank == 0:
    final_results = []
    for res_list in all_results:
        final_results.extend(res_list)
    for res in final_results:
        print(f"Lambda: {res[0]}, CV_SE: {res[1]}")
    with open("mpi_results.txt", "w") as f: # write to file
        for res in final_results:
            f.write(f"{res[0]},{res[1]}\n")
