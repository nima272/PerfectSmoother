
import numpy as np
from scipy.sparse import diags, csr_matrix, csc_matrix
from sksparse.cholmod import cholesky
import math

def calculate_CVSE_for_lambda(lambda_, diff_order, w, y):    

    """
    Calculate the cross validation standard error of smoothed data for a given lambda_.
    
    Based on methodology proposed by:
        * Eilers, P.H., 2003. A perfect smoother. Analytical chemistry, 75(14), pp.3631-3636, 
    after:
        * Whittaker, E.T., 1922. On a new method of graduation. Proceedings of the Edinburgh Mathematical Society, 41, pp.63-75.
    
    Args:
        lambda_:    penalty parameter.
        diff_order: differentiation order.
        w:          weight matrix (0 for missing value, and 1 otherwise).  
        y:          raw data, uniformly sampled, but can include missing values.

    Returns:
        lambda_:    penalty parameter.
        CV_SE:      cross validation standard error for the given lambda_ 
    """
    # import math
    
    # building the differentiation 2D sparse matrix

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
  
    # building diagonal matrix of w
    W = diags(w, 0, format='csc')
    
    A = (W + lambda_ * (D.T @ D)) # Convert to dense matrix
    
    wy = w * y
    factor = cholesky(A)
    z=factor(wy)
    
    A_inv=factor.inv()
    
    H = A_inv @ W
    
    diagonal_elements = H.diagonal()
    
    # Filter out zero elements from the diagonal 
    nonzero_diagonal_elements = diagonal_elements[diagonal_elements != 0]
    
    # Calculate the mean of the nonzero diagonal elements
    mean_diagonal = np.mean(nonzero_diagonal_elements)
    
    # cross validation standard error
    CV_SE=0
    non_zero_no=0
    
    for ii in range(0,len(y)):
        if y[ii]!=0:
            non_zero_no+=1
            CV_SE+=((y[ii]-z[ii])/(1-mean_diagonal))**2
    CV_SE=np.sqrt((CV_SE/non_zero_no))

    return lambda_, CV_SE
