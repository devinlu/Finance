from cvxopt import solvers, matrix
import numpy as np

def calculateVariance(revenues, weights):
    E = np.asmatrix(np.cov(revenues))
    w = np.asmatrix(weights)
    return w * E * w.T

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

portfolio_value = 10
n = 10
t = 30
w = rand_weights(n)
revenues = np.random.rand(n, t)
alpha = w[n-1] # fb spend. TODO: verify this


E = np.cov(revenues)
G = -np.eye(n)
q = np.zeros(n)
h = np.zeros(n)
A = np.zeros((2,n))
A[1] = np.ones(n)
A[(0, n-1)] = 1
b = np.zeros(2)
b[0] = alpha
b[1] = 1

f = matrix
wt = solvers.qp(f(E), f(q), f(G), f(h), f(A), f(b))['x'].T
print wt, w
print calculateVariance(revenues, w), "-->", calculateVariance(revenues, wt)
