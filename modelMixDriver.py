#!/usr/bin/python 

#
# modelMixDriver.py
#   Driver for the modelMix module
# by Shih-Ho Cheng (shihho.cheng@gmail.com)
#

from pylab import *
from matplotlib.patches import *
import scipy.stats as st
import modelMix

Nmod = 4
Npts = 800
numIter = 1000
errorTolerance = 1e-10

fig = figure()
ax  = fig.add_subplot(111)

# Sample two random sets of 2D gaussians
par = array( [ [0., 0. , .2, 1.2, 0.8] , \
               [3., 3., 1.2, 2.2, -0.5] , \
               [6, 0., 1., 0.5, 0.4] , \
               [0., -5., 2., 3., -0.7] ] )
s = []
for k in range(len(par)):
  m_k = array( [ par[k,0], par[k,1] ] )
  c_k = array( [ [par[k,2], par[k,4]], [par[k,4], par[k,3]] ] )
  s_k = multivariate_normal(m_k, c_k, size=Npts/Nmod)
  ax.plot(s_k[:,0],s_k[:,1],'.', ms=3.5)
  s.append(s_k)
s = array(s).reshape(Npts,2)
alpha = ones(Nmod)/(Nmod*1.)

rnd_idx = random_integers(0,Npts-1,size=8)
par = []
for k in range(Nmod):
   par.append( [s[k,0], s[k+1,1], std(s[:,0]), std(s[:,1]), 0.] )
par = array( par )

gm = modelMix.GaussMix(s,par)
respMat = gm.getRespMat(par, alpha)
prev_expLhood = gm.getExpLogLhood(par, alpha, respMat)

for j in range(numIter):
  par = gm.getParameter(respMat)
  alpha = gm.getAlpha(respMat)
  expLhood = gm.getExpLogLhood(par, alpha, respMat)
  ferror = fabs((expLhood - prev_expLhood)/prev_expLhood)
  print "Iter =", j, " | ExpLogLhood = %.2f"%expLhood, " | fracError = %.2e"%ferror
  if ferror<errorTolerance:
    break
  else:
    respMat = gm.getRespMat(par, alpha)
    prev_expLhood = expLhood
    continue
print

print ">> Estimated Parameters (mu_X, mu_Y, sigma_X, sigma_Y, cov[X,Y]):"
print par
print

ellipsePar = gm.getErrorEllipse(par)

for k in range(Nmod):
  ax.add_patch( Ellipse(par[k,0:2],ellipsePar[k,0]*3.,ellipsePar[k,1]*3.,ellipsePar[k,2], \
                alpha=0.4, fc='grey', ec='black') )

ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
show()
