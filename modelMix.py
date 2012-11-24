from pylab import *
import scipy.stats as st
import sys

class GaussMix:
  """Gaussian Mixture for K models based on the EM algorithm 
     for normally distributed random numbers in 2 dimensions.

     Args:
      Z: sample data array
      par: a list of K parameter arrays composed of meanX, meanY
           sigmaX, sigmaY, and sqrt(covXY)

     Attributes:
      N: Number of sample points
      D: Dimensionality
      K: Number of models to fit
      The following two are just to simplify calculations:
      X: 2D array of x-component repeated (columnwise) K times 
      Y: 2D array of y-component repeated (columnwise) K times
  """

  def __init__(self, Z, par):
    """Constructor that initializes the object given: 
      the sample array Z and parameter array 
      (meanX, meanY, sigmaX, sigmaY, sqrt(covXY)).
      The dimensionalities are:
        Z   --> N x 2
        par --> K x 5
    """
    self.N = Z.shape[0]   # Number of sample points
    self.D = 2            # Dimensionality
    self.K = len(par)     # Number of models to fit
    self.X = repeat( Z[:,0].reshape(self.N,1) , self.K, axis=1 )
    self.Y = repeat( Z[:,1].reshape(self.N,1) , self.K, axis=1 )
   
  def getNormal2dPdf(self, par):
    """Returns the probability density for each of the rows in Z.
       output --> N x K
    """
    rho = par[:,4]/(par[:,2]*par[:,3])                      # correlation (K x 1)
    fconstant = (1-rho**2.)**(-1./2.) / (2.*pi*par[:,2]*par[:,3]) # front constant (K x 1)
    P = transpose(par)   # to make the next computation convinient (5 x K)
    mobDist = (   (self.X - P[0,:])**2. / P[2,:]**2. \
                + (self.Y - P[1,:])**2. / P[3,:]**2. \
                - 2.*(self.X-P[0,:])*(self.Y-P[1,:])*rho/(P[2,:]*P[3,:]) \
              ) / (1. - rho**2.)
    return fconstant * exp( -0.5 * mobDist )
                

    return constant*exp(delta) 

  def getRespMat(self, par, alpha):
    """Returns the responsability matrix with dimensionality N x K
       NOTE: alpha has dimensionality K
    """
    num = alpha*self.getNormal2dPdf(par)
    den = sum( num, axis=1 )
    return num / repeat( den.reshape(self.N,1), self.K, axis=1 )

  def getParameter(self, respMat):
    """Returns the updated (with respect to respMat) par
       array of dimensionality K x 5.
    """
    respMatNorm = sum( respMat, axis=0 )
    # Get the components of the means (K long array for each component)
    mX = sum( respMat*self.X, axis=0 ) / respMatNorm
    mY = sum( respMat*self.Y, axis=0 ) / respMatNorm
    # Get the diagonal components of the cov. matrices (k long for each component)
    scovXX = sqrt( sum( respMat*(self.X-mX)**2., axis=0 ) / respMatNorm )
    scovYY = sqrt( sum( respMat*(self.Y-mY)**2., axis=0 ) / respMatNorm )
    # Get the off-diagonal components of the cov. matrices (k long)
    covXY = sum( respMat*(self.X-mX)*(self.Y-mY), axis=0 ) / respMatNorm
    par = []
    for k in range(self.K):
      par.append( [ mX[k], mY[k], scovXX[k], scovYY[k], covXY[k] ] )
    return array(par)

  def getAlpha(self, respMat):
    """Calculates the mixing weights of each model.
       The output array has dimension K.
    """
    return sum( respMat, axis=0 ) / (self.N*1.)

  def getExpLogLhood(self, par, alpha, respMat):
    """Calculates the expectation of the log likelihood.
       Returns a scalar.
    """
    return sum( sum( respMat * ( log(alpha) + log(self.getNormal2dPdf(par)) ), axis=1 ), axis=0 )


  def getErrorEllipse(self, par):
    """Returns the error ellipse parameters given the
       covariance matrix.
       The said parameters are:
        sqrt(eigenVal_1), sqrt(eigenVal_2), angle( eigenVec_1 , <1,0> )
    """
    outputEllPar = []
    for k in range(self.K):
      covMat = matrix([ [ par[k,2]**2. , par[k,4]     ] , \
                        [ par[k,4]     , par[k,3]**2. ] ]) 
      eigenvalue  = eig(covMat)[0]
      eigenvector = eig(covMat)[1] 
      rotAngle = arccos( transpose(eigenvector[:,0])*array([[1],[0]]) ) * 180./pi
      if eigenvector[:,0][1]<0:
        rotAngle *= -1.
      outputEllPar.append( array([sqrt(eigenvalue[0]), sqrt(eigenvalue[1]), rotAngle]) )
    return array( outputEllPar )
