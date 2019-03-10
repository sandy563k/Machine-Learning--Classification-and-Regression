import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    
    labels = np.unique(y)
    means = np.zeros(shape = (X.shape[1], len(labels)))
    
    for label in labels:
        elements = X[y[:,0] == label]
        means[:,int(label)-1] = np.mean(elements, axis=0)
    
    covmat = np.cov(X.transpose())
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    uniqueClasses = np.unique(y)
    means_matrix = []
    covmats = [np.zeros((X.shape[1],X.shape[1]))] * uniqueClasses.size
    i=0;
    for group in uniqueClasses:
        
        Xg = X[y.flatten() == group, :]
        means_matrix.append(Xg.mean(0))
        
        Yg = Xg.T
        covmats[i] = np.cov(Yg)
        i=i+1
    
    means=np.transpose(np.asarray(means_matrix))
    
    return means,covmats
    
    
    
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    cov_inv = np.linalg.inv(covmat)
    labels = np.shape(means)[1]
    total_pdf = np.zeros(shape=(Xtest.shape[0],labels))
    result = 0
    
    for i in range(0,labels):
        for j in range(0,Xtest.shape[0]):
            xSubMean = Xtest[j,:]-(means[:,i]).transpose()
            pdf = np.dot(np.dot(xSubMean,cov_inv),(xSubMean))
            total_pdf[j,i] = pdf
    myClasses = np.zeros(shape=(Xtest.shape[0],1))
    myClasses = (np.argmin(total_pdf,axis=1)) + 1
    for i in range(0, Xtest.shape[0]):
        if(ytest[i] == myClasses[i]):
            result = result + 1
        ypred = myClasses.reshape(Xtest.shape[0],1)
    
    acc = (result/len(ytest))*100
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    classes = means.shape[1]
    total = np.zeros((Xtest.shape[0],classes))
    
    for i in range(classes):
        sigma = np.linalg.det(covmats[i])
        inverse_sigma = np.linalg.inv(covmats[i])
        denom = np.sqrt(2 * np.pi) * np.square(sigma)
        xMinusU = Xtest - means[:,i]
        total[:,i] = np.exp(-0.5*np.sum(xMinusU * np.dot(inverse_sigma, xMinusU.T).T,1))/denom
    
    label = np.argmax(total,1)
    label = label + 1
    ytest = ytest.reshape(ytest.size)
    acc = 100*np.mean(label == ytest)
    
    return acc, label

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD   

    invX = np.linalg.pinv(X)
    w = np.dot(invX, y)
                                                    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD SANDEEP

    #Formula to implement
    # w =  inverse((λI + transpose(X) * X)) * transpose(X) * y;

    N = X.shape[0]
    d = X.shape[1]

    # I
    identityMatrix = np.identity(d)

    # (λI + transpose(X) * X)
    a = np.dot(X.T, X) + (lambd * identityMatrix)

    # inverse((λI + transpose(X) * X))
    inverse_a = inv(a)


    # inverse((λI + transpose(X) * X)) * transpose(X) * y
    w = np.dot(inverse_a, np.dot(X.T, y))

    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    N = Xtest.shape[0]
    mse = np.sum(np.square(ytest - np.dot(Xtest,w)))/N
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD   
    w = w.reshape(65,1)
    # Formuls: error = 0.5*((y - w*X).T * (y - W)) + 0.5*lambd(w.T*w)
    err_var1 = y - np.dot(X,w)
    err_var2 = 0.5*lambd*np.dot(w.transpose(),w)
    
    error = 0.5*np.dot(err_var1.transpose(),err_var1) + err_var2
    
    # Formula: error_grad = (X.T*X)W - X.T*y + lambd*w
    
    err_grad_var1 = np.dot(np.dot(X.transpose(),X), w)
    err_grad_var2 = np.dot(X.transpose(),y)
    err_grad_var3 = lambd*w
    
    error_grad = (err_grad_var1 - err_grad_var2) + err_grad_var3
    error_grad = error_grad.flatten()
    
    return error, error_grad
   

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = x ** i
    return Xd   
   

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
# =============================================================================


k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
    
    
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
# 
# 
# # Problem 5
pmax = 7
lambda_opt = 0.059 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
# 
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
# 
# =============================================================================
