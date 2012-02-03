###Gaussian Process Linear Regression Bishop, 6.4.1

import numpy as np
import matplotlib.pyplot as plt
def GaussKernel(x1,x2,sigma):
	GK=np.exp(-np.power((x1-x2),2)/(2*np.power(sigma,2)))
	return GK
def expoKernel(x1,x2,theta):
	eK=np.exp(-theta*np.abs(x1-x2))
	return eK
def regKernel(x1,x2,theta0,theta1,theta2,theta3):
	rgK=theta0*np.exp(-theta1*np.power((x1-x2),2)/2)+theta2+theta3*x1*x2
	return rgK

x=np.r_[-1:1:0.1]
Gk=np.empty((np.size(x),np.size(x)))
exk=np.empty((np.size(x),np.size(x)))
regk=np.empty((np.size(x),np.size(x)))
sigmaGK=1
thetaeK=1
theta0=9.0
theta1=4.0
theta2=0.0
theta3=5.0
print np.size(x)
for i in range(np.size(x)):
	for j in range(np.size(x)):
		Gk[i,j]=GaussKernel(x[i],x[j],sigmaGK)
		exk[i,j]=expoKernel(x[i],x[j],thetaeK)
		regk[i,j]=regKernel(x[i],x[j],theta0,theta1,theta2,theta3)

mean=np.zeros(np.size(x))
plt.figure(0)
sampNo=5
sampGk=np.random.multivariate_normal(mean,Gk,sampNo)
sampeexk=np.random.multivariate_normal(mean,exk,sampNo)
sampregk=np.random.multivariate_normal(mean,regk,sampNo)

print np.shape(sampGk)
plt.plot(sampGk.T)
plt.savefig('GkSamples')
plt.figure(1)
plt.plot(sampeexk.T)
plt.savefig('exkSamples')
plt.figure(2)
plt.plot(sampregk.T)
plt.savefig('regkSamples')

	
	
	


		
