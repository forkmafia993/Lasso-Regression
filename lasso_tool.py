A fan is rotating with a constant angular velocity, $\omega_o=2\pi(R+2)$ radians/s, 
where R is the last digit of your roll number. 
You are measuring the angle of the fan, $\theta(t)=\omega_ot$ ;
$\theta(t) ∈ [0,2\pi)$, with respect to time t, 
sampled at a frequency Fs=128Hz for a duration of 1s.

import numpy as np
import matplotlib.pyplot as plt

def generate_theta(omega,n):

    theta=(omega*n)%(2*np.pi)
    return theta,n

def setlimit(i):
    if i==0:
        plt.ylim(-5,10)
    elif i==1:
        plt.ylim(-10,10)
    else:
        plt.ylim(-10,20)
 
 def plot_theta(theta,x):

    plt.plot(x,theta,'-o')
    plt.show()

def generate_train_sample(omega,n):
  
    
    np.random.seed(3)
    np.random.shuffle(n)

    
    nts=n[0:10]                      
    nts.sort()
    

    thta,n=generate_theta(omega,nts)     
    return thta,n
