#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""  This script is to simulate a example of unit vector in the chapter 3.6.4 in book 
Sliding Mode Cotrol Theory and Applications by Christopher Edwars """

__author__ = '{Miguel Angel Pimentel Vallejo}'
__email__ = '{miguel.pimentel@umich.mx}'
__date__= '{19/may/2020}'

#Import the modules needed to run the script
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from control.matlab import *

#Parameters given for the book
R = 1.2
LO = 0.05
JO = 0.125
Ke = 0.6
Kt = 0.6
J = 0.1352
b = 0
A = np.matrix([[0,1,0],[0,-b/JO,Kt/JO],[0,-Ke/LO,-R/LO]])
B = np.matrix([[0],[0],[1/LO]])
C = np.identity(3)

A11 = A[0:2,0:2]
A12 = A[0:2,2]
A21 = A[2,0:2]
A22 = A[2,2]

wn = 2.5
z = 0.95

M = (1/A12[1,0])*np.matrix([[wn**2,2*z*wn]])
S = np.concatenate((M,np.matrix([[1]])),axis=1)

Caphi = -20
Leq = np.linalg.inv(S*B)*S*A
Lrc = Leq + np.linalg.inv(S*B)*Caphi*S

L = 0.046
Aper = np.matrix([[0,1,0],[0,-b/J,Kt/J],[0,-Ke/J,-R/L]])
Bper = np.matrix([[0],[0],[1/L]])
Cper = np.identity(3)

X = np.matrix([[1],[0],[0]])
gamma2 = 0.01

#Proposed parameters that meet with the in book's model
B2 = 1/LO
xiL = (LO-L)/L
xiJ = Kt/JO - Kt/J 

#Proposed parameters that meet to \phi*P_2 + P_2*\phi = -I
P2 = 1/40

#Fuction that contains de DC motor model
def model(t,x):

    xdot = [0,0,0,0,0,0,0,0,0]

    #Switching fuction
    s = S*np.matrix(x[6:9]).T

    #Liner input 
    ul = Lrc*np.matrix(x[6:9]).T
    
    #No linear input
    rho = (1/(9*LO))*( np.absolute(ul) + Ke*np.absolute(x[7]) + R*np.absolute(x[8]) + 5*LO*(M[0,0]**2 + M[0,1]**2)*(np.sqrt(x[6]**2 + x[7]**2 + x[8]**2) + 10*LO*gamma2))
    un = -rho*np.sign(P2*s)

    #Input with out control
    u = 0

    #Original system
    xdot[0:3] = (Aper*np.matrix(x[0:3]).T + Bper*u ).T.tolist()[0]

    #Uncertainty functions
    fm = np.matrix(-(1/LO)*xiL*(Ke*x[4] + R*x[5] - u))
    fu = np.matrix([[0],[-xiJ*(M[0,0]*x[3] - M[0,1]*x[4] )]])
    
    #System with uncertainty
    xdot[3:6] = (Aper*np.matrix(x[3:6]).T + Bper*u +  np.concatenate((fu,fm),axis=0) ).T.tolist()[0]

    #Inuput control
    u = ul + un

    #Uncertainty functions
    fm = np.matrix(-(1/LO)*xiL*(Ke*x[7] + R*x[8] - u))
    fu = np.matrix([[0],[-xiJ*(M[0,0]*x[6] - M[0,1]*x[7] )]])

    #System with control
    xdot[6:9] = (Aper*np.matrix(x[6:9]).T + Bper*u + np.concatenate((fu,fm),axis=0)).T.tolist()[0]

    return xdot

#Initials condictions for the system
x0 = [1,0,0,1,0,0,1,0,0]

#Initial time
t0 = 0

#ODE with Runge Kutta
r = ode(model).set_integrator('dopri5',atol=1.e-3,rtol=1.e-3)
r.set_initial_value(x0, t0)

#Final time
tf = 25

#Step size
dt = 0.006

#Create list to save the result of the solver

#Original system
xo1 = [x0[0]]
xo2 = [x0[1]]
xo3 = [x0[2]]

#System with uncertainty
xu1 = [x0[3]]
xu2 = [x0[4]]
xu3 = [x0[5]]
fum_v = [(-(1/LO)*xiL*(Ke*x0[4] + R*x0[5]))]
fuu_v = [(np.matrix([[0],[-xiJ*(M[0,0]*x0[3] - M[0,1]*x0[4] )]]))[1,0]]

#System with control
xc1 = [x0[6]]
xc2 = [x0[7]]
xc3 = [x0[8]]
s = S*np.matrix(x0[6:9]).T
ul_v = [(Lrc*np.matrix(x0[6:9]).T)[0,0]]
rho = (1/(9*LO))*( np.absolute(ul_v[0]) + Ke*np.absolute(x0[7]) + R*np.absolute(x0[8]) + 5*LO*(M[0,0]**2 + M[0,1]**2)*(np.sqrt(x0[6]**2 + x0[7]**2 + x0[8]**2) + 10*LO*gamma2) )
un_v = [(-rho*np.sign(P2*s)[0,0])]
u_v = [ul_v[0] + un_v[0]]
fcm_v = [(-(1/LO)*xiL*(Ke*x0[7] + R*x0[8] - u_v[0]))]
fcu_v = [(np.matrix([[0],[-xiJ*(M[0,0]*x0[6] - M[0,1]*x0[7] )]]))[1,0]]
t = [t0]

#Loop to solve the ODE
while r.successful() and r.t < tf:
    r.t+dt
    r.integrate(r.t+dt)
    xo1.append(r.y[0])
    xo2.append(r.y[1])
    xo3.append(r.y[2])
    xu1.append(r.y[3])
    xu2.append(r.y[4])
    xu3.append(r.y[5])
    fum_v.append(-(1/LO)*xiL*(Ke*r.y[4] + R*r.y[5]))
    fuu_v.append((np.matrix([[0],[-xiJ*(M[0,0]*r.y[3] - M[0,1]*r.y[4] )]]))[1,0])
    xc1.append(r.y[6])
    xc2.append(r.y[7])
    xc3.append(r.y[8])
    x = [r.y[6],r.y[7],r.y[8]]
    s = S*np.matrix(x).T
    ul = Lrc*np.matrix(x).T
    rho = (1/(9*LO))*( np.absolute(ul) + Ke*np.absolute(x[1]) + R*np.absolute(x[2]) + 5*LO*(M[0,0]**2 + M[0,1]**2)*(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) + 10*LO*gamma2))
    un = -rho*np.sign(P2*s)
    u = ul + un
    fm = - (1/LO)*xiL*(Ke*x[1] + R*x[2] - u)
    fu = np.matrix([[0],[-xiJ*(M[0,0]*x[0] - M[0,1]*x[1] )]])
    ul_v.append(ul[0,0])
    un_v.append(un[0,0])
    u_v.append(u[0,0])
    fcm_v.append(fm[0,0])
    fcu_v.append(fu[1,0])
    t.append(r.t)
       
#Plot results

#Plot the original system
plt.figure()
plt.title('Original system')
plt.plot(t,xo1, label = "$x_1$")
plt.plot(t,xo2, label = "$x_2$")
plt.plot(t,xo3, label = "$x_3$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()
plt.xlim(0,5)

#Plot the system with uncertainty signals
plt.figure()
plt.title('System with uncertainty signals')
plt.plot(t,xu1, label = "$x_1$")
plt.plot(t,xu2, label = "$x_2$")
plt.plot(t,xu3, label = "$x_3$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()

#Plot the uncertainty signals
plt.figure()
plt.title('Uncertainty signals')
plt.plot(t,fum_v, label = "$f_m$")
plt.plot(t,fuu_v, label = "$f_u$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()

#Plot the system with control
plt.figure()
plt.title('System with control')
plt.plot(t,xc1, label = "$x_1$")
plt.plot(t,xc2, label = "$x_2$")
plt.plot(t,xc3, label = "$x_3$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()
plt.xlim(0,5)

#Plot the signal control
plt.figure()
plt.title('Input control')
plt.plot(t,ul_v, label="$u_l$")
plt.plot(t,un_v, label="$u_n$")
plt.plot(t,u_v, label="$u$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()
plt.xlim(0,5)

#Plot the uncertainty signals
plt.figure()
plt.title('Uncertainty signals')
plt.plot(t,fcm_v, label = "$f_m$")
plt.plot(t,fcu_v, label = "$f_u$")
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()
plt.xlim(0,5)

plt.show()