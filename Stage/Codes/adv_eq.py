import numpy as np
from matplotlib import pyplot as plt

#Advection equation

def adv_eq(nt=None,nx=None):
   if nt == None:
      nt = 100.
   if nx == None:
      nx = 100.
   #Initialization
   leng = 100. #meters
   nx,nt,leng=float(nx),float(nt),float(leng)
   T = 4. #days
   dx = leng / nx
   dt = T / nt
   xx = np.linspace(0,leng,nx+1)
   V = dx / dt
   C = V * dt / dx
   conc = np.zeros((nt+1,nx+1))
   conc[0,1] = 1.
   fig = plt.figure()
   plt.ion()
   plt.show()
   cont_t = 0
   while cont_t < nt:
         cont_x = 1
         while cont_x < nx :
               conc[cont_t+1,cont_x] = conc[cont_t,cont_x] - C * (conc[cont_t,cont_x] - conc[cont_t,cont_x-1])
               cont_x = cont_x + 1
         plt.plot(xx,conc[cont_t,:])
         plt.ylim(ymax=1.,ymin=0)
         lines = plt.plot(xx,conc[cont_t,:])
         lines[0].set_ydata(conc[cont_t,:])
         plt.pause(0.5)
         plt.draw()
         fig.clear()
         cont_t = cont_t + 1
   


   
