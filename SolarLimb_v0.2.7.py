# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 14:15:16 2018

@author: Krad
"""
# to prevent error of division by 0
from __future__ import division
from __future__ import unicode_literals
#from __future__ import print_function

import os, os.path
import matplotlib
from scipy.interpolate import *
from pylab import *
import matplotlib.lines as mlines
# graphical interface libraries
import wx
# for making a scrollable panel
import wx.lib.scrolledpanel
from wx.lib.scrolledpanel import ScrolledPanel
# for browsing files
import wx.lib.filebrowsebutton
import wx.lib.agw.aui as aui
# for working with gridTables
import wx.grid as gridlib
# regular pubsub import - for transferring variables
from wx.lib.pubsub import pub
# the easiest way to get the full statistical results for the fit is to use statsmodels
import statsmodels.api as sm 
# to show borders around sizers and other stuff - making app more interactive
#import wx.lib.inspection
#import wx.lib.mixins.inspection
#wx.lib.inspection.InspectionTool().Show()
#import random
# for Showing, reshaping and calculating data
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import random
import wx.lib.mixins.listctrl  as  listmix
#from Tkinter import *
#from matplotlib.figure import Figure

###############################################################################
''' Class for reshaping and calculation all things '''
class Calculator():
    
    ''' Constructor of the class '''
    def __init__(self):
        #############################################
        ## Definitions of values for Our Analysis ##
        #############################################
        # variable to distinguish between even and odd indecies of gridsizer
        self.number = 0
        # the index of data in the list to find the right one
        self.index = 0
        
        self.im = []
        self.imresh = []
        # for the intensity curve of each picture
        self.imresh_intensity_curve = []
        # chosing a threshold
        self.threshold = 0
        self.Cx = [0 for col in range(100)]
        self.Cy = [0 for col in range(100)]
        #To find the center of the sun
        self.FirstPoint = []
        self.SecondPoint = []
        self.ThirdPoint = []
        self.FourthPoint = []
        #---------------------------------------------------------------------#
        I = 1619
        J = 1219
        # For data Analysis part
        self.imreshSum = [[0 for col in range(I)] for row in range(J)]
        self.imSum = [[0 for col in range(I)] for row in range(J)]
        self.SkySum = [[0 for col in range(I)] for row in range(J)]
        self.SkySumFinal = [[0 for col in range(I)] for row in range(J)]
        self.imreshSumFinal = [[0 for col in range(I)] for row in range(J)]
        #---------------------------------------------------------------------#
        self.rout = [0 for x in range(10)]
        self.rin = [0 for x in range(10)]
        self.r = [0 for x in range(10)]
        self.delr = [0 for x in range(9)]
        self.miuk = [0 for row in range(9)]
        self.delmiu = [0 for row in range(9)]
        self.miukSum = 0
        self.miukFIN = [0 for row in range(9)]

        self.x = 0
        #---------------------------------------------------------------------#
        self.Ik = [0 for row in range(10)]
        self.Il0 = 0   #[0 for row in range(10)]
        self.Il = [0 for row in range(10)]
        self.S = [0 for row in range(9)]
        #self.S = 0
        self.Ilratio = [0 for row in range(10)]
        self.miu = 0
        self.sIk = [0 for row in range(1619)]
        self.IkN = [[0 for col in range(1619)] for row in range(1219)]
        self.IkSum = [[0 for col in range(1619)] for row in range(1219)]
        self.IkFIN = [[0 for col in range(1619)] for row in range(1219)]
        self.IlSum = [[0 for col in range(10)] for row in range(10)]
        self.IlFIN = [[0 for col in range(10)] for row in range(10)]
        
        
        ########################
        ########################
        self.Error = [0 for col in range(9)]
        self.ErrorSum = 0
        #########################
        #---------------------------------------------------------------------#
        # The bunch of constants
        #self.lambd = [1, 420e-9, 547e-9, 871e-9, 648e-9]
        #self.Ilambd = [1, 3.6e13, 4.5e13, 1.6e13, 2.8e13]
        #self.deltalambd = [17.5, 45, 16.5, 118, 78.5]
        
        self.hPl = 6.626e-34
        self.kBolz = 1.38e-23
        self.clight = 3e8
        #What changes per filter:
        #self.Ratio1 = [0 for col in range(7)]
        #self.Ratio2 = [0 for col in range(7)]
        
        #self.p2Array = [0 for col in range(7)]
        #self.IlArray = [0 for col in range(7)]
        #self.TeffArray = [0 for col in range(7)]
        #self.TtaulArray = [0 for col in range(7)]
        
        self.Ratio1 = [0 for col in range(100)]
        self.Ratio2 = [0 for col in range(100)]
        
        self.p2Array = [0 for col in range(100)]
        self.IlArray = [0 for col in range(100)]
        self.TeffArray = [0 for col in range(100)]
        self.TtaulArray = [0 for col in range(100)]

        
        self.delMiuk = 0        
    
    def ShowImages(self, DataPanel, listLength, listPaths, 
                   ResultPanel, globalPageIndex):
        
        print ResultPanel.comboSelection[0]
        print ResultPanel.comboSelection[1]
        print ResultPanel.comboSelection[2]

        # taking the value of threshold
        self.threshold = ResultPanel.threshold
        #stands for filter in Ruslan's interpretation goes through values from 2 to 5
        #important to remember it is not index or y, but anather variabl for another loop
        self.f = globalPageIndex# just for testing purposes
        print ('self.f', self.f)
        #---------------------------------------------------------------------#
        #retrieving the values of constants
        #self.lambd[self.f] = ResultPanel.comboSelection[0]
        #self.Ilambd[self.f] = ResultPanel.comboSelection[1]
        #self.deltalambd[self.f] = ResultPanel.comboSelection[2]
        self.lambd = ResultPanel.comboSelection[0]
        self.Ilambd = ResultPanel.comboSelection[1]
        self.deltalambd = ResultPanel.comboSelection[2]
        # variable for making matrix of data
        sizerRow = listLength * 2
        self.index = 0
        #Creating a sizer which will hold all canvases with data
        DataPanel.gridSizerShow = wx.GridSizer(rows = sizerRow, cols = 2, hgap=1, vgap=1)
        # This loop is for showing obtained images
        for y in xrange(0, sizerRow):
            DataPanel.listShowFigure.append(plt.figure())
            # creating the axes
            DataPanel.listShowAxe.append(DataPanel.listShowFigure[y].add_subplot(1, 1, 1))
            DataPanel.listShowFigureCanvas.append(FigureCanvas(DataPanel, -1, DataPanel.listShowFigure[y]))
            # To put all observed data in the left column we need to distinguish
            # between even and odd numbers
            if self.number % 2 == 0:
                # Drawing the data on the canvas
                DataPanel.listShowFigure[y].set_canvas(DataPanel.listShowFigureCanvas[y])
                # Clearing the axes
                DataPanel.listShowAxe[y].clear()
                # reading an image from the same folder
                self.im = misc.imread(listPaths[self.index])
                # rehsping an array (changing the ) 
                self.imresh = self.im.reshape(1219,1619)
                # reading an image from the same folder
                self.imresh_intensity_curve = misc.imread(listPaths[self.index])
                self.imresh_intensity_curve_reshaped = self.imresh_intensity_curve.reshape(1219,1619)
                #print ("reshaped image ", self.imresh_intensity_curve_reshaped)
                # for drawing the intensity curve (it will evenatually forget 
                #the iniial value, so we need to save it somewhere)
                #self.imresh_intensity_curve = self.im

                # the reshaping for sky images
                self.Sky = misc.imread(DataPanel.listSkyPaths[self.index])
                self.Skyresh = (self.Sky.reshape(1219,1619))/1000
                ####################
                ##Reshaping images##
                ####################
                ''' Making the condition if image less than threshold etc. '''
                for j in range(0,1619):
                    for i in range(0,1219):
                        if (self.imresh[i,j] < self.threshold):
                            self.imresh[i,j] = 0
                        else: self.imresh[i,j] = 1

                ##################################
                ##Finding the center of each Sun##
                ##################################
                # Now, it is time to find the center of the Sun
                found = False
                for j in range(0,1619):
                    if found:
                        break
                    for i in range(0,1219):
                        if (self.imresh[i,j] == 1):
                            self.FirstPoint = [j,i]
                            found = True
                            break  
        
        
                found = False
                for j in range(1619-1, -1, -1):
                    if found:
                        break
                    for i in range(0,1219):
                        if (self.imresh[i,j] == 1):
                            self.SecondPoint = [j,i]
                            found = True
                            break             
        
        
                found = False            
                for i in range(0,1219):
                    if found:
                        break
                    for j in range(0,1619):
                        if (self.imresh[i,j] == 1):
                            self.ThirdPoint = [j,i]
                            found = True
                            break   
        
                found = False            
                for i in range(1219-1, -1, -1):
                    if found:
                        break
                    for j in range(0, 1619):
                        if (self.imresh[i,j] == 1):
                            self.FourthPoint = [j,i]
                            found = True
                            break 
        
                #Now let us mathematically find an intersection of these 2 lines:
                # Center of the sun is C = (Cx, Cy)
                self.Cx[self.index] = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[0]-self.FourthPoint[0])-(self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0]))//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]-self.FourthPoint[0]))
                self.Cy[self.index] = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0]))//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]-self.FourthPoint[0]))
                # Let's not shift the center of the next image to the center of the first one in order to find an average
                self.imreshSum = self.imreshSum + (np.roll(np.roll(self.imresh_intensity_curve_reshaped, (self.Cy[0]-self.Cy[self.index]), axis=None), (self.Cx[0]-self.Cx[self.index]), axis=0))
                self.SkySum = self.SkySum + self.Skyresh
                #And plot everything, together with the intersection point:                
                #---------------------------------------------------------#
                # invoking native method for showing images -Sun
                DataPanel.listShowAxe[y].imshow(self.imresh_intensity_curve_reshaped)
                
                # Showing the intersection of two lines and the center of each image
                DataPanel.listShowAxe[y].plot((self.FirstPoint[0], self.SecondPoint[0]), (self.FirstPoint[1], self.SecondPoint[1]), color = 'g')
                DataPanel.listShowAxe[y].plot((self.ThirdPoint[0], self.FourthPoint[0]), (self.ThirdPoint[1], self.FourthPoint[1]), color = 'g')
                DataPanel.listShowAxe[y].plot([self.Cx], [self.Cy], marker='x', color='g')
                #DataPanel.listAxe[x].imshow(Center[x], cmap="hot")
                # draw canvas
                DataPanel.listShowFigureCanvas[y].draw()
                
                #################
                ##Data Analysis##
                #################
                
                self.R1 = self.Cx[self.index] - self.FirstPoint[0]
                self.R2 = self.SecondPoint[0] - self.Cx[self.index]
                self.R = (self.R1+self.R2)/2
                
                self.r1 = self.FirstPoint[0]
                self.r2 = self.FirstPoint[0] + (2*self.R/9)
                self.x = 0
                self.y = (2*self.R/9)
                ###########   ###########  ##############
                # We should find the Distance parameters $\mu$ and Intensity 
                #ratios for each k-th region of the sun
                # Then conduct the Polynomial fit of their dependance 
                #=> relating to the place parallel approximation
                # we can find the optical depth $\tau$ and then 
                #effective temperature of the sun (which is our goal)  
                self.R = float(self.SecondPoint[0]-self.Cx[self.index])
                self.rin[0] = float(0)
                self.rout[0] = float(self.R/9)
                
                # Here we dicvide our sun for nine regions (concentric rings) 
                #with the center in the solar center,
                # and look for distance parameter and intensity ratio:
                for k in range(0,9):
                    self.DEV = 0
                    self.sIk = 0
                    self.r[k] = (self.rin[k]+self.rout[k])/2         
                    self.delr[k] = (self.rin[k]-self.rout[k])/2 
                    
                    self.Error[k] = self.delr[k]/self.r[k]
        
                    self.miuk[k] = (np.sqrt(1-(self.r[k]**2/self.R**2))).astype(float)

        
                    self.delmiu[k] = (self.r[k]*self.delr[k])/(np.sqrt(1- (self.r[k]**2/self.R**2)))
                    self.IkN = np.asarray([[0 for col in range(1619)] for row in range(1219)])
                    #Now we determine the average intensity for each k-th region
                    
                    self.Ik[k] = (self.imresh_intensity_curve_reshaped[self.Cy[self.index],
                                  (self.Cx[self.index]+int(self.rin[k]))]+self.imresh_intensity_curve_reshaped[self.Cy[self.index],(self.Cx[self.index]+int(self.rout[k]))])//2
                    
                    self.dev = ((self.imresh_intensity_curve_reshaped[self.Cy[self.index],
                                                                      (self.Cx[self.index]+int(self.rout[k]))]**2-(self.Ik[k])**2))
                    self.DEV = self.DEV + self.dev
                
                    #Standard deviation sI of Intensity Ii:
                    self.sIk = np.sqrt(np.fabs(self.DEV)/(2))
                    
                    self.rin[k+1] = self.rin[k] + (self.R/9)
                    self.rout[k+1] = self.rout[k] + (self.R/9) 
        
                    self.Il0 = self.imresh_intensity_curve_reshaped[self.Cx[self.index],self.Cy[self.index]]
                    self.Il[k] = self.Ik[k].astype(float)/self.Ik[0]
                    self.ErrorSum += abs(self.Error[k])

                self.ErrorI = self.ErrorSum/9

                print ("miuk", self.miuk)
                print ("Ik ", self.Ik)
        
                #Weighted mean overl all the pictures for each filter:
                np.array(self.IlSum)[self.f][k] += np.array(self.Il)[k]
                #####################
                self.IlFIN[self.f][k] = float(self.IlSum[self.f][k])/(self.index+1) 
                
            else:
                # Drawing the data on the canvas
                DataPanel.listShowFigure[y].set_canvas(DataPanel.listShowFigureCanvas[y])
                # Clearing the axes
                DataPanel.listShowAxe[y].clear()               
                # Intensities for every picture in cycle:
                DataPanel.listShowAxe[y].set_ylabel('Intensity (ADU)')
                DataPanel.listShowAxe[y].set_xlabel('Linear position')
                # for each variable we need to account the skies to get reasonable values for the intensity curves
                image_minus_sky = self.imresh_intensity_curve_reshaped - self.Skyresh
                print ('image_minus_sky', image_minus_sky)
                #Vertical profile
                DataPanel.listShowAxe[y].plot(image_minus_sky[(self.ThirdPoint[1]-50):(self.FourthPoint[1]+50),self.Cx[self.index]], 'r')
                #Horizontal profile
                DataPanel.listShowAxe[y].plot(image_minus_sky[self.Cy[self.index],(self.FirstPoint[0]-50):(self.SecondPoint[0]+50)], 'b')
                # draw canvas
                DataPanel.listShowFigureCanvas[y].draw()
                # keeping this index he same until now to draw nice intensity plot
                self.index = self.index + 1
            
            self.number = self.number + 1

            #adding plots to sizer
            DataPanel.gridSizerShow.Add(DataPanel.listShowFigureCanvas[y], 0, wx.ALL)
            
            
        # Setting a sizer to a panel   
        DataPanel.SetSizer(DataPanel.gridSizerShow)
        #DataPanel.Update()

        self.imreshSumFinal = self.imreshSum/self.index
        self.SkySumFinal = self.SkySum/self.index
        self.imreshSumFinal = self.imreshSumFinal - self.SkySumFinal
        
        print ('self.imreshSumFinal', self.imreshSumFinal)
        np.savetxt('data_a.txt', self.imreshSumFinal)
        ###########################################################
        ## Looking for the center of the Sun (resulting picture) ##
        ###########################################################
        b = np.loadtxt('data_a.txt')  
        c = np.loadtxt('data_a.txt')
        
        for j in range(0,1619):
            for i in range(0,1219):
                if (b[i,j] < self.threshold):
                    b[i,j] = 0
                else: b[i,j] = 1
    
        # 4 cycles in order to find (left, right, top and bottom) points of the sun => 
        #then we will connect them with the lines and find the center by intersecting these 2 lines
        found = False
        for j in range(0,1619):
            if found:
                break
            for i in range(0,1219):
                if (b[i,j] == 1) and (b[i+1,j] == 1):
                    self.FirstPoint = [j,i]
                    found = True
                    break  
    
    
        found = False
        for j in range(1619-1, -1, -1):
            if found:
                break
            for i in range(0,1219):
                if (b[i,j] == 1) and (b[i+1,j] == 1):
                    self.SecondPoint = [j,i]
                    found = True
                    break             
    
    
        found = False            
        for i in range(0,1219):
            if found:
                break
            for j in range(0,1619):
                if (b[i,j] == 1) and (b[i+1,j] == 1):
                    self.ThirdPoint = [j,i]
                    found = True
                    break   
    
        found = False            
        for i in range(1219-1, -1, -1):
            if found:
                break
            for j in range(0, 1619):
                if (b[i,j] == 1) and (b[i+1,j] == 1):
                    self.FourthPoint = [j,i]
                    found = True
                    break
        #NAgain, mathematical formula for the intersection of the two lines (i.e., center of the sun): 
        self.Gx = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[0]-self.FourthPoint[0])
             -(self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0])
             )//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*
                (self.ThirdPoint[0]-self.FourthPoint[0]))
    
        self.Gy = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])
             -(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0])
             )//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*
                (self.ThirdPoint[0]-self.FourthPoint[0]))
        # Drawing the data on the canvas
        ResultPanel.listFigure[0].set_canvas(ResultPanel.listFigureCanvas[0])
        # Clearing the axes
        ResultPanel.listAxe[0].clear()
        ResultPanel.listAxe[0].imshow(c)
        # Setting the axes labels
        ResultPanel.listAxe[0].set_xlabel(r'Linear position')
        ResultPanel.listAxe[0].set_ylabel('Intensity (ADU)')
        # Setting the central points of the sun
        ResultPanel.listAxe[0].plot([self.FirstPoint[0]], [self.FirstPoint[1]], marker='^', color='g')
        ResultPanel.listAxe[0].plot([self.SecondPoint[0]], [self.SecondPoint[1]], marker='o', color='r')
        ResultPanel.listAxe[0].plot([self.ThirdPoint[0]], [self.ThirdPoint[1]], marker='s', color='y')
        ResultPanel.listAxe[0].plot([self.FourthPoint[0]], [self.FourthPoint[1]], marker='^', color='y')
        ResultPanel.listAxe[0].plot((self.FirstPoint[0], self.SecondPoint[0]), (self.FirstPoint[1], self.SecondPoint[1]), color = 'g')
        ResultPanel.listAxe[0].plot((self.ThirdPoint[0], self.FourthPoint[0]), (self.ThirdPoint[1], self.FourthPoint[1]), color = 'g')
        
        # Putting the center to the picture
        ResultPanel.listAxe[0].plot([self.Gx], [self.Gy], marker='x', color='g')
        ResultPanel.listFigureCanvas[0].draw()
        ##########################################
        ##The intensity plot (resulting picture)##
        ##########################################
        # Drawing the data on the canvas
        ResultPanel.listFigure[1].set_canvas(ResultPanel.listFigureCanvas[1])
        # Clearing the axes
        ResultPanel.listAxe[1].clear()

        ResultPanel.listAxe[1].plot(self.imreshSumFinal[
                (self.ThirdPoint[1]-50):(self.FourthPoint[1]+50),self.Gx], 'r')
        #Horizontal profile

        ResultPanel.listAxe[1].plot(self.imreshSumFinal[
                self.Gy,(self.FirstPoint[0]-50):(self.SecondPoint[0]+50)], 'b')
            
        ResultPanel.listFigureCanvas[1].draw()
        ####################
        ## Polynomial fit ##
        ####################
        #We should firstly make both arrays having the same size 
        #(Il array has one extra element because it helped us with the calculation of its elements)
    
        #Calculating \tau, optical depth in order to find the source function 
        # (with source function we can calculate an effective temperature from the Planck distribution)
    
        del self.IlFIN[-1]
        del self.Il[-1]
        print('Intensity ratio: ' + str(self.Il))
        
        self.miuk1 = [0 for x in range(5)]
        self.Il1 = [0 for x in range(5)]
    
        j=0;
        for i in range (9):
            if self.miuk[i] < 0.9:
                self.miuk1[j] = self.miuk[i]
                self.Il1[j]=self.Il[i]
                j=j+1
        
        print ("miuk1 ", self.miuk1)
        print ("Il1 ", self.Il1)
        
        n=3
        # roughly speaking it provides you with any possible test
        results = sm.OLS(self.Il1, np.vander(self.miuk1, n)).fit()
        print (results.summary())
        # these parameters should be the same as for polyfit (and they are the same,
        # the only difference here is that n=3, I don't know why:))
        print ('Parameters:', results.params)
        self.p2 = results.params
        self.p2Array[self.f] = self.p2
        print('Standard errors: ', results.bse)

        print('R2: ', results.rsquared)

        #Coefficients
        global globalA0
        global globalA1
        global globalA2
        #Errors to be shown in the final table
        global globalA0Error
        global globalA1Error
        global globalA2Error
        # Result for R squared test to be shown in the resulting table
        global globalRSquaredTest
        
        print(" ")
        print("p2 = " + str(self.p2))
        print("a0 = " + str(self.p2[2]))
        print("a1 = " + str(self.p2[1]))
        print("a2 = " + str(self.p2[0]/2))
        print(" ")  
        
        #getting the values of the coefficients for one particular filter
        globalA0 = self.p2[2]
        globalA1 = self.p2[1]
        globalA2 = self.p2[0]/2
        # errors for coefficients
        globalA0Error = results.bse[2]
        globalA1Error = results.bse[1]
        globalA2Error = results.bse[0]/2
        # R squared test
        globalRSquaredTest = results.rsquared
        
        self.tau = np.arange(0., 2., 0.01)
        #self.S = (self.Ilambd[self.f])*(self.p2[2] + self.tau*self.p2[1] + (self.tau**2)*self.p2[0]/2 )
        self.S = (self.Ilambd)*(self.p2[2] + self.tau*self.p2[1] + (self.tau**2)*self.p2[0]/2 )
    
        #self.Ratio1[self.f] = (self.hPl*self.clight)/((self.kBolz*self.lambd[self.f]))
        self.Ratio1[self.f] = (self.hPl*self.clight)/((self.kBolz*self.lambd))
        print ("Ratio1 ", self.Ratio1)
        #self.Ratio2[self.f] = (2*self.hPl*(self.clight**2))/(self.lambd[self.f]**5)
        self.Ratio2[self.f] = (2*self.hPl*(self.clight**2))/(self.lambd**5)
        print ("Ratio2 ", self.Ratio2)
    
        self.Ttaul = self.Ratio1[self.f] / (np.log(1 + self.Ratio2[self.f]/self.S ))

        self.tau23 = 0.666
        #self.Seff = (self.Ilambd[self.f])*(self.p2[2] + (self.tau23)*self.p2[1] + (self.tau23**2)*self.p2[0]/2 )
        self.Seff = (self.Ilambd)*(self.p2[2] + (self.tau23)*self.p2[1] + (self.tau23**2)*self.p2[0]/2 )
        
        self.Teff = self.Ratio1[self.f] / (np.log(1 + self.Ratio2[self.f]/self.Seff))
        # calculating the result error for the coefficients
        k=9
        WholeError = ((results.bse[2]/results.params[2])**2 + (results.bse[1]/results.params[1])**2
                      +(results.bse[0]/results.params[0]/2)**2
                      +2*(results.bse[2]/results.params[2])*(results.bse[1]/results.params[1])
                      +2*(results.bse[1]/results.params[1])*(results.bse[0]/results.params[0]/2)
                      +2*(results.bse[0]/results.params[0]/2)*(results.bse[2]/results.params[2])) 
        #SEr = np.sqrt(WholeError**2 + self.deltalambd[self.f]**2 )
        SEr = np.sqrt(WholeError**2 + self.deltalambd**2 )
        #TeffError = np.sqrt((self.deltalambd[self.f])**2 + (np.var(self.miuk)/(k+1))**2 + SEr**2
        #               +2*self.deltalambd[self.f]*np.var(self.miuk)/(k+1) +2*self.deltalambd[self.f]*SEr +2*SEr*np.var(self.miuk)/(k+1))
        TeffError = np.sqrt((self.deltalambd)**2 + (np.var(self.miuk)/(k+1))**2 + SEr**2
                       +2*self.deltalambd*np.var(self.miuk)/(k+1) +2*self.deltalambd*SEr +2*SEr*np.var(self.miuk)/(k+1))
        
        self.IlArray[self.f] = self.Il
        self.TeffArray[self.f] = self.Teff
        self.TtaulArray[self.f] = self.Ttaul
        
        self.polyval = np.polyval(self.p2Array[self.f],self.miuk1)
        #---------------------------------------------------------------------#
        # assiging global variables to retrieve their values later
        global globalTeff
        globalTeff = self.Teff
        global globalTeffError
        globalTeffError = TeffError
        global globalTtaul
        globalTtaul = self.Ttaul
        global globalIl
        globalIl = self.Il
        
        global globalmiuk
        globalmiuk = self.miuk
        global globalmiuk1
        globalmiuk1 = self.miuk1
        
        global globalPolyval
        globalPolyval = self.polyval
        
        self.ColShape = ['rv', 'rv', 'bs', 'c.', 'm^', 'b:', 'rv', 'rv', 'bs', 'c.', 'm^', 'b:', 'rv', 'rv', 'bs', 'c.', 'm^', 'b:']
        self.ColShape2 = ['rv', 'rv', 'r-', 'b-.', 'm--', 'b:', 'rv', 'rv', 'r-', 'b-.', 'm--', 'b:', 'rv', 'rv', 'r-', 'b-.', 'm--', 'b:']
        self.Color = ['y', 'm', 'r', 'b', 'c', 'm', 'y', 'm', 'r', 'b', 'c', 'm', 'y', 'm', 'r', 'b', 'c', 'm']
        self.Color2 = ['y', 'r', 'r', 'b.', 'm', 'b', 'y', 'r', 'r', 'b.', 'm', 'b', 'y', 'r', 'r', 'b.', 'm', 'b']
        self.Shape = [u'D', u'v', u's', u'.', u'^', u'o',u'D', u'v', u's', u'.', u'^', u'o',u'D', u'v', u's', u'.', u'^', u'o'] # I changed the first one from u'--' to u'd'
        self.Shape2 = [u'--', u'-', u'--', u'-.', u'--', u':', u'--', u'-', u'--', u'-.', u'--', u':', u'--', u'-', u'--', u'-.', u'--', u':']
        self.lines = [0 for col in range(7)]
        self.labels = [0 for col in range(7)]
        
        #Drawing the pre-last curve
        # Drawing the data on the canvas
        ResultPanel.listFigure[2].set_canvas(ResultPanel.listFigureCanvas[2])
        # Clearing the axes
        ResultPanel.listAxe[2].clear()
        ResultPanel.listAxe[2].set_xlabel(r'$\mu$')
        ResultPanel.listAxe[2].set_ylabel('Intensity ratio I(0,' + r'$\mu$)' + '/I(0,1)')
        ResultPanel.listAxe[2].scatter(self.miuk, self.IlArray[self.f], marker=self.Shape[self.f])      
        ResultPanel.listAxe[2].plot(self.miuk1, self.polyval)   

        ResultPanel.listFigureCanvas[2].draw()
        
        #Drawing the last curve
        # Drawing the data on the canvas
        ResultPanel.listFigure[3].set_canvas(ResultPanel.listFigureCanvas[3])
        # Clearing the axes
        ResultPanel.listAxe[3].clear()
        ResultPanel.listAxe[3].set_xlabel(r'$\tau$')
        ResultPanel.listAxe[3].set_ylabel('Temperature T(' + r'$\tau$)')

        ResultPanel.listAxe[3].plot(self.tau, self.Ttaul, linestyle = self.Shape2[self.f], color=self.Color[self.f])
        ResultPanel.listAxe[3].plot(self.tau23, self.Teff, marker = '*')
        ResultPanel.listFigureCanvas[3].draw()
        
    
###############################################################################
''' Our TabPanel Class to calculate and show things:
    It is the Main Panel which will contain others, 
    such as DataPanel and ResultPanel '''
class TabPanel(wx.Panel):
    """
    A simple wx.Panel class
    """
    #----------------------------------------------------------------------
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        #First retrieve the screen size of the device
        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]/2

        ''' DataPanel Class will retrieve and show data obtained '''
        DataPanel = ScrolledPanel(self,-1, size=(screenWidth,400),
                                                    style=wx.SIMPLE_BORDER)
        DataPanel.SetupScrolling()
        DataPanel.SetBackgroundColour('#FDDF99')
                                      
        self.dataPanel = DataPanel
        
        ''' ResultPanel Class will show us the result of the calculation '''                           
        ResultPanel = wx.Panel(self,-1,size=(screenWidth,400), style=wx.SIMPLE_BORDER)
        ResultPanel.SetBackgroundColour('#FFFFFF')           
        
        #creating a sizer to manage two panels on parent panel
        self.panelSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panelSizer.Add(DataPanel, 0, wx.EXPAND|wx.ALL,border=5)
        self.panelSizer.Add(ResultPanel, 0, wx.EXPAND|wx.ALL,border=5)
        self.SetSizer(self.panelSizer)
        #---------------------------------------------------------------------#
        DataPanel.listPaths = []
        DataPanel.listLength = 0
        # This is for additional information for instance skyiamges which we are using in our analysis
        DataPanel.listSkyPaths = []
        DataPanel.listSkyLength = 0
        # variable to distinguish between even and odd indecies of gridsizer
        self.number = 0
        # the index of data in the list to find the right one
        self.index = 0
        # Chosing a threshold
        DataPanel.listShowFigure = [] # empty list
        DataPanel.listShowAxe = []
        DataPanel.listShowFigureCanvas = []
        
        DataPanel.listFigure = [] # empty list
        DataPanel.listAxe = []
        DataPanel.listFigureCanvas = []
        #---------------------------------------------------------------------#
        ResultPanel.label1 = wx.StaticText(ResultPanel, label = " Browse your pictures:")
        ResultPanel.label2 = wx.StaticText(ResultPanel, label = " Additional information:")
        #---------------------------------------------------------------------#
        ResultPanel.labelChoseFile = wx.StaticText(ResultPanel, label = " Choose the file(s): ")
        ResultPanel.chosenPath = wx.TextCtrl(ResultPanel, value = "", size = (-1, -1))
        # for testing aponing button
        ResultPanel.browseButton = wx.Button(ResultPanel, -1, "Open")
        # Adding an event handling to the browseButton
        ResultPanel.browseButton.Bind(wx.EVT_BUTTON, lambda event, 
                               arg1 = DataPanel, arg2 = ResultPanel: self.onOpenFile1(event, arg1, arg2))
        #---------------------------------------------------------------------#
        # For the additional information
        ResultPanel.labelChoseAdditionalFile2 = wx.StaticText(ResultPanel, label = " Choose the file(s): ")
        ResultPanel.chosenAdditionalPath = wx.TextCtrl(ResultPanel, value = "", size = (-1, -1))
        # for testing oponing button
        ResultPanel.browseAdditionalButton = wx.Button(ResultPanel, -1, "Open")
        # Adding an event handling to the browseButton
        ResultPanel.browseAdditionalButton.Bind(wx.EVT_BUTTON, lambda event, 
                               arg1 = DataPanel, arg2 = ResultPanel: self.onOpenFile2(event, arg1, arg2))
        #---------------------------------------------------------------------#
        ResultPanel.labelThreshold = wx.StaticText(ResultPanel, label = " Threshold:")
        ResultPanel.textCtrlThreshold = wx.TextCtrl(ResultPanel, value = "2500", size = (-1, -1))
        #---------------------------------------------------------------------#
        self.Filters = ["U", "B", "V", "R", "I"]
        self.lambd = ["366e-9", "420e-9", "547e-9", "648e-9", "871e-9"]
        self.Ilambd = ["4.2e13", "4.5e13", "3.6e13", "2.8e13", "1.6e13"]
        self.deltalambd = ["17.5", "36.5", "45", "78.5", "118"]
        ResultPanel.labelFilterLambda = wx.StaticText(ResultPanel, label = ' Custom')
        ResultPanel.labelFilterILambda = wx.StaticText(ResultPanel, label = ' Custom')
        ResultPanel.labelFilterDeltaLambda = wx.StaticText(ResultPanel, label = ' Custom')
        # creating the sizer for holding these labels and respective comboboxes
        ResultPanel.sizerFilterLambda = wx.BoxSizer(wx.HORIZONTAL)
        ResultPanel.sizerFilterILambda = wx.BoxSizer(wx.HORIZONTAL)
        ResultPanel.sizerFilterDeltaLambda = wx.BoxSizer(wx.HORIZONTAL)
        # for finding the right value of the right combobox
        ResultPanel.comboSelection = ['' for f in xrange(0, 3)]
        
        ResultPanel.labelLambda = wx.StaticText(ResultPanel, label = u'\u03BB' + ':')
        # create a combo box 
        ResultPanel.comboboxLambda = wx.ComboBox(ResultPanel, choices=self.lambd)
        ResultPanel.comboboxLambda.Bind(wx.EVT_COMBOBOX, lambda event, 
                               arg1 = ResultPanel: self.OnLambdaCombo(event, arg1))
        
        ResultPanel.labelILambda = wx.StaticText(ResultPanel, label = u'I_'+u'\u03BB:')
        # create a combo box
        ResultPanel.comboboxILambda = wx.ComboBox(ResultPanel, choices=self.Ilambd)
        ResultPanel.comboboxILambda.Bind(wx.EVT_COMBOBOX, lambda event, 
                               arg1 = ResultPanel: self.OnILambdaCombo(event, arg1))
        
        ResultPanel.labelDeltaLambda = wx.StaticText(ResultPanel, label = '\u0394' +u'\u03BB')
        # create a combo box for \Delta\lambda
        ResultPanel.comboboxDeltaLambda = wx.ComboBox(ResultPanel, choices=self.deltalambd)
        ResultPanel.comboboxDeltaLambda.Bind(wx.EVT_COMBOBOX, lambda event, 
                               arg1 = ResultPanel: self.OnDeltaLambdaCombo(event, arg1))
        #---------------------------------------------------------------------#
        ResultPanel.sizerFilterLambda.Add(ResultPanel.labelFilterLambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerFilterLambda.Add(ResultPanel.comboboxLambda, proportion = 2, flag = wx.ALL)
        
        ResultPanel.sizerFilterILambda.Add(ResultPanel.labelFilterILambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerFilterILambda.Add(ResultPanel.comboboxILambda, proportion = 2, flag = wx.ALL)
        
        ResultPanel.sizerFilterDeltaLambda.Add(ResultPanel.labelFilterDeltaLambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerFilterDeltaLambda.Add(ResultPanel.comboboxDeltaLambda, proportion = 2, flag = wx.ALL)
        #---------------------------------------------------------------------#
        # Button for showing data
        ResultPanel.buttonShowData = wx.Button(ResultPanel, -1, "Show")
        # Adding an event handling to the showButton
        ResultPanel.buttonShowData.Bind(wx.EVT_BUTTON, lambda event, 
                               arg1 = DataPanel, arg2 = ResultPanel: self.onShowImages(event, arg1, arg2))
        #---------------------------------------------------------------------#
        ResultPanel.sizer1 = wx.BoxSizer(wx.VERTICAL)
        ResultPanel.sizer2 = wx.BoxSizer(wx.VERTICAL)
        #---------------------------------------------------------------------#
        # Sizer for holding stuff in Result Panel
        ResultPanel.mainSizer = wx.BoxSizer(wx.VERTICAL)
        #sizer for browsing information
        ResultPanel.infoSizer = wx.BoxSizer(wx.VERTICAL)
        ResultPanel.pathSizer = wx.BoxSizer(wx.HORIZONTAL)
        # Sizer for adding additional information such as
        ResultPanel.additionalInfoSizer = wx.BoxSizer(wx.HORIZONTAL)
        #sizer for holding additional data such as thresholds and various parameters 
        ResultPanel.sizerOtherData = wx.BoxSizer(wx.HORIZONTAL)
        ResultPanel.sizerThreshold = wx.BoxSizer(wx.VERTICAL)
        
        ResultPanel.sizerLambda = wx.BoxSizer(wx.VERTICAL)
        ResultPanel.sizerILambda = wx.BoxSizer(wx.VERTICAL)
        ResultPanel.sizerDeltaLambda = wx.BoxSizer(wx.VERTICAL)
        # for holding results
        ResultPanel.resultSizer = wx.GridSizer(rows = 2, cols=2, hgap=1, vgap=1)
        
        ResultPanel.infoSizer.Add(ResultPanel.sizer1, proportion = 1, flag = wx.ALL)
        ResultPanel.infoSizer.Add(ResultPanel.sizer2, proportion = 1, flag = wx.ALL)
        
        ResultPanel.sizer1.Add(ResultPanel.label1)
        ResultPanel.sizer1.Add(ResultPanel.pathSizer)
        
        ResultPanel.sizer2.Add(ResultPanel.label2)
        ResultPanel.sizer2.Add(ResultPanel.additionalInfoSizer)
        ResultPanel.sizer2.Add(ResultPanel.sizerOtherData)
        
        # Adding button and textctrl to the pathSizer
        ResultPanel.pathSizer.Add(ResultPanel.labelChoseFile, proportion = 1, flag = wx.ALL)
        ResultPanel.pathSizer.Add(ResultPanel.chosenPath, proportion = 6, flag = wx.ALL)
        ResultPanel.pathSizer.Add(ResultPanel.browseButton, proportion = 1, flag = wx.ALL)
        # Adding button and textctrl to the addtionalInfoSizer
        ResultPanel.additionalInfoSizer.Add(ResultPanel.labelChoseAdditionalFile2, proportion = 1, flag = wx.ALL)
        ResultPanel.additionalInfoSizer.Add(ResultPanel.chosenAdditionalPath, proportion = 6, flag = wx.ALL)
        ResultPanel.additionalInfoSizer.Add(ResultPanel.browseAdditionalButton, proportion = 1, flag = wx.ALL)
        # Adding the last things - threshold, parameters and show button
        ResultPanel.sizerThreshold.Add(ResultPanel.labelThreshold, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerThreshold.Add(ResultPanel.textCtrlThreshold, proportion = 1, flag = wx.ALL)
        #for \lambda varable
        ResultPanel.sizerLambda.Add(ResultPanel.labelLambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerLambda.Add(ResultPanel.sizerFilterLambda, proportion = 1, flag = wx.ALL)
        #for \I_\lambda variable
        ResultPanel.sizerILambda.Add(ResultPanel.labelILambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerILambda.Add(ResultPanel.sizerFilterILambda, proportion = 1, flag = wx.ALL)
        #ResultPanel.sizerILambda.Add(ResultPanel.textCtrlThreshold, proportion = 1, flag = wx.ALL)
        #for \Delta\lambda variable
        ResultPanel.sizerDeltaLambda.Add(ResultPanel.labelDeltaLambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerDeltaLambda.Add(ResultPanel.sizerFilterDeltaLambda, proportion = 1, flag = wx.ALL)
        #ResultPanel.sizerDeltaLambda.Add(ResultPanel.textCtrlThreshold, proportion = 1, flag = wx.ALL)
        
        #combining all sizers (for variable and show button) together
        ResultPanel.sizerOtherData.Add(ResultPanel.sizerThreshold, proportion = 2, flag = wx.ALL)
        ResultPanel.sizerOtherData.Add(ResultPanel.sizerLambda, proportion = 2, flag = wx.ALL)
        ResultPanel.sizerOtherData.Add(ResultPanel.sizerILambda, proportion = 2, flag = wx.ALL)
        ResultPanel.sizerOtherData.Add(ResultPanel.sizerDeltaLambda, proportion = 2, flag = wx.ALL)
        
        
        ResultPanel.sizerOtherData.Add(ResultPanel.buttonShowData, proportion = 1, flag = wx.ALL)
        # Adding list of canvases which represnts and will show the final data for a given tab
        ResultPanel.listFigure = [] # empty list
        ResultPanel.listAxe = []
        ResultPanel.listFigureCanvas = []
        # adding canvases to a grid
        for n in xrange(0, 4):
            #adding the figures to the list
            ResultPanel.listFigure.append(plt.figure())
            # creating the axes
            ResultPanel.listAxe.append(ResultPanel.listFigure[n].add_subplot(1, 1, 1))
            ResultPanel.listFigureCanvas.append(FigureCanvas(ResultPanel, -1, ResultPanel.listFigure[n]))
            # Adding canvases to the grid sizer
            ResultPanel.resultSizer.Add(ResultPanel.listFigureCanvas[n], 0, wx.ALL)
        # Combining sizers together
        ResultPanel.mainSizer.Add(ResultPanel.infoSizer, proportion = 1, flag = wx.ALL)
        ResultPanel.mainSizer.Add(ResultPanel.resultSizer, proportion = 3, flag = wx.ALL)
        # Setting main ResultPanel Sizer
        ResultPanel.SetSizer(ResultPanel.mainSizer)
        #---------------------------------------------------------------------#
        # identifying global variables to plot the resulting graph
        global globalIlArray

        global globalIl
        #print ("globalIlArray ", globalIlArray)
        global globalTeffArray

        global globalTtaulArray

        global globalmiukArray
        global globalmiuk1Array
        global globalPolyvalArray
        #identifying global variable to have an array of coefficients
        global globalA0Array
        global globalA1Array
        global globalA2Array
        #identifying tha same for errors in coefficients
        global globalA0ErrorArray
        global globalA1ErrorArray
        global globalA2ErrorArray
        # error in the temperatures
        global globalTeffErrorArray
        # For R sqruared test
        global globalRSquaredTestArray
        
    ''' ComboBox Methods '''
    def OnLambdaCombo(self, event, ResultPanel):
        
        data = ResultPanel.comboboxLambda.GetValue()
        ResultPanel.comboSelection[0] = float(data)
        if data == self.lambd[0]:
            ResultPanel.labelFilterLambda.SetLabel(self.Filters[0])
        if data == self.lambd[1]:
            ResultPanel.labelFilterLambda.SetLabel(self.Filters[1])
        if data == self.lambd[2]:
            ResultPanel.labelFilterLambda.SetLabel(self.Filters[2])
        if data == self.lambd[3]:
            ResultPanel.labelFilterLambda.SetLabel(self.Filters[3])
        if data == self.lambd[4]:
            ResultPanel.labelFilterLambda.SetLabel(self.Filters[4])
        
    def OnILambdaCombo(self, event, ResultPanel):
        
        data = ResultPanel.comboboxILambda.GetValue()
        ResultPanel.comboSelection[1] = float(data)
        if data == self.Ilambd[0]:
            ResultPanel.labelFilterILambda.SetLabel(self.Filters[0])
        if data == self.Ilambd[1]:
            ResultPanel.labelFilterILambda.SetLabel(self.Filters[1])
        if data == self.Ilambd[2]:
            ResultPanel.labelFilterILambda.SetLabel(self.Filters[2])
        if data == self.Ilambd[3]:
            ResultPanel.labelFilterILambda.SetLabel(self.Filters[3])
        if data == self.Ilambd[4]:
            ResultPanel.labelFilterILambda.SetLabel(self.Filters[4])
        
    def OnDeltaLambdaCombo(self, event, ResultPanel):
        
        data = ResultPanel.comboboxDeltaLambda.GetValue()
        ResultPanel.comboSelection[2] = float(data)
        if data == self.deltalambd[0]:
            ResultPanel.labelFilterDeltaLambda.SetLabel(self.Filters[0])
        if data == self.deltalambd[1]:
            ResultPanel.labelFilterDeltaLambda.SetLabel(self.Filters[1])
        if data == self.deltalambd[2]:
            ResultPanel.labelFilterDeltaLambda.SetLabel(self.Filters[2])
        if data == self.deltalambd[3]:
            ResultPanel.labelFilterDeltaLambda.SetLabel(self.Filters[3])
        if data == self.deltalambd[4]:
            ResultPanel.labelFilterDeltaLambda.SetLabel(self.Filters[4])

    ''' Defining a dialog to open a file(or multiple files)'''
    def onOpenFile1(self, event, DataPanel, ResultPanel):
        
        flat_list=[]
        # create a file dialog
        dlg = wx.FileDialog(
                self, message = "Choose a file",
                defaultDir = os.getcwd(), 
                defaultFile = '',
                wildcard = "TIFF files (*.tif)|*.tif|BMP and GIF files (*.bmp;*.gif)|*.bmp;*.gif|PNG files (*.png)|*.png",
                style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
                )
        # os.getcwd() (returns "a string representing the current working directory")
        if dlg.ShowModal() == wx.ID_OK:
            # adding the values to the paths
            DataPanel.listPaths.append(dlg.GetPaths())
            #creating one list out of list of lists
            for sublist in DataPanel.listPaths:
                for item in sublist:
                    flat_list.append(item)
                #print flat_list
                DataPanel.listPaths = flat_list
                # getting the number of elements in the list of data (self.listPaths)
                DataPanel.listLength = len(DataPanel.listPaths)
        #close the dialog
        dlg.Destroy()
        
        # putting the vaalues into the text control
        str1 = ''.join(DataPanel.listPaths)
        ResultPanel.chosenPath.SetValue(str1)
    
    ''' Defining a dialog to open a file(or multiple files)'''
    def onOpenFile2(self, event, DataPanel, ResultPanel):
        
        # This method is for chosing additional information (for instance your sky images)
        flat_list=[]
        # create a file dialog
        dlg = wx.FileDialog(
                self, message = "Choose a file",
                defaultDir = os.getcwd(), 
                defaultFile = '',
                wildcard = "TIFF files (*.tif)|*.tif|BMP and GIF files (*.bmp;*.gif)|*.bmp;*.gif|PNG files (*.png)|*.png",
                style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
                )
        # os.getcwd() (returns "a string representing the current working directory")
        if dlg.ShowModal() == wx.ID_OK:
            # adding the values to the paths
            DataPanel.listSkyPaths.append(dlg.GetPaths())
            #creating one list out of list of lists
            for sublist in DataPanel.listSkyPaths:
                for item in sublist:
                    flat_list.append(item)
                #print flat_list
                DataPanel.listSkyPaths = flat_list
                # getting the number of elements in the list of data (self.listPaths)
                DataPanel.listSkyLength = len(DataPanel.listSkyPaths)
                #print listLength
        #close the dialog
        dlg.Destroy()
        
        str2 = ''.join(DataPanel.listSkyPaths)
        ResultPanel.chosenAdditionalPath.SetValue(str2)
        
    ''' Method for showing the data '''
    def onShowImages(self, event, DataPanel, ResultPanel):
        # to retrieve the index of the chosen page 
        #(the one we are working with now)
        global globalPageIndex

        # getting the value for the threshold from text Control
        ResultPanel.threshold = float(ResultPanel.textCtrlThreshold.GetValue())
        # Invoking methods from class Calculator
        Calculator().ShowImages(DataPanel, DataPanel.listLength, DataPanel.listPaths, 
                  ResultPanel, globalPageIndex)
        
        #---------------------------------------------------------------------#
        #retrieving calculated data
        global globalTeff
        print ("globalTeff ", globalTeff)
        # error in effective temperature for each particular filter
        global globalTeffError
        
        global globalTtaul
        print ("globalTtaul ", globalTtaul)
        
        global globalmiuk
        
        global globalmiuk1
        
        global globalPolyval
        #instantiating the coefficients
        global globalA0
        global globalA1
        global globalA2
        #instantiating errors for coefficients
        global globalA0Error
        global globalA1Error
        global globalA2Error
        # result for rsquared test for each particular filter
        global globalRSquaredTest
        #---------------------------------------------------------------------#
        #filling in the global array to show its data on the resulting screen
        globalTeffArray[globalPageIndex] = globalTeff
        print ("globalTeffArray", globalTeffArray)
        globalTeffErrorArray[globalPageIndex] = globalTeffError
        print ("globalTeffErrorArray", globalTeffErrorArray)
        globalTtaulArray[globalPageIndex] = globalTtaul
        globalIlArray[globalPageIndex] = globalIl
        
        globalmiukArray[globalPageIndex] = globalmiuk
        globalmiuk1Array[globalPageIndex] = globalmiuk1
        
        globalPolyvalArray[globalPageIndex] = globalPolyval
        #putting coefficients for different tabs into one array
        globalA0Array[globalPageIndex] = globalA0
        globalA1Array[globalPageIndex] = globalA1
        globalA2Array[globalPageIndex] = globalA2
        #putting errors for coefficients into one array
        globalA0ErrorArray[globalPageIndex] = globalA0Error
        globalA1ErrorArray[globalPageIndex] = globalA1Error
        globalA2ErrorArray[globalPageIndex] = globalA2Error
        #putting the results for each r squred test into array of data
        globalRSquaredTestArray[globalPageIndex] = globalRSquaredTest

###############################################################################
''' Custom Dialog Class '''
class NameDialog(wx.Dialog):
    def __init__(self, parent, id=-1, title="Filter Name"):
        wx.Dialog.__init__(self, parent, id, title, size=(240, 165))

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.label = wx.StaticText(self, label="Enter Name:")
        self.field = wx.TextCtrl(self, value = "", size=(240, 20))
        # creating the Ok button
        self.okButton = wx.Button(self, label = "Ok", id = wx.ID_OK)
        # creating the Close button, which will close dialog
        self.closeButton = wx.Button(self, label = "Close", id = wx.ID_CLOSE)

        self.mainSizer.Add(self.label, 0, wx.ALL, 8 )
        self.mainSizer.Add(self.field, 1, wx.ALL, 8 )
        
        # Adding OkButton to sizer 
        self.buttonSizer.Add(self.okButton, 1, wx.ALL, 8 )
        # Adding CloseButton to sizer
        self.buttonSizer.Add(self.closeButton, 1, wx.ALL, 8 )

        self.mainSizer.Add(self.buttonSizer, 0, wx.ALL, 0)
        
        # Binding an event to  okButton
        self.Bind(wx.EVT_BUTTON, self.OnOK, id = wx.ID_OK)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnOK)
        # Binding an event to closeButton
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id = wx.ID_CLOSE)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnCancel)

        self.SetSizer(self.mainSizer)
        #self.Layout()
        
        self.tabResult = None
        self.nameResult = None
        
    ''' Method for okButton '''   
    def OnOK(self, event):
        self.nameResult = self.field.GetValue()
        #if self.nameResult == '':
        if self.nameResult == '':
            #self.nameResult = None
            None
        else:
            self.tabResult = self.nameResult
            self.Destroy()
        
    ''' Method for closeButton '''    
    def OnCancel(self, event):
        self.nameResult = None
        self.tabResult = ''
        self.Destroy()      
###############################################################################
class AboutFrame(wx.Frame):
    def __init__(self):
        #First retrieve the screen size of the device
        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]/2
        screenHeight = screenSize[1]/2
        
        wx.Frame.__init__(self, None, wx.ID_ANY, "About the program", size = (screenWidth, screenHeight))
        
        DescriptionPanel = ScrolledPanel(self, -1, style=wx.SIMPLE_BORDER)
        VersionPanel = ScrolledPanel(self, -1, style=wx.SIMPLE_BORDER)
        
        DescriptionPanel.SetupScrolling()
        DescriptionPanel.SetBackgroundColour('#FDDF99')

        VersionPanel.SetupScrolling()
        VersionPanel.SetBackgroundColour('#FFFFFF') 
        # creating an AUI manager
        self.auiManager = aui.AuiManager()
        # tell AuiManager to manage this frame
        self.auiManager.SetManagedWindow(self)
        # defining the notebook variable
        self.auiNotebook = aui.AuiNotebook(self)
        
        #self.auiNotebook.AddPage(self.panel, tabName)
        self.auiNotebook.AddPage(DescriptionPanel, 'Program Description')
        self.auiNotebook.AddPage(VersionPanel, 'Program Version')
        self.auiManager.Update() 
###############################################################################
''' Method for managing listCTrl '''
class EditableListCtrl(wx.ListCtrl, listmix.TextEditMixin):
    ''' TextEditMixin allows any column to be edited. '''
 
    #----------------------------------------------------------------------
    def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        """Constructor"""
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.TextEditMixin.__init__(self)
###############################################################################
''' Panel for merging data '''
class CombinedInfoPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        
        # for holding results
        self.resultSizer = wx.BoxSizer(wx.VERTICAL)
        self.graphSizer = wx.GridSizer(rows = 1, cols=2, hgap=1, vgap=1)
        
        self.tableSizer = wx.BoxSizer(wx.HORIZONTAL)
        #---------------------------------------------------------------------#
        #to retrieve the number of pages
        global globalNumberOfPages
        # retrieving the names of pages
        #global globalPageName
        global globalListPageName
        #---------------------------------------------------------------------#
        # identifying global variables to plot the resulting graph
        global globalIlArray
        
        global globalTeffArray
        global globalTeffErrorArray
        
        global globalTtaulArray
        #print ("globalTtaulArray ", globalTtaulArray)
        global globalmiukArray
        global globalmiuk1Array
        
        global globalPolyvalArray
        #Initialysing an array of coefficients
        global globalA0Array
        global globalA1Array
        global globalA2Array
        #initialising an array for errors of coefficients
        global globalA0ErrorArray
        global globalA1ErrorArray
        global globalA2ErrorArray
        # Instantiating the array of variable for R squared test
        global globalRSquaredTestArray
        #Weighted average of effective temperature
        self.TeffSum = 0
        self.TeffErrorSum = 0
        self.TeffErrorSum2 = 0
        self.TeffFinal = 0
        #self.numberOfTemperatures = 0
        for f in range(0, globalNumberOfPages):
            #self.TeffSum += globalTeffArray[f]
            self.TeffSum += globalTeffArray[f]/np.power(globalTeffErrorArray[f],2)
            self.TeffErrorSum += 1/np.power(globalTeffErrorArray[f],2)
            self.TeffErrorSum2 += np.power((globalTeffErrorArray[f]* np.power(globalTeffArray[f],-1)),2)

        self.TeffFinal = self.TeffSum/self.TeffErrorSum
        # Calculating the error associated with final effective temperature
        self.TeffFinalError = self.TeffFinal*np.sqrt(self.TeffErrorSum2)

        self.tau = np.arange(0., 2., 0.01)
        self.tau23 = 0.666

        #---------------------------------------------------------------------#
        self.listControlFinalDataTable = EditableListCtrl(self, style = wx.LC_REPORT|wx.BORDER_SUNKEN|wx.LC_HRULES|wx.LC_VRULES)
        #---------------------------------------------------------------------#
        # Adding list of canvases which represnts and will show the final data for a given tab
        self.listFigure = [] # empty list
        self.listAxe = []
        self.listFigureCanvas = []
        # variables for grid table
        self.listGrid = []
        # adding canvases to a grid
        for n in xrange(0, 2):
            #adding the figures to the list
            self.listFigure.append(plt.figure())
            # creating the axes
            self.listAxe.append(self.listFigure[n].add_subplot(1, 1, 1))
            self.listFigureCanvas.append(FigureCanvas(self, -1, self.listFigure[n]))
            # Adding canvases to the grid sizer
            self.graphSizer.Add(self.listFigureCanvas[n], 0, wx.ALL)
        #---------------------------------------------------------------------#
        self.ColShape = ['rv', 'rv', 'bs', 'c.', 'm^', 'b:', 'rv', 'rv', 'bs', 'c.', 'm^', 'b:', 'rv', 'rv', 'bs', 'c.', 'm^', 'b:']
        self.ColShape2 = ['rv', 'rv', 'r-', 'b-.', 'm--', 'b:', 'rv', 'rv', 'r-', 'b-.', 'm--', 'b:', 'rv', 'rv', 'r-', 'b-.', 'm--', 'b:']
        self.Color = ['y', 'm', 'r', 'b', 'c', 'm', 'y', 'm', 'r', 'b', 'c', 'm', 'y', 'm', 'r', 'b', 'c', 'm', 'y', 'm', 'r', 'b', 'c', 'm']
        #self.Color = ['' for col in range (globalNumberOfPages)]
        self.Color2 = ['y', 'r', 'r', 'b.', 'm', 'b', 'y', 'r', 'r', 'b.', 'm', 'b', 'y', 'r', 'r', 'b.', 'm', 'b']
        self.Shape = [u'.', u'v', u's', u'.', u'^', u'o', u'.', u'v', u's', u'.', u'^', u'o', u'.', u'v', u's', u'.', u'^', u'o', u'.', u'v', u's', u'.', u'^', u'o'] # I changed the first one from u'--' to u'd'
        self.Shape2 = [u'--', u'-', u'-.', u'-.', u'--', u':', u'--', u'-', u'-.', u'-.', u'--', u':', u'--', u'-', u'-.', u'-.', u'--', u':', u'--', u'-', u'-.', u'-.', u'--', u':']
        self.lines = [0 for col in range(globalNumberOfPages)]
        self.lines2 = [0 for col in range(globalNumberOfPages)]
        self.labels = [0 for col in range(globalNumberOfPages)]
        self.labels2 = [0 for col in range(globalNumberOfPages)]
        #---------------------------------------------------------------------#
        #List of all available colors in python#
        #for name, hex in matplotlib.colors.cnames.iteritems():
        #    COLORS = name
        #self.Color = random.choice(COLORS)
        #print (self.Color)
        #self.line = []
        self.currentDirectory = os.getcwd()
        self.saveButton = wx.Button(self, label="Save")
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSaveFile)
        #---------------------------------------------------------------------#
        #working with graphs
        self.listAxe[0].set_xlabel(r'$\mu$')
        self.listAxe[0].set_ylabel('Intensity ratio I(0,' + r'$\mu$)' + '/I(0,1)')
        #---------------------------------------------------------------------#
        self.listAxe[1].set_xlabel(r'$\tau$')
        self.listAxe[1].set_ylabel('Temperature T(' + r'$\tau$)')
        
        #drawing actual (resulting) data
        for f in xrange(0, globalNumberOfPages):
            #making the graph more interactive
            self.line2 = mlines.Line2D([], [], color=self.Color[f], marker=self.Shape[f], 
                                      markersize=5, label=globalListPageName[f])
            self.lines[f] = self.line2
            
            self.line3 = mlines.Line2D([], [], color=self.Color[f], marker=self.Shape[f], 
                                      markersize=5, label=globalListPageName[f])
            self.lines2[f] = self.line3
            #for plotting in the first canvas
            self.listAxe[0].scatter(globalmiukArray[f], globalIlArray[f], marker=self.Shape[f])
            self.listAxe[0].plot(globalmiuk1Array[f], globalPolyvalArray[f], color = self.Color[f])
            #for plotting on the second canvas
            self.listAxe[1].plot(self.tau, globalTtaulArray[f], linestyle = self.Shape2[f], color=self.Color[f])#self.ColShape2[self.f])
            self.listAxe[1].plot(self.tau23, globalTeffArray[f], marker = '*') #self.ColShape2[self.f])
            #horizontal line for temperature (different for each)
            self.listAxe[1].axhline(globalTeffArray[f], 0, self.tau23/2, color='black', linestyle='--')
        # plotting the vertical line for temperature (supposed to be the same for each filter)
        self.listAxe[1].plot([self.tau23, self.tau23], [0, np.amax(globalTeffArray)], color='black', linestyle='--')
        
        self.labels = [self.line.get_label() for self.line in self.lines]
        self.listAxe[0].legend(self.lines, self.labels)
        
        self.labels2 = [self.line.get_label() for self.line in self.lines2]
        self.listAxe[1].legend(self.lines2, self.labels2)

        self.listFigureCanvas[1].draw()
        #---------------------------------------------------------------------#
        #putting data in the table
        self.listControlFinalDataTable.InsertColumn(0, "Quantities\Filters", wx.LIST_FORMAT_CENTER, width = 160)
        for columnIndex in xrange (0, globalNumberOfPages):

            #here we are inserting the column
            self.listControlFinalDataTable.InsertColumn((columnIndex + 1), globalListPageName[columnIndex], wx.LIST_FORMAT_CENTER, width = 160)
        # inserting the names in the first column:
        self.listControlFinalDataTable.InsertStringItem(0, 'Averaged Teff')
        self.listControlFinalDataTable.InsertStringItem(0, 'Teff')
        self.listControlFinalDataTable.InsertStringItem(0, 'R^2 test')
        self.listControlFinalDataTable.InsertStringItem(0, 'a2')
        self.listControlFinalDataTable.InsertStringItem(0, 'a1')
        self.listControlFinalDataTable.InsertStringItem(0, 'a0')
        
        self.trickyIndex = 0
        for columnIndex in xrange (1, (globalNumberOfPages+1)):
            #here we are inserting the first value of the row
            self.listControlFinalDataTable.SetStringItem(0, columnIndex, "%.2f" % globalA0Array[self.trickyIndex] + '+/-' + "%.2f" % globalA0ErrorArray[self.trickyIndex])
            self.listControlFinalDataTable.SetStringItem(1, columnIndex, "%.2f" % globalA1Array[self.trickyIndex] + '+/-' + "%.2f" % globalA1ErrorArray[self.trickyIndex])
            self.listControlFinalDataTable.SetStringItem(2, columnIndex, "%.2f" % globalA2Array[self.trickyIndex] + '+/-' + "%.2f" % globalA2ErrorArray[self.trickyIndex])
            self.listControlFinalDataTable.SetStringItem(3, columnIndex, "%.2f" % globalRSquaredTestArray[self.trickyIndex])
            self.listControlFinalDataTable.SetStringItem(4, columnIndex, "%.2f" % globalTeffArray[self.trickyIndex] + '+/-' + "%.2f" % globalTeffErrorArray[self.trickyIndex])
            self.trickyIndex = self.trickyIndex + 1
        self.listControlFinalDataTable.SetStringItem(5, 1, "%.2f" % self.TeffFinal + '+/-' + "%.2f" % self.TeffFinalError)
        #putting table in the sizer
        self.tableSizer.Add(self.listControlFinalDataTable, 1, wx.ALL)
        #---------------------------------------------------------------------#
        #mixing everything up
        self.resultSizer.Add(self.graphSizer, 1, wx.ALL)
        self.resultSizer.Add(self.tableSizer, 1, wx.ALL)
        
        self.SetSizer(self.resultSizer)
        
        
    #-------------------------------------------------------------------------#
    ''' Method for saving final table as .png file '''
    def onSaveFile(self, event):
        """
        Create and show the Save FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save file as ...", 
            defaultDir=self.currentDirectory, 
            defaultFile="", 
            wildcard="PNG files (*.png)|*.png|TIFF files (*.tif)|*.tif|BMP (*.bmp)|*.bmp|GIF files (*.gif)|*.gif", 
            style=wx.FD_SAVE
            )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            print "You chose the following filename: %s" % path
        dlg.Destroy()
        
###############################################################################
''' This frame is for merged data '''
class CombinedInfoFrame(wx.Frame):
    def __init__(self):
        #First retrieve the screen size of the device
        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]/2
        screenHeight = screenSize[1]/2
        
        wx.Frame.__init__(self, None, wx.ID_ANY, "Finalized Data", size = (screenWidth, screenHeight))
        combinedInfoPanel = CombinedInfoPanel(self)
        combinedInfoPanel.SetBackgroundColour('black')
        
        # Creating a sizers to show data                                      
    
    def ShowFinalData(self, event):
        None
###############################################################################
class MainFrame(wx.Frame):
    
    """
    wx.Frame class
    """
    #----------------------------------------------------------------------
    def __init__(self):
        #First retrieve the screen size of the device
        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]/1.1
        screenHeight = screenSize[1]/1.1
        
        wx.Frame.__init__(self, None, wx.ID_ANY, "SolarLimb", size = (screenWidth, screenHeight))
        # defining a global (ok it is not that global) variablbe, which is the identifier of the tab
        self.tabIndex = 1
        
        # creating the array of filter names
        global globalListPageName
        #putting a limited value for filters,
        #because there is not that many existing :)
        globalListPageName = ["" for f in xrange(0, 100)]
        # creating an AUI manager
        self.auiManager = aui.AuiManager()
        # tell AuiManager to manage this frame
        self.auiManager.SetManagedWindow(self)
        # defining the notebook variable
        self.auiNotebook = aui.AuiNotebook(self)
        
        # Bind an event to page of notebook
        self.auiNotebook.Bind(aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.OnPageClicked)
        self.auiNotebook.Bind(aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.OnPageClose)
        # show the menu
        self.MenuBar()
        #---------------------------------------------------------------------#
        # identifying global variables to plot the resulting graph
        global globalIlArray
        globalIlArray = [0 for col in range(100)]
        
        global globalTeffArray
        globalTeffArray = [0 for col in range(100)]
        # array of errors for temperatures
        global globalTeffErrorArray
        globalTeffErrorArray = [0 for col in range(100)]
        
        global globalTtaulArray
        globalTtaulArray = [0 for col in range(100)]
        
        #array of distance parameters
        global globalmiukArray
        globalmiukArray = [0 for col in range(100)]
        global globalmiuk1Array
        globalmiuk1Array = [0 for col in range(100)]
        
        # for graphics
        global globalPolyvalArray
        globalPolyvalArray = [0 for col in range(100)]
        
        #for coeeficients a0, a1 and a2
        global globalA0Array
        globalA0Array = [0 for col in range(100)]
        global globalA1Array
        globalA1Array = [0 for col in range(100)]
        global globalA2Array
        globalA2Array = [0 for col in range(100)]
        #initialising an array for errors of coefficients
        global globalA0ErrorArray
        globalA0ErrorArray = [0 for col in range(100)]
        global globalA1ErrorArray
        globalA1ErrorArray = [0 for col in range(100)]
        global globalA2ErrorArray
        globalA2ErrorArray = [0 for col in range(100)]

        global globalRSquaredTestArray
        globalRSquaredTestArray = [0 for col in range(100)]
        
    ''' defining a menu bar method '''
    def MenuBar(self):
        
        # Setting up the menu.
        fileMenu= wx.Menu()
        addTab = fileMenu.Append(103, "Add", "Add")
        combinedData = fileMenu.Append(104, "Combine","Combine")
        aboutTheProgram = fileMenu.Append(wx.ID_ABOUT, "About","About")
        quitTheApp = fileMenu.Append(wx.ID_EXIT,"Exit","Close")
        
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu,"File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        
        # adding an eventhandling to the addTab variable: creating new tab,
        # but before that the dialog window will apper asking about the name
        # of the tab 
        self.Bind(wx.EVT_MENU, self.GetName, addTab)
        
        self.Bind(wx.EVT_MENU, self.CombinedData, combinedData)
        # adding the event handling for the about variable
        self.Bind(wx.EVT_MENU, self.AboutTheProgram, aboutTheProgram)
        # adding an eventhandling to the quitTheApp variable
        self.Bind(wx.EVT_MENU, self.OnQuit, quitTheApp)

        self.Centre()
        self.Show(True)
        
        #print self.pageIndex
        
    ''' defining method to quit the program from the menu '''
    def OnQuit(self, event):
        self.Close()
        
    ''' Method for adding new tab '''
    def AddTab(self, event, tabName):
        

        self.panel = TabPanel(self.auiNotebook)
        self.auiNotebook.AddPage(self.panel, tabName)
        self.auiManager.Update() 
        #Getting the index of newly created tab
        #---------------------------------------------------------------------#
        # Splitting the page programmatically
        #self.auiNotebook.Split(self.tab_num, wx.RIGHT)
        # tell the manager to "commit" all the changes just made
               
        #self.tab_num += 1
        
        #to account for any changes, we need to do the following procedure
        # indentifying the global variable to address it from other classes
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()

    
    ''' Method for rewriting the tab label (like a dialog window) '''
    def GetName(self, event):
        dialog = NameDialog(self)
        dialog.ShowModal()
        tabName = dialog.tabResult
        if tabName != "":
            self.AddTab(event, tabName)
        
    
    ''' Method for getting the index of the chosen tab (by click) ''' 
    def OnPageClicked(self, event):
        # to retrieve the index of the chosen page
        global globalPageIndex
        globalPageIndex = event.GetSelection()
        #print globalPageIndex
        # indentifying the global variable to address it from other classes
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()
        # to retrieve the name of the page
        global globalPageName
        globalPageName = self.auiNotebook.GetPageText(globalPageIndex)
        #inserting the name of the tab into the array for showing final data
        globalListPageName[globalPageIndex] = globalPageName
    
    ''' method which will erase the page and you can get some information about erased page '''
    def OnPageClose(self, event):
        # For accounting the deleted page (i.e. the reduced number), 
        #we change the value of the global variable
        
        # to retrieve the index oof the chosen page
        global globalPageIndex
        globalPageIndex = event.GetSelection()
        #print globalPageIndex
        # indentifying the global variable to address it from other classes
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()
        # to retrieve the name of the page
        global globalPageName
        globalPageName = self.auiNotebook.GetPageText(globalPageIndex)
            
    ''' Method for combining data from different tabs '''
    def CombinedData(self, event):        
        
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()
        # creating a condition that if we have a tab we can merge data
        if globalNumberOfPages > 0:
            combinedInfoFrame = CombinedInfoFrame()
            combinedInfoFrame.Show()
    
    ''' Method for about button from the Menu '''
    def AboutTheProgram(self, event):
        aboutFrame = AboutFrame()
        aboutFrame.Show()
        
            
#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    #app = wx.PySimpleApp()
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    #wx.lib.inspection.InspectionTool().Show()
    app.MainLoop()
    
    # To solve the problem - PyNoAppError: The wx.App object must be created first!
    del app

