# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 14:15:16 2018

@author: Krad
"""
# to prevent error of division by 0
from __future__ import division
from __future__ import unicode_literals

import os, os.path
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
        #self.R1 = self.Cx - self.FirstPoint[0]
        #self.R2 = self.SecondPoint[0] - self.Cx
        #self.R = (self.R1+self.R2)/2
        #print(R) 
        #print(2*R/9)
        self.rout = [0 for x in range(10)]
        self.rin = [0 for x in range(10)]
        self.r = [0 for x in range(10)]
        self.delr = [0 for x in range(9)]
        self.miuk = [0 for row in range(9)]
        self.delmiu = [0 for row in range(9)]
        self.miukSum = 0
        self.miukFIN = [0 for row in range(9)]
        #self.r1 = self.FirstPoint[0]
        #self.r2 = self.FirstPoint[0] + (2*self.R/9)
        self.x = 0
        #self.y = (2*self.R/9)
        ###########   ###########  ##############  
        self.Ik = [0 for row in range(10)]
        self.Il0 = 0   #[0 for row in range(10)]
        self.Il = [0 for row in range(10)]
        self.S = [0 for row in range(9)]
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
        #FilterName = ['U', 'V', 'B', 'I', 'R']
        self.lambd = [1, 420e-9, 547e-9, 871e-9, 648e-9]
        self.Ilambd = [1, 3.6e13, 4.5e13, 1.6e13, 2.8e13]
        self.deltalambd = [17.5, 45, 16.5, 118, 78.5]
        # goes through all filters (from 1 to 4):
        
        #hPl = 6.626 * 10**(-34)
        #kBolz = 1.38 * 10**(-23)
        #clight = 3 * 10**8
        #lambd = 420e-9
        
        self.hPl = 6.626e-34
        self.kBolz = 1.38e-23
        self.clight = 3e8
        #lambd = 420e-9
        #lambd = [420e-9, 555e-9, 871e-9, 648e-9]
        #What changes per filter:
        self.Ratio1 = [0 for col in range(7)]
        self.Ratio2 = [0 for col in range(7)]
        
        self.p2Array = [0 for col in range(7)]
        self.IlArray = [0 for col in range(7)]
        self.TeffArray = [0 for col in range(7)]
        self.TtaulArray = [0 for col in range(7)]
        
        self.ColShape = ['w--', 'rv', 'bs', 'c.', 'm^', 'b:']
        self.ColShape2 = ['w--', 'rv', 'r-', 'b-.', 'm--', 'b:']
        self.Color = ['w', 'm', 'r', 'b', 'c', 'm']
        self.Color2 = ['w', 'r', 'r', 'b.', 'm', 'b']
        self.Shape = [u'D', u'v', u's', u'.', u'^', u'o'] # I changed the first one from u'--' to u'd'
        self.Shape2 = [u'--', u'-', u'--', u'-.', u'--', u':']
        self.lines = [0 for col in range(7)]
        self.labels = [0 for col in range(7)]
        
        #global Teff
        
        # to retrieve the index of the chosen page
        #global globalPageIndex
        #print globalPageIndex
        #globalPageIndex = event.GetSelection()
        #print globalPageIndex
        # indentifying the global variable to address it from other classes
        #global globalNumberOfPages
        #globalNumberOfPages = self.auiNotebook.GetPageCount()
        # to retrieve the name of the page
        #global globalPageName
        
    
    def ShowImages(self, DataPanel, listLength, listPaths, 
                   ResultPanel, globalPageIndex):
        
        print ResultPanel.comboSelection[0]
        print ResultPanel.comboSelection[1]
        print ResultPanel.comboSelection[2]
        
        #print self.pageIndex
        #print pageIndex
        # taking the value of threshold
        self.threshold = ResultPanel.threshold
        #stands for filter in Ruslan's interpretation goes through values from 2 to 5
        #important to remember it is not index or y, but anather variabl for another loop
        #self.f = globalPageIndex
        #self.f = 3
        self.f = globalPageIndex# just for testing purposes
        #---------------------------------------------------------------------#
        #retrieving the values of constants
        self.lambd[self.f] = ResultPanel.comboSelection[0]
        self.Ilambd[self.f] = ResultPanel.comboSelection[1]
        self.deltalambd[self.f] = ResultPanel.comboSelection[2]
        # variable for making matrix of data
        sizerRow = listLength * 2
        #sizerColumns = [1, 2, 3]
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
                #myfilesky = ('skyf' + str(f) + '_' + str(p) + '.tif') 
                #thisFileSky = ('R:\\AstroMundus\\AstroLab\\Sky\\' + myfilesky)
                self.Sky = misc.imread(DataPanel.listSkyPaths[self.index])
                self.Skyresh = self.Sky.reshape(1219,1619)
                #print ("Skyresh  ", self.Skyresh)

                ####################
                ##Reshaping images##
                ####################
                ''' Making the condition if image less than threshold etc. '''
                for j in range(0,1619):
                    for i in range(0,1219):
                        if (self.imresh[i,j] < self.threshold):
                            self.imresh[i,j] = 0
                        else: self.imresh[i,j] = 1

                #self.index = self.index + 1
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
                            #print("1st point is: ")
                            self.FirstPoint = [j,i]
        #                    print('1st point is:' + str(FirstPoint))
                            #print(self.FirstPoint)
                            found = True
                            break  
        
        
                found = False
                for j in range(1619-1, -1, -1):
                    if found:
                        break
                    for i in range(0,1219):
                        if (self.imresh[i,j] == 1):
                            #print("2nd point is: ")
                            self.SecondPoint = [j,i]
        #                    print('2nd point is:' + str(SecondPoint))
                            #print(SecondPoint)
                            found = True
                            break             
        
        
                found = False            
                for i in range(0,1219):
                    if found:
                        break
                    for j in range(0,1619):
                        if (self.imresh[i,j] == 1):
                            #print("3rd point is: ")
                            self.ThirdPoint = [j,i]
        #                    print('3rd point is:' + str(ThirdPoint))
                            #print(ThirdPoint)
                            found = True
                            break   
        
                found = False            
                for i in range(1219-1, -1, -1):
                    if found:
                        break
                    for j in range(0, 1619):
                        if (self.imresh[i,j] == 1):
                            #print("4th point is: ")
                            self.FourthPoint = [j,i]
        #                    print('4th point is:' + str(FourthPoint))
                            #print(FourthPoint)
                            found = True
                            break 
        
                #Now let us mathematically find an intersection of these 2 lines:
                # Center of the sun is C = (Cx, Cy)
                self.Cx[self.index] = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[0]-self.FourthPoint[0])-(self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0]))//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]-self.FourthPoint[0]))
                self.Cy[self.index] = ((self.FirstPoint[0]*self.SecondPoint[1]-self.FirstPoint[1]*self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]*self.FourthPoint[1]-self.ThirdPoint[1]*self.FourthPoint[0]))//((self.FirstPoint[0]-self.SecondPoint[0])*(self.ThirdPoint[1]-self.FourthPoint[1])-(self.FirstPoint[1]-self.SecondPoint[1])*(self.ThirdPoint[0]-self.FourthPoint[0]))
                #print (self.Cx[self.index])
                # Let's not shift the center of the next image to the center of the first one in order to find an average
                self.imreshSum = self.imreshSum + (np.roll(np.roll(self.imresh_intensity_curve_reshaped, (self.Cy[0]-self.Cy[self.index]), axis=None), (self.Cx[0]-self.Cx[self.index]), axis=0))
                self.SkySum = self.SkySum + self.Skyresh
                #print('imreshSum=', self.imreshSum)
                #And plot everything, together with the intersection point:                
                #---------------------------------------------------------#
                # invoking native method for showing images
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
                
                #print(R) 
                #print(2*R/9)
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
        #            print("Error " + str(Error[k]))
        
                    self.miuk[k] = (np.sqrt(1-(self.r[k]**2/self.R**2))).astype(float)
                    #print('r[k]/R = ' + str(r[k]/R))
                    #print('r[k]**2/R**2 = ' + str(float(r[k]**2/R**2)))
                    #print('np.sqrt(1-(r[k]**2/R**2)) = ' + str(float(np.sqrt(1-(r[k]**2/R**2)))))
        
                    self.delmiu[k] = (self.r[k]*self.delr[k])/(np.sqrt(1- (self.r[k]**2/self.R**2)))
                    self.IkN = np.asarray([[0 for col in range(1619)] for row in range(1219)])
                    #Now we determine the average intensity for each k-th region
            ##    for z in range (int(rin[k]), int(rout[k])):
                    #Intensity of all z pixels in i-th region
            ##        IkN = IkN + imresh[Cy[p],(Cx[p]+z)]
                #print('For ' + str(z) + ' steps')
                #print('Roots and Squares: '+ str((np.sqrt((x+(2*R/9))*(x+(2*R/9)) +(y+(2*R/9))*(y+(2*R/9)))+1-np.sqrt(x*x +y*y))))
                #Number of pixels in k-th region
            #    Nk = (2*math.pi*(np.sqrt((x+(2*R/9))*(x+(2*R/9)) +(y+(2*R/9))*(y+(2*R/9)))+1-np.sqrt(x*x +y*y))/z)
                #print('For ' + str(Nk) + ' Pixels')
                #Intensity associated with i-th region is the average over the intensities of all pixels
            ##    Ik = IkN / z
                    # Average intensity for k-th region
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
                #print ("Ik is", self.Ik)
                self.ErrorI = self.ErrorSum/9
                #print("Intensity error " + str(self.ErrorI))
                print ("miuk", self.miuk)
                print ("Ik ", self.Ik)
        
                #Weighted mean overl all the pictures for each filter:
                np.array(self.IlSum)[self.f][k] += np.array(self.Il)[k]
                #####################
                self.IlFIN[self.f][k] = float(self.IlSum[self.f][k])/(self.index+1) 
                #print self.IlFIN
                #####################
                ### Polynomial fit ##
                #####################
                
            
                self.imreshSumFinal = self.imreshSum/(self.index+1)
                self.SkySumFinal = self.SkySum/(self.index+1)
                self.imreshSumFinal = self.imreshSumFinal - self.SkySumFinal
                #print (self.imreshSumFinal[self.index])
                #print ("SkySumFinal", self.SkySumFinal[self.index])


                #################################################################################### 
                #Just the picture of the sun after averaging over all the pictuers for this particular filter
                #plt.figure()
                #plt.title('Resulting picture for filter No ' + str(self.f) + ', and ' + str(self.index+1) + ' pictures')
                #plt.imshow(self.imreshSumFinal)
                
                #myfile = ('Picture_for_f_' + str(self.f) + '_over_' + str(self.index+1) + '.tif') 
                #thisFileName = ('R:\\AstroMundus\\AstroLab\\oldad_average\\' + myfile)
                #plt.savefig(thisFileName)
            
                
                #myfile = ('Averaged_for_f_' + str(self.f) + '_over_' + str(self.index+1) + '_pictures.tif') 
                #thisFileName = ('R:\\AstroMundus\\AstroLab\\oldad_average\\' + myfile)
                #imsave(thisFileName, imreshSumFinal)
                #np.save('test1.txt', thisFileName)
                
                #np.savetxt('data_a.txt', self.imreshSumFinal)
                
            else:
                # Drawing the data on the canvas
                DataPanel.listShowFigure[y].set_canvas(DataPanel.listShowFigureCanvas[y])
                # Clearing the axes
                DataPanel.listShowAxe[y].clear()
                # reading an image from the same folder
                #self.imresh_intensity_curve = misc.imread(listPaths[self.index])
                # rehsping an array (changing the ) 
                #self.imresh = self.im.reshape(1219,1619)
                # reading an image from the same folder
                #self.im = misc.imread(listPaths[self.index])
                # rehsping an array (changing the ) 
                #self.imresh = self.im.reshape(1219,1619)
                
                # Intensities for every picture in cycle:
                DataPanel.listShowAxe[y].set_ylabel('Intensity (ADU)')
                DataPanel.listShowAxe[y].set_xlabel('Linear position')
                #DataPanel.listShowFigure[y].gca().invert_xaxis()
                #print self.SecondPoint[1]
                #Vertical profile
                DataPanel.listShowAxe[y].plot(self.imresh_intensity_curve[(self.ThirdPoint[1]-50):(self.FourthPoint[1]+50),self.Cx[self.index]], 'r')
                #Horizontal profile
                DataPanel.listShowAxe[y].plot(self.imresh_intensity_curve[self.Cy[self.index],(self.FirstPoint[0]-50):(self.SecondPoint[0]+50)], 'b')
                #Vertical profile
                #DataPanel.listShowAxe[y].plot(self.imresh_intensity_curve[(self.ThirdPoint[1]):(self.FourthPoint[1]),self.Cx[self.index]], 'r')
                #Horizontal profile
                #DataPanel.listShowAxe[y].plot(self.imresh_intensity_curve[self.Cy[self.index],(self.FirstPoint[0]):(self.SecondPoint[0])], 'b')
                # draw canvas
                DataPanel.listShowFigureCanvas[y].draw()
                # keeping this index he same until now to draw nice intensity plot
                self.index = self.index + 1
            
            self.number = self.number + 1
            #self.IlFIN[self.f][k] = float(self.IlSum[self.f][k])/(self.index+1) 
        
            # The values of the central intensity and the intensity ratio:
            #print('Central intensity: ' + str(self.Ik[0]))

            #adding plots to sizer
            DataPanel.gridSizerShow.Add(DataPanel.listShowFigureCanvas[y], 0, wx.ALL)
            
            
        # Setting a sizer to a panel   
        DataPanel.SetSizer(DataPanel.gridSizerShow)
        ####################################################################################
        ################################
        ##Resulting picture of the Sun##
        ################################
        #Just the picture of the sun after averaging over all the pictuers for this particular filter
        # Drawing the data on the canvas
        #ResultPanel.listFigure[0].set_canvas(ResultPanel.listFigureCanvas[0])
        # Clearing the axes
        #ResultPanel.listAxe[0].clear()
        #Showing the averaged picture of the Sun
        #ResultPanel.listAxe[0].imshow(self.imreshSumFinal)
        #ResultPanel.listFigureCanvas[0].draw()
        
        np.savetxt('data_a.txt', self.imreshSumFinal)
        #print ("imreshSumFinal: ", self.imreshSum)
        ###########################################################
        ## Looking for the center of the Sun (resulting picture) ##
        ###########################################################
        b = np.loadtxt('data_a.txt')  
        c = np.loadtxt('data_a.txt')
        
        #threshold = 10000
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
                if (b[i,j] == 1):
                    #print("1st point is: ")
                    self.FirstPoint = [j,i]
    #                print('Left (1st) point is:' + str(FirstPoint))
                    #print(FirstPoint)
                    found = True
                    break  
    
    
        found = False
        for j in range(1619-1, -1, -1):
            if found:
                break
            for i in range(0,1219):
                if (b[i,j] == 1):
                    #print("2nd point is: ")
                    self.SecondPoint = [j,i]
    #                print('Right (2nd) point is:' + str(SecondPoint))
                    #print(SecondPoint)
                    found = True
                    break             
    
    
        found = False            
        for i in range(0,1219):
            if found:
                break
            for j in range(0,1619):
                if (b[i,j] == 1):
                    #print("3rd point is: ")
                    self.ThirdPoint = [j,i]
    #                print('Top (3rd) point is:' + str(ThirdPoint))
                    #print(ThirdPoint)
                    found = True
                    break   
    
        found = False            
        for i in range(1219-1, -1, -1):
            if found:
                break
            for j in range(0, 1619):
                if (b[i,j] == 1):
                    #print("4th point is: ")
                    self.FourthPoint = [j,i]
    #                print('Bottom (4th) point is:' + str(FourthPoint))
                    #print(FourthPoint)
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
    #    print('Final Center is: (' + str(Gx) + ', ' + str(Gy) +')')    
    
        # After we have our center, we can plot an intensity curve
        # Opening of averaged pictures from where they were saved previously: 
        #b = np.loadtxt('data_a.txt')    
        ####################################################################################
        #Just the picture of the sun after averaging over all the pictuers for this particular filter:
        #plt.figure()
        #plt.title('Averaged picture of the Sun for filter ' + str(f) + '_(' + str(p+1) + ' pictures)')
        # Drawing the data on the canvas
        ResultPanel.listFigure[0].set_canvas(ResultPanel.listFigureCanvas[0])
        # Clearing the axes
        ResultPanel.listAxe[0].clear()
        ResultPanel.listAxe[0].imshow(c)
        ResultPanel.listAxe[0].plot([self.FirstPoint[0]], [self.FirstPoint[1]], marker='^', color='g')
        ResultPanel.listAxe[0].plot([self.SecondPoint[0]], [self.SecondPoint[1]], marker='o', color='r')
        ResultPanel.listAxe[0].plot([self.ThirdPoint[0]], [self.ThirdPoint[1]], marker='s', color='y')
        ResultPanel.listAxe[0].plot([self.FourthPoint[0]], [self.FourthPoint[1]], marker='^', color='y')
        ResultPanel.listAxe[0].plot((self.FirstPoint[0], self.SecondPoint[0]), (self.FirstPoint[1], self.SecondPoint[1]), color = 'g')
        ResultPanel.listAxe[0].plot((self.ThirdPoint[0], self.FourthPoint[0]), (self.ThirdPoint[1], self.FourthPoint[1]), color = 'g')
    
        ResultPanel.listAxe[0].plot([self.Gx], [self.Gy], marker='x', color='g')
        ResultPanel.listFigureCanvas[0].draw()
        #myfile = ('Final_for_f_' + str(f) + '_(' + str(p+1) + ' pictures)+Center_Point.tif') 
        #thisFileName = ('R:\\AstroMundus\\AstroLab\\oldad_average\\' + myfile)
        #plt.savefig(thisFileName)
        ##########################################
        ##The intensity plot (resulting picture)##
        ##########################################
        # Drawing the data on the canvas
        ResultPanel.listFigure[1].set_canvas(ResultPanel.listFigureCanvas[1])
        # Clearing the axes
        ResultPanel.listAxe[1].clear()
        #plt.title('Everything_final_f_' + str(f) + '_(20 pictures)')
        #plt.ylabel('Intensity (ADU)')
        #plt.xlabel('Linear position')
        #plt.gca().invert_xaxis()
        #Vertical profile
        ResultPanel.listAxe[1].plot(self.imresh_intensity_curve[
                (self.ThirdPoint[1]-50):(self.FourthPoint[1]+50),self.Gx], 'r')
        #Horizontal profile
        ResultPanel.listAxe[1].plot(self.imresh_intensity_curve[
                self.Gy,(self.FirstPoint[0]-50):(self.SecondPoint[0]+50)], 'b')    
        #myfile = ('Final_Center_to_Limb_f_' + str(f) + '_' + str(p+1) + '_picts.tif') 
        #thisFileName = ('R:\\AstroMundus\\AstroLab\\oldad_average\\' + myfile)
        #plt.savefig(thisFileName)
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
        
        self.p2 = np.polyfit(self.miuk1, self.Il1, 2)    
        self.p2Array[self.f] = self.p2
        
        global globalA0
        global globalA1
        global globalA2
        
        print(" ")
        #print("Second order fit coefficients for filter " + self.FilterName[self.f])
        print("p2 = " + str(self.p2))
        print("a0 = " + str(self.p2[2]))
        print("a1 = " + str(self.p2[1]))
        print("a2 = " + str(self.p2[0]/2))
        print(" ")  
        
        #getting the values of the coefficients for one particular filter
        globalA0 = self.p2[2]
        globalA1 = self.p2[1]
        globalA2 = self.p2[0]/2
        
        self.tau = np.arange(0., 2., 0.01)
        self.S = (self.Ilambd[self.f])*(self.p2[2] + self.tau*self.p2[1] + (self.tau**2)*self.p2[0]/2 )
    
        self.Ratio1[self.f] = (self.hPl*self.clight)/(2*(self.kBolz*self.lambd[self.f]))
        print ("Ratio1 ", self.Ratio1)
        self.Ratio2[self.f] = (2*self.hPl*(self.clight**2))/(self.lambd[self.f]**5)
        print ("Ratio2 ", self.Ratio2)
        #print("Ratio1 for filter" + str(self.f) + ' - ' + self.FilterName[self.f] + ' is ' + str(self.Ratio1[self.f]))
        #print("Ratio2 for filter" + str(self.f) + ' - ' + self.FilterName[self.f] + ' is ' + str(self.Ratio2[self.f]))
    
        self.Ttaul = self.Ratio1[self.f] / (np.log10(1 + self.Ratio2[self.f]/self.S ))
        #print ("Ttaul ", self.Ttaul)
        self.tau23 = 0.666
        self.Seff = (self.Ilambd[self.f])*(self.p2[2] + (self.tau23)*self.p2[1] + (self.tau23**2)*self.p2[0]/2 )
        #print("Seff ", self.Seff)
        self.Teff = self.Ratio1[self.f] / (np.log10(1 + self.Ratio2[self.f]/self.Seff))
        #print ("Teff ", self.Teff)
        #print("EFFECTIVE TEMPERATURE for filter " + FilterName[f] + " " + str(Teff))    
    #    ErrorTeff = 2*ErrorI + deltalambd[f] 
    #    deltaTeff = abs(Teff*(1-abs(ErrorTeff)))
    #    print("Delta Teff " + str(deltaTeff))
        
        self.IlArray[self.f] = self.Il
        self.TeffArray[self.f] = self.Teff
        self.TtaulArray[self.f] = self.Ttaul
        
        self.polyval = np.polyval(self.p2Array[self.f],self.miuk1)
        
        # assiging global variables to retrieve their values later
        global globalTeff
        globalTeff = self.Teff
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
        # identifying global variables to plot the resulting graph
        #global globalIlArray
        #globalIlArray = [0 for col in range(7)]
        #globalIlArray[self.f] = self.Il
        #print ("globalIlArray ", globalIlArray)
        
        #global globalTeffArray
        #globalTeffArray = [0 for col in range(7)]
        #globalTeffArray[self.f] = self.Teff
        
        #global globalTtaulArray
        #globalTtaulArray = [0 for col in range(7)]
        #globalTtaulArray[self.f] = self.Ttaul
        
        #Drawing the pre-last curve
        # Drawing the data on the canvas
        ResultPanel.listFigure[2].set_canvas(ResultPanel.listFigureCanvas[2])
        # Clearing the axes
        ResultPanel.listAxe[2].clear()
        ResultPanel.listAxe[2].set_xlabel(r'$\mu$')
        ResultPanel.listAxe[2].set_ylabel('Intensity ratio I(0,' + r'$\mu$)' + '/I(0,1)')
        ResultPanel.listAxe[2].scatter(self.miuk, self.IlArray[self.f], marker=self.Shape[self.f])   
        #plt.plot(miuk, Il, 'g^')    
        ResultPanel.listAxe[2].plot(self.miuk1, self.polyval)   
        #plt.legend([red_dot, (red_dot, white_cross), ], ["B", "I", "R"])
        self.line2 = mlines.Line2D([], [], color='r', marker='^', markersize=5, label="B")
        self.line3 = mlines.Line2D([], [], color='b', marker='s', markersize=5, label="I")
        self.line4 = mlines.Line2D([], [], color='g', marker='.', markersize=5, label="R")
        self.lines = [self.line2, self.line3, self.line4]
        self.labels = [self.line.get_label() for self.line in self.lines]
        ResultPanel.listAxe[2].legend(self.lines, self.labels)
        ResultPanel.listFigureCanvas[2].draw()
        
        #Drawing the last curve
        # Drawing the data on the canvas
        ResultPanel.listFigure[3].set_canvas(ResultPanel.listFigureCanvas[3])
        # Clearing the axes
        ResultPanel.listAxe[3].clear()
        ResultPanel.listAxe[3].set_xlabel(r'$\tau$')
        ResultPanel.listAxe[3].set_ylabel('Temperature T(' + r'$\tau$)')
        #ResultPanel.listAxe[3].plot(self.tau, self.TtaulArray[self.f], self.ColShape2[self.f])
        #ResultPanel.listAxe[3].plot(self.tau23, self.TeffArray[self.f], self.ColShape2[self.f])
        ResultPanel.listAxe[3].plot(self.tau, self.Ttaul, self.ColShape2[self.f])
        ResultPanel.listAxe[3].plot(self.tau23, self.Teff, self.ColShape2[self.f])

        self.line2 = mlines.Line2D([], [], color='m', linestyle=self.Shape2[2], label="B")
        self.line3 = mlines.Line2D([], [], color='r', linestyle=self.Shape2[1], label="I")
        self.line4 = mlines.Line2D([], [], color='b', linestyle=self.Shape2[3], label="R")
        self.lines = [self.line2, self.line3, self.line4]
        self.labels = [self.line.get_label() for self.line in self.lines]
        ResultPanel.listAxe[3].legend(self.lines, self.labels)
        ResultPanel.listFigureCanvas[3].draw()
        
        #Atlast, the effective temperature
        
    
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
        # to retrieve the index of the chosen page
        #global globalPageIndex
        #print globalPageIndex
        #globalPageIndex = event.GetSelection()
        #print globalPageIndex
        # indentifying the global variable to address it from other classes
        #global globalNumberOfPages
        #globalNumberOfPages = self.auiNotebook.GetPageCount()
        # to retrieve the name of the page
        #global globalPageName
        #print globalPageName
        #globalPageName = self.auiNotebook.GetPageText(globalPageIndex)
        #---------------------------------------------------------------------#
        # getting the value of the selected index to know which graph 
        #(and data in general) I am addressing
        #self.tabIndex = event.GetSelection()
        #creating a list of paths to the photos
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
        #self.threshold = 10000
        #creating an list(array) of drawing data in DataPanel
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
        ResultPanel.textCtrlThreshold = wx.TextCtrl(ResultPanel, value = "10000", size = (-1, -1))
        
        self.lambd = ["1", "420e-9", "547e-9", "871e-9", "648e-9"]
        self.Ilambd = ['1', '3.6e13', '4.5e13', '1.6e13', '2.8e13']
        self.deltalambd = ["17.5", "45", "16.5", "118", "78.5"]
        
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
        ResultPanel.sizerLambda.Add(ResultPanel.comboboxLambda, proportion = 1, flag = wx.ALL)
        #for \I_\lambda variable
        ResultPanel.sizerILambda.Add(ResultPanel.labelILambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerILambda.Add(ResultPanel.comboboxILambda, proportion = 1, flag = wx.ALL)
        #ResultPanel.sizerILambda.Add(ResultPanel.textCtrlThreshold, proportion = 1, flag = wx.ALL)
        #for \Delta\lambda variable
        ResultPanel.sizerDeltaLambda.Add(ResultPanel.labelDeltaLambda, proportion = 1, flag = wx.ALL)
        ResultPanel.sizerDeltaLambda.Add(ResultPanel.comboboxDeltaLambda, proportion = 1, flag = wx.ALL)
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
        #self.listPageName = []
        # identifying global variables to plot the resulting graph
        global globalIlArray
        #globalIlArray = [0 for col in range(7)]
        #globalIlArray[self.f] = self.Il
        global globalIl
        #print ("globalIlArray ", globalIlArray)
        global globalTeffArray
        #globalTeffArray = [0 for col in range(7)]
        #globalTeffArray[self.f] = self.Teff
        global globalTtaulArray
        #globalTtaulArray = [0 for col in range(7)]
        #globalTtaulArray[self.f] = self.Ttaul
        #globalPageIndex = event.GetSelection()
        global globalmiukArray
        global globalmiuk1Array
        global globalPolyvalArray
        #identifying global variable to have a an array of coefficients
        global globalA0Array
        global globalA1Array
        global globalA2Array
        #print globalPageIndex
        # indentifying the global variable to address it from other classes
        #global globalNumberOfPages
        #globalNumberOfPages = self.auiNotebook.GetPageCount()
        # to retrieve the name of the page
        #global globalPageName
        # for making an array of filters and other stuff
        #for f in xrange (0, globalNumberOfPages):
        #    self.listPageName[f] = globalPageName
        
    ''' ComboBox Methods '''
    def OnLambdaCombo(self, event, ResultPanel):
        
        ResultPanel.comboSelection[0] = float(ResultPanel.comboboxLambda.GetValue())
        #print ResultPanel.comboSelection[0]
        
    def OnILambdaCombo(self, event, ResultPanel):
        
        ResultPanel.comboSelection[1] = float(ResultPanel.comboboxILambda.GetValue())
        #print ResultPanel.comboSelection[1]
        
    def OnDeltaLambdaCombo(self, event, ResultPanel):
        
        ResultPanel.comboSelection[2] = float(ResultPanel.comboboxDeltaLambda.GetValue())
        #print ResultPanel.comboSelection[2]

    ''' Defining a dialog to open a file(or multiple files)'''
    def onOpenFile1(self, event, DataPanel, ResultPanel):
        # chosing a threshold
#        threshold = 10000
#        Cx = [0 for col in range(100)]
#        Cy = [0 for col in range(100)]

        # to retrieve the index of the chosen page 
        #(the one we are working with now)
        #global globalPageIndex
        #print globalPageIndex
        
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
                #print DataPanel.listLength
        #close the dialog
        dlg.Destroy()
        
        # putting the vaalues into the text control
        str1 = ''.join(DataPanel.listPaths)
        ResultPanel.chosenPath.SetValue(str1)
        
        # Invoking methods from class Calculator
        #Calculator().ShowImages(DataPanel, DataPanel.listLength, DataPanel.listPaths, 
        #          ResultPanel)
    
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
        #print globalPageIndex
        # Getting the name of the chosen tab
        #self.pageName = self.GetPageText(self.pageIndex)
        #print self.pageName
        # getting the value for the threshold from text Control
        ResultPanel.threshold = float(ResultPanel.textCtrlThreshold.GetValue())
        # Invoking methods from class Calculator
        Calculator().ShowImages(DataPanel, DataPanel.listLength, DataPanel.listPaths, 
                  ResultPanel, globalPageIndex)
        #---------------------------------------------------------------------#
        #retrieving calculated data
        global globalTeff
        print ("globalTeff ", globalTeff)
        global globalTtaul
        print ("globalTtaul ", globalTtaul)
        
        global globalmiuk
        
        global globalmiuk1
        
        global globalPolyval
        
        global globalA0
        global globalA1
        global globalA2
        #---------------------------------------------------------------------#
        #globalIlArray[globalPageIndex] = Il
        #print ("globalIlArray ", globalIlArray)
        #filling in the global array to show its data on the resulting screen
        globalTeffArray[globalPageIndex] = globalTeff
        print ("globalTeffArray", globalTeffArray)
        globalTtaulArray[globalPageIndex] = globalTtaul
        globalIlArray[globalPageIndex] = globalIl
        
        globalmiukArray[globalPageIndex] = globalmiuk
        globalmiuk1Array[globalPageIndex] = globalmiuk1
        
        globalPolyvalArray[globalPageIndex] = globalPolyval
        
        globalA0Array[globalPageIndex] = globalA0
        globalA1Array[globalPageIndex] = globalA1
        globalA2Array[globalPageIndex] = globalA2
        
        #self.IlArray[self.f] = self.Il
        #self.TeffArray[self.f] = self.Teff
        #self.TtaulArray[self.f] = self.Ttaul
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
            None
        else:
            self.tabResult = self.nameResult
        self.Destroy()
        
    ''' Method for closeButton '''    
    def OnCancel(self, event):
        self.nameResult = None
        self.Destroy()      
###############################################################################
#class DataTransfer(object):
#    def __init__(self):
#        pub.subscribe(self.RetrievingData, 'data.retrieved')
# 
#    def RetrievingData(self, numberOfPages):
#        self.numberOfPages = numberOfPages
#        print self.numberOfPages
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
        #creating an array of names of the tabs
        #self.pageNameArray = ["" for name in xrange(0, globalNumberOfPages)]
        #---------------------------------------------------------------------#
        # identifying global variables to plot the resulting graph
        global globalIlArray
        #globalIlArray = [0 for col in range(7)]
        #globalIlArray[self.f] = self.Il
        #print ("globalIlArray ", globalIlArray)
        
        global globalTeffArray
        #print ("globalTeffArray ", globalTeffArray)
        #globalTeffArray = [0 for col in range(7)]
        #globalTeffArray[self.f] = self.Teff
        
        global globalTtaulArray
        #print ("globalTtaulArray ", globalTtaulArray)
        global globalmiukArray
        global globalmiuk1Array
        
        global globalPolyvalArray
        #Initialysing an array of coefficients
        global globalA0Array
        global globalA1Array
        global globalA2Array
        
        #Weighted average of effective temperature
        self.TeffSum = 0
        self.TeffFinal = 0
        #self.numberOfTemperatures = 0
        for f in range(0, globalNumberOfPages):
            self.TeffSum += globalTeffArray[f]
            #o += 1
    
        self.TeffFinal = self.TeffSum/globalNumberOfPages
        
        #print ('TeffFinal is ', self.TeffFinal)
        #print ('A0Array', globalA0Array)
        self.tau = np.arange(0., 2., 0.01)
        self.tau23 = 0.666
        #globalTtaulArray = [0 for col in range(7)]
        #globalTtaulArray[self.f] = self.Ttaul
        #---------------------------------------------------------------------#
        self.listControlFinalDataTable = wx.ListCtrl(self, style = wx.LC_REPORT|wx.BORDER_SUNKEN|wx.LC_HRULES|wx.LC_VRULES)
                                                        #size = (400, 400), pos = (15, 15),
                                                     #style = wx.LC_REPORT|wx.BORDER_SUNKEN|wx.LC_HRULES|wx.LC_VRULES)
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
            #if n >= 2:
                #self.listGrid.append(gridlib.Grid(self, style=wx.BORDER_SUNKEN))
                #self.listGrid.CreateGrid(25,8)
    
    
                #self.tableSizer.Add(self.listGrid, 1, wx.EXPAND)
                #self.resultSizer.Add(self.tableSizer, 0, wx.ALL)
        #working with graphs
        self.listAxe[0].set_xlabel(r'$\mu$')
        self.listAxe[0].set_ylabel('Intensity ratio I(0,' + r'$\mu$)' + '/I(0,1)')
        #---------------------------------------------------------------------#
        self.listAxe[1].set_xlabel(r'$\tau$')
        self.listAxe[1].set_ylabel('Temperature T(' + r'$\tau$)')
        
        #drawing actual (resulting) data
        for f in xrange(0, globalNumberOfPages):
            #for plotting in the first canvas
            self.listAxe[0].scatter(globalmiukArray[f], globalIlArray[f], marker='*')
            self.listAxe[0].plot(globalmiuk1Array[f], globalPolyvalArray[f]) 
            #for plotting on the second canvas
            self.listAxe[1].plot(self.tau, globalTtaulArray[f], marker = '.')#self.ColShape2[self.f])
            self.listAxe[1].plot(self.tau23, globalTeffArray[f], marker = '*') #self.ColShape2[self.f])

        #self.line2 = mlines.Line2D([], [], color='m', linestyle=self.Shape2[2], label="B")
        #self.line3 = mlines.Line2D([], [], color='r', linestyle=self.Shape2[1], label="I")
        #self.line4 = mlines.Line2D([], [], color='b', linestyle=self.Shape2[3], label="R")
        #self.lines = [self.line2, self.line3, self.line4]
        #self.labels = [self.line.get_label() for self.line in self.lines]
        #self.listAxe[1].legend(self.lines, self.labels)
        self.listFigureCanvas[1].draw()
        #---------------------------------------------------------------------#
        #putting data in the table
        self.listControlFinalDataTable.InsertColumn(0, "Quantities\Filters", wx.LIST_FORMAT_CENTER, width = 160)
        for columnIndex in xrange (0, globalNumberOfPages):
        #self.listControlFinalDataTable.InsertColumn(0, u'first Column', wx.LIST_FORMAT_LEFT, width = 160)
        #self.listControlFinalDataTable.InsertColumn(1, u'second Column', wx.LIST_FORMAT_RIGHT, width = 160)
            #here we are inserting the column
            self.listControlFinalDataTable.InsertColumn((columnIndex + 1), globalListPageName[columnIndex], wx.LIST_FORMAT_CENTER, width = 160)
            #line = "Line %s" % self.index
        # inserting the names in the first column:
        self.listControlFinalDataTable.InsertStringItem(0, 'Averaged Teff')
        self.listControlFinalDataTable.InsertStringItem(0, 'Teff')
        self.listControlFinalDataTable.InsertStringItem(0, 'a2')
        self.listControlFinalDataTable.InsertStringItem(0, 'a1')
        self.listControlFinalDataTable.InsertStringItem(0, 'a0')
        
        self.trickyIndex = 0
        #for rowIndex in xrange(1,6):
        for columnIndex in xrange (1, (globalNumberOfPages+1)):
            #here we are inserting the first value of the row
            self.listControlFinalDataTable.SetStringItem(0, columnIndex, str(globalA0Array[self.trickyIndex]))
            self.listControlFinalDataTable.SetStringItem(1, columnIndex, str(globalA1Array[self.trickyIndex]))
            self.listControlFinalDataTable.SetStringItem(2, columnIndex, str(globalA2Array[self.trickyIndex]))
            self.listControlFinalDataTable.SetStringItem(3, columnIndex, str(globalTeffArray[self.trickyIndex]))
            self.trickyIndex = self.trickyIndex + 1
                #self.list_ctrl.SetStringItem(rowIndex, 2, "USA")
        self.listControlFinalDataTable.SetStringItem(4, 1, str(self.TeffFinal))
        #putting table in the sizer
        self.tableSizer.Add(self.listControlFinalDataTable, 1, wx.ALL)
        #---------------------------------------------------------------------#
        #mixing everything up
        self.resultSizer.Add(self.graphSizer, 1, wx.ALL)
        self.resultSizer.Add(self.tableSizer, 1, wx.ALL)
        #self.resultSizer.Add(self.listControlFinalDataTable, 1, wx.ALL)
        
        self.SetSizer(self.resultSizer)
        

        #print globalNumberOfPages
        # retrieving the names of pages
        #global globalPageName
        # creating the array of filter names
        #global globalListPageName
        #self.globalListPageName = []
        #for f in xrange (0, globalNumberOfPages):
        #    self.globalListPageName = globalPageName
        #print self.globalListPageName
        #self.numberOfPages = DataTransfer.numberOfPages
        #pub.subscribe(self.RetrievingData, 'data.retrieved')
        
    #def RetrievingData(self, numberOfPages):
    #    self.numberOfPages = numberOfPages
    #    #print self.numberOfPages
        
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
        #screenSize = wx.DisplaySize()
        #First retrieve the screen size of the device
        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]/1.1
        screenHeight = screenSize[1]/1.1
        
        wx.Frame.__init__(self, None, wx.ID_ANY, "SolarLimb_v0.2.7", size = (screenWidth, screenHeight))
        # defining a global variablbe, which is the identifier of the tab
        #self.tab_num = 0
        self.tabIndex = 1
        
        # creating the array of filter names
        global globalListPageName
        #putting a limited value for filters,
        #because there is not that many existing :)
        globalListPageName = ["" for f in xrange(0, 100)]
    
        #global numberOfPages
        #creating teh variable which will count the page index
        #self.pageIndex = 0
        #self.pageName = None
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
        globalIlArray = [0 for col in range(7)]
        #globalIlArray[self.f] = self.Il
        #print ("globalIlArray ", globalIlArray)
        
        global globalTeffArray
        globalTeffArray = [0 for col in range(7)]
        #globalTeffArray[self.f] = self.Teff
        
        global globalTtaulArray
        globalTtaulArray = [0 for col in range(7)]
        
        #array of distance parameters
        global globalmiukArray
        globalmiukArray = [0 for col in range(7)]
        global globalmiuk1Array
        globalmiuk1Array = [0 for col in range(7)]
        
        # for graphics
        global globalPolyvalArray
        globalPolyvalArray = [0 for col in range(7)]
        
        #for coeeficients a0, a1 and a2
        global globalA0Array
        globalA0Array = [0 for col in range(7)]
        global globalA1Array
        globalA1Array = [0 for col in range(7)]
        global globalA2Array
        globalA2Array = [0 for col in range(7)]
        #self.testList = []
        
    ''' defining a menu bar method '''
    def MenuBar(self):
        
        # Setting up the menu.
        fileMenu= wx.Menu()
        #fileMenu.Append(101, "Open", "Open")
        #fileMenu.Append(102, "Save", "Save")
        addTab = fileMenu.Append(103, "Add", "Add")
        combinedData = fileMenu.Append(104, "Combine","Combine")
        fileMenu.Append(wx.ID_ABOUT, "About","About")
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
        # adding an eventhandling to the quitTheApp variable
        self.Bind(wx.EVT_MENU, self.OnQuit, quitTheApp)

        #self.SetSize((300, 200))
        #self.SetTitle('Simple menu')
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
        #Getting the index of newly created tab
        #---------------------------------------------------------------------#
        # Splitting the page programmatically
        #self.auiNotebook.Split(self.tab_num, wx.RIGHT)
        # tell the manager to "commit" all the changes just made
        self.auiManager.Update()        
        #self.tab_num += 1
        
        #to account for any changes, we need to do the following procedure
        # indentifying the global variable to address it from other classes
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()
        # creating the array of filter names
        #global globalListPageName
        #putting a limited value for filters,
        #because there is not that many existing :)
        #globalListPageName = ["" for f in xrange(0, globalNumberOfPages)]

    
    ''' Method for rewriting the tab label (like a dialog window) '''
    def GetName(self, event):
        dialog = NameDialog(self)
        dialog.ShowModal()
        tabName = dialog.tabResult
        # calling the AddTab class which will create a new tab with the
        # specified name
        #if tabName != None:
        #if tabName != '':
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
        #print globalListPageName[globalPageIndex]
        # retrieving the names of pages
        #global globalPageName

        # giving the value of the filter to the list 
        #(we need to repeat this procedure every time because we 
        #have dynamically chnaging tabs)
        #globalListPageName[globalPageIndex] = globalPageName
        #print globalListPageName
        #for f in xrange (0, globalNumberOfPages):
        #    globalListPageName = globalPageName
        #print globalListPageName
    
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
        #print numberOfPages
        # get the total number of pages in the notebook
        #self.pageAmount = self.auiNotebook.GetPageCount()
        #getting the index of the closed tab
        #pageIndex = event.GetSelection()
        #print pageIndex
        #page = self.auiNotebook.GetPage(pageIndex)
        # Page is the window held by the tab if you just wanted the tab text
        # Getting the name of the tab
        #tabName = self.auiNotebook.GetPageText(pageIndex)
        #print tabName
        # If you named the panel and want that
        #name = page.GetName()
        #print name
        # when the page is closed, the number will be reduced
        #self.tab_num -= 1
        #print self.tab_num
            
    ''' Method for combining data from different tabs '''
    def CombinedData(self, event):        
        
        global globalNumberOfPages
        globalNumberOfPages = self.auiNotebook.GetPageCount()
        # creating a condition that if we have a tab we can merge data
        if globalNumberOfPages > 0:
            combinedInfoFrame = CombinedInfoFrame()
            combinedInfoFrame.Show()
            
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

