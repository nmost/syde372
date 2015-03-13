# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:58:03 2015

@author: vic
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy #matrices :)
import matplotlib.pylab as plt #plotting
import compiler
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from sympy import Symbol
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr

#Class A
covA = numpy.matrix([[8,0],[0,4]])
NA = 200
uA = numpy.matrix([5,10]).T
testA = numpy.random.multivariate_normal(uA.T.A1, covA, NA)

#Class B
covB = numpy.matrix([[8,0],[0,4]])
NB = 200
uB = numpy.matrix([10,15]).T
testB = numpy.random.multivariate_normal(uB.T.A1, covB, NB)

prob_A = float(NA)/(NA + NB)
prob_B = float(NB)/(NA+NB)

#Class C
covC = numpy.matrix([[8,4],[4,40]])
NC = 100
uC = numpy.matrix([5,10]).T
testC = numpy.random.multivariate_normal(uC.T.A1, covC, NC)

#Class D
covD = numpy.matrix([[8,0],[0,8]])
ND = 200
uD = numpy.matrix([15,10]).T
testD = numpy.random.multivariate_normal(uD.T.A1, covD, ND)

#Class E
covE = numpy.matrix([[10,-5],[-5,20]])
NE = 150
uE = numpy.matrix([10,5]).T
testE = numpy.random.multivariate_normal(uE.T.A1, covE, NE)

prob_C = float(NC)/(NC + ND + NE)
prob_D = float(ND)/(NC + ND + NE)
prob_E = float(NE)/(NC + ND + NE)

#Generate MED Decision Boundary
def medEquation(u, u2):
        w = (u-u2).T
        wo = (0.5*((u2.T*u2)-(u.T*u)))
        
        #Generate symbols for our expression
        x = Symbol('x1')
        z = Symbol('x2')
        
        #Manually write out our Ax+b=0 
        xa=w[0,0]
        xb=w[0,1]
        
        #This gives us an expression for wx+wo=0
        expressiondic = solve([(xa*x+xb*z)+wo],dict='true')
        return expressiondic
     


#Generate MED Decision Metric
def MEDDecisionMetric(point, prototype):
        w = -(prototype).T
        wo = (0.5*(prototype.T*prototype))
        val = (w.A1*numpy.matrix(point).T)+wo
        return val.A1[0]


def GEDDecisionMetric(point, prototype, covariance):
    res = (point - prototype.T) * numpy.linalg.inv(covariance) * (point - prototype.T).T
    res = numpy.sqrt(res)
    return res
    
def MAPDecisionMetric(point, mean, cov, prior):
    res = multivariate_normal.pdf(point, mean, cov)
    res = res * prior
    return res
    
def euc_dist(x1, y1, x2, y2):
    return numpy.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
def knn_class(point,classes_array, k):
    x = point[0]
    y = point[1]
    k_lowest = []
    k_lowest_points = []
    min_distance = 99999
    min_class = -1
    for i, klass in enumerate(classes_array):
        k_lowest.append([])
        k_lowest_points.append([])
        for j in range(0, k):
            k_lowest[i].append(9999)
        for sample in klass:
            dist = euc_dist(x, y, sample[0], sample[1])
            for j, value in enumerate(k_lowest[i]):
                if (dist < value):
                    k_lowest[i] = k_lowest[i][0:j] + [dist] + k_lowest[i][j:]
                    k_lowest_points[i] = k_lowest_points[i][0:j] + [sample] + k_lowest_points[i][j:]
                    k_lowest[i] = k_lowest[i][0:k]
                    k_lowest_points[i] = k_lowest_points[i][0:k]
                    break
        average = [0, 0]
        for point in k_lowest_points[i]:
            average[0] += point[0]
            average[1] += point[1]
        average[0] /= k
        average[1] /= k
        #distance = euc_dist(x,y,average[0],average[1])
        dist = euc_dist(x,y,average[0],average[1])
        if (min_distance > dist):
            min_distance = dist
            min_class = i
    return min_class + 1

#Data Plots for A and B
fig1 = plt.figure(1)
plt.plot()
plt.plot(testA[:,0], testA[:,1], linestyle="None", marker=".")
plt.plot(testB[:,0], testB[:,1], linestyle="None", marker=".")
plt.title("Test Data for Classes A and B")



#Contour Plots for A and B
eigen_values_A, eigen_vectors_A = numpy.linalg.eig(covA)
angle_A = numpy.arctan(eigen_vectors_A[0,1]/eigen_vectors_A[0,0]) * 180 / numpy.pi
ellipse_A = Ellipse(uA, numpy.sqrt(eigen_values_A[0]), numpy.sqrt(eigen_values_A[1]), angle_A)
eigen_values_B, eigen_vectors_B = numpy.linalg.eig(covB)
angle_B = numpy.arctan(eigen_vectors_B[0,1]/eigen_vectors_B[0,0]) * 180 / numpy.pi
ellipse_B = Ellipse(uB, numpy.sqrt(eigen_values_B[0]), numpy.sqrt(eigen_values_B[1]), angle_B)
ax = fig1.axes[0]
ax.add_artist(ellipse_A)
ellipse_A.set_clip_box(ax.bbox)
ellipse_A.set_alpha(0.5)
ellipse_B.set_facecolor("blue")
ax.add_artist(ellipse_B)
ellipse_B.set_clip_box(ax.bbox)
ellipse_B.set_alpha(0.5)
ellipse_B.set_facecolor("green")
ax.set_xlim(0,20)
ax.set_ylim(0,20)

h = 0.2 #mesh step size
AA, BB = numpy.meshgrid(numpy.arange(0, 20, h), numpy.arange(0, 20, h))
points = numpy.c_[AA.ravel(), BB.ravel()]

testpoint = points[5000]
blag = MAPDecisionMetric(testpoint, uA.T.A1, covA, prob_A)
confGED=numpy.zeros((2, 2))
confMED=numpy.zeros((2, 2))
confMAP=numpy.zeros((2, 2))
confNN = numpy.zeros((2, 2))
confKNN = numpy.zeros((2, 2))


dist = 1000
pt = 0

def decision_classes_1(point):
    if GEDDecisionMetric(point, uA, covA) < GEDDecisionMetric(point, uB, covB):
        GEDClas=1
    else:
        GEDClas=2
    if MEDDecisionMetric(point, uA) < MEDDecisionMetric(point, uB):
        MEDClas=1
    else:
        MEDClas=2
    if MAPDecisionMetric(point, uA.T.A1, covA, prob_A) > MAPDecisionMetric(point, uB.T.A1, covB, prob_B):
        MAPClas=1
    else:
        MAPClas=2
    return (GEDClas, MEDClas, MAPClas)
    
#for point in points:
 #   GEDClass, MEDClas, MAPClas = decision_classes_1(point)
#    for A in testA:
#        tempdist = numpy.sqrt((point[0]-A[0])**2+(point[1]-A[1])**2)
#        if tempdist < dist:
#            dist = tempdist
#            pt = A
#            
#    if(dist <= 0.1):
#        if(GEDClas==1):
#            confGED[0,0]+=1
#        else:
#            confGED[0,1]+=1
#        if(MEDClas==1):
#            confMED[0,0]+=1
#        else:
#            confMED[0,1]+=1
#        if(MAPClas==1):
#            confMAP[0,0]+=1
#        else:
#            confMAP[0,1]+=1
            
#    dist = 1000
#    for B in testB:
#        tempdist = numpy.sqrt((point[0]-B[0])**2+(point[1]-B[1])**2)
#        if tempdist < dist:
#            dist = tempdist
#            pt = B            
#            
#    if(dist <= 0.1):
#        if(GEDClas==1):
#            confGED[1,0]+=1
#        else:
#            confGED[1,1]+=1
#        if(MEDClas==1):
#            confMED[1,0]+=1
#        else:
#            confMED[1,1]+=1
#        if(MAPClas==1):
#            confMAP[1,0]+=1
#        else:
#            confMAP[1,1]+=1
#    dist = 1000

#Class A
nnA = numpy.random.multivariate_normal(uA.T.A1, covA, NA)

#Class B
nnB = numpy.random.multivariate_normal(uB.T.A1, covB, NB)

#Class C
nnC = numpy.random.multivariate_normal(uC.T.A1, covC, NC)

#Class D
nnD = numpy.random.multivariate_normal(uD.T.A1, covD, ND)

#Class E
nnE = numpy.random.multivariate_normal(uE.T.A1, covE, NE)

errorGED1 = 0
errorMED1 = 0
errorMAP1 = 0
errorNN1 = 0
errorKNN1 = 0
for A in testA:
    GEDClas, MEDClas, MAPClas = decision_classes_1(A)        
    
    if(GEDClas==1):
        confGED[0,0]+=1
    else:
        confGED[0,1]+=1
        errorGED1 += 1
    if(MEDClas==1):
        confMED[0,0]+=1
    else:
        confMED[0,1]+=1
        errorMED1 += 1
    if(MAPClas==1):
        confMAP[0,0]+=1
    else:
        confMAP[0,1]+=1
        errorMAP1 += 1

    
            
for B in testB:
    GEDClas, MEDClas, MAPClas = decision_classes_1(B)
    
    if(GEDClas==1):
        confGED[1,0]+=1
        errorGED1 += 1
    else:
        confGED[1,1]+=1
        
    if(MEDClas==1):
        confMED[1,0]+=1
        errorMED1 += 1
    else:
        confMED[1,1]+=1
        
    if(MAPClas==1):
        confMAP[1,0]+=1
        errorMAP1 += 1
    else:
        confMAP[1,1]+=1
        
   
for point in nnA:
    nnclass = knn_class(point, [testA, testB], 1)
    knnclass = knn_class(point, [testA, testB], 5)
    
    if (nnclass == 1):
        confNN[0,0] += 1
    else:
        confNN[0, 1] += 1
        errorNN1 += 1
        
    
    if (knnclass == 1):
        confKNN[0,0] += 1
    else:
        confKNN[0, 1] += 1
        errorKNN1 += 1
        
for point in nnB:
    nnclass = knn_class(point, [testA, testB], 1)
    knnclass = knn_class(point, [testA, testB], 5)
    
    if (nnclass == 2):
        confNN[1,1] += 1
    else:
        confNN[1,0] += 1
        errorNN1 += 1
        
    if (knnclass == 2):
        confKNN[1,1] += 1
    else:
        confKNN[1, 0] += 1
        errorKNN1 += 1
    
print confGED
print confMED
print confMAP
print confNN
print confKNN

GEDClassification = []
MEDClassification = []
MAPClassification = []
NNClassification = []
KNNClassification = []
combined= []
combinedNN = []

for point in points:
    GEDClas, MEDClas, MAPClas = decision_classes_1(point)
    GEDClassification.append(GEDClas)
    combined.append(GEDClas)
    MEDClassification.append(MEDClas)
    combined[len(combined) - 1] += 2 + MEDClas
    MAPClassification.append(MAPClas)
    combined[len(combined) - 1] += 4 + MAPClas
    
    nnclass = knn_class(point, [testA, testB], 1)
    knnclass = knn_class(point, [testA, testB], 5)
    NNClassification.append(nnclass)
    KNNClassification.append(knnclass)
    combinedNN.append(nnclass)
    combinedNN[len(combinedNN) - 1] += 2 + knnclass

    
GEDClassification = numpy.reshape(GEDClassification, AA.shape)
MEDClassification = numpy.reshape(MEDClassification, AA.shape)
MAPClassification = numpy.reshape(MAPClassification, AA.shape)
NNClassification = numpy.reshape(NNClassification, AA.shape)
KNNClassification = numpy.reshape(KNNClassification, AA.shape)

combined = numpy.reshape(combined, AA.shape)
combinedNN = numpy.reshape(combinedNN, AA.shape)

plt.set_cmap(plt.cm.BuGn)
plt.pcolormesh(AA, BB, NNClassification)

#Data Plots for C,D and E
fig2 = plt.figure(2)
plt.plot(testC[:,0], testC[:,1], linestyle="None", marker=".")
plt.plot(testD[:,0], testD[:,1], linestyle="None", marker=".")
plt.plot(testE[:,0], testE[:,1], linestyle="None", marker=".")
plt.title("Test Data for Classes C, D and E")

#Contour plots for C, D, and E
eigen_values_C, eigen_vectors_C = numpy.linalg.eig(covC)
angle_C = numpy.arctan(eigen_vectors_C[0,1]/eigen_vectors_C[0,0]) * 180 / numpy.pi
ellipse_C = Ellipse(uC, numpy.sqrt(eigen_values_C[0]), numpy.sqrt(eigen_values_C[1]), angle_C)
eigen_values_D, eigen_vectors_D = numpy.linalg.eig(covD)
angle_D = numpy.arctan(eigen_vectors_D[0,1]/eigen_vectors_D[0,0]) * 180 / numpy.pi
ellipse_D = Ellipse(uD, numpy.sqrt(eigen_values_D[0]), numpy.sqrt(eigen_values_D[1]), angle_D)
eigen_values_E, eigen_vectors_E = numpy.linalg.eig(covE)
angle_E = numpy.arctan(eigen_vectors_E[0,1]/eigen_vectors_E[0,0]) * 180 / numpy.pi
ellipse_E = Ellipse(uE, numpy.sqrt(eigen_values_E[0]), numpy.sqrt(eigen_values_E[1]), angle_E)
ax = fig2.axes[0]
ax.add_artist(ellipse_C)
ellipse_C.set_clip_box(ax.bbox)
ellipse_C.set_alpha(0.5)
ellipse_C.set_facecolor("blue")
ax.add_artist(ellipse_D)
ellipse_D.set_clip_box(ax.bbox)
ellipse_D.set_alpha(0.5)
ellipse_D.set_facecolor("green")
ax.add_artist(ellipse_E)
ellipse_E.set_clip_box(ax.bbox)
ellipse_E.set_alpha(0.5)
ellipse_E.set_facecolor("red")
ax.set_xlim(0,20)
ax.set_ylim(0,20)

h = 0.2 #mesh step size
AA, BB = numpy.meshgrid(numpy.arange(0, 20, h), numpy.arange(0, 20, h))
points = numpy.c_[AA.ravel(), BB.ravel()]

#GED Plots for C and D, E
GEDClassification = []
MEDClassification = []
MAPClassification = []

confGED3=numpy.zeros((3, 3))
confMED3=numpy.zeros((3, 3))
confMAP3=numpy.zeros((3, 3))
confNN3=numpy.zeros((3, 3))
confKNN3=numpy.zeros((3, 3))

def decision_classes_2(point):
    if GEDDecisionMetric(point, uC, covC) < GEDDecisionMetric(point, uD, covD) and GEDDecisionMetric(point, uC, covC) < GEDDecisionMetric(point, uE, covE):
        GEDClas=2
    elif GEDDecisionMetric(point, uD, covD) < GEDDecisionMetric(point, uC, covC) and GEDDecisionMetric(point, uD, covD) < GEDDecisionMetric(point, uE, covE):
        GEDClas=3
    else:
        GEDClas=1
    if MEDDecisionMetric(point, uC) < MEDDecisionMetric(point, uD) and MEDDecisionMetric(point, uC) < MEDDecisionMetric(point, uE):
        MEDClas=2
    elif MEDDecisionMetric(point, uD) < MEDDecisionMetric(point, uC) and MEDDecisionMetric(point, uD) < MEDDecisionMetric(point, uE):
        MEDClas=3
    else:
        MEDClas=1
    if MAPDecisionMetric(point, uC.T.A1, covC, prob_C) > MAPDecisionMetric(point, uD.T.A1, covD, prob_D) and MAPDecisionMetric(point, uC.T.A1, covC, prob_C) > MAPDecisionMetric(point, uE.T.A1, covE, prob_E):
        MAPClas=2
    elif MAPDecisionMetric(point, uD.T.A1, covD, prob_D) > MAPDecisionMetric(point, uE.T.A1, covE, prob_E) and MAPDecisionMetric(point, uD.T.A1, covD, prob_D) > MAPDecisionMetric(point, uC.T.A1, covC, prob_C):
        MAPClas = 3
    else:
        MAPClas = 1
        
    return(GEDClas, MEDClas, MAPClas)

#CDE
#231


errorGED2 = 0
errorMED2 = 0
errorMAP2 = 0
errorNN2 = 0
errorKNN2 = 0

for C in testC:
    GEDClas, MEDClas, MAPClas = decision_classes_2(C)
    
    if(GEDClas==1):
        confGED3[0,2]+=1
        errorGED2 += 1
    elif(GEDClas==2):
        confGED3[0,0]+=1
    else:
        confGED3[0,1]+=1
        errorGED2 += 1
       ##MED 
    if(MEDClas==1):
        confMED3[0,2]+=1
        errorMED2 += 1
    elif(MEDClas==2):
        confMED3[0,0]+=1
    else:
        confMED3[0,1]+=1
        errorMED2 += 1

    if(MAPClas==1):
        confMAP3[0,2]+=1
        errorMAP2 += 1
    elif(MEDClas==2):
        confMAP3[0,0]+=1
    else:
        confMAP3[0,1]+=1
        errorMAP2 += 1

        
    
            
for D in testD:
        GEDClas, MEDClas, MAPClas = decision_classes_2(D)
       
        if(GEDClas==1):
            confGED3[1,2]+=1
            errorGED2 += 1
        elif(GEDClas==2):
            confGED3[1,0]+=1
            errorGED2 += 1
        else:
            confGED3[1,1]+=1
            ##MED
        if(MEDClas==1):
            confMED3[1,2]+=1
            errorMED2 += 1
        elif(MEDClas==2):
            confMED3[1,0]+=1
            errorMED2 += 1
        else:
            confMED3[1,1]+=1
        if(MAPClas==1):
            confMAP3[1,2]+=1
            errorMAP2 += 1
        elif(MAPClas==2):
            confMAP3[1,0]+=1
            errorMAP2 += 1
        else:
            confMAP3[1,1]+=1
            
        
            
for E in testE:
        GEDClas, MEDClas, MAPClas = decision_classes_2(E)
     
        if(GEDClas==1):
            confGED3[2,2]+=1
        elif(GEDClas==2):
            confGED3[2,0]+=1
            errorGED2 += 1
        else:
            confGED3[2,1]+=1
            errorGED2 += 1
            ##MED
        if(MEDClas==1):
            confMED3[2,2]+=1
        elif(MEDClas==2):
            confMED3[2,0]+=1
            errorMED2 += 1
        else:
            confMED3[2,1]+=1
            errorMED2 += 1
        if(MAPClas==1):
            confMAP3[2,2]+=1
        elif(MAPClas==2):
            confMAP3[2,0]+=1
            errorMAP2 += 1
        else:
            confMAP3[2,1]+=1
            errorMAP2 += 1
            


for point in nnC:
    nnclass = knn_class(point, [testC, testD, testE], 1)
    knnclass = knn_class(point, [testC, testD, testE], 5)
    
    if (nnclass == 1):
        confNN3[0,0] += 1
    elif (nnclass == 2):
        confNN3[0, 1] += 1
        errorNN2 += 1
    elif (nnclass == 3):
        confNN3[0, 2] += 1
        errorNN2 += 1
        
    if (knnclass == 1):
        confKNN3[0,0] += 1
    elif (knnclass == 2):
        confKNN3[0, 1] += 1
        errorKNN2 += 1
    elif (knnclass == 3):
        confKNN3[0,2] += 1
        errorKNN2 += 1
        
for point in nnD:
    nnclass = knn_class(point, [testC, testD, testE], 1)
    knnclass = knn_class(point, [testC, testD, testE], 5)
    
    if (nnclass == 2):
        confNN3[1,1] += 1
    elif (nnclass == 1):
        confNN3[1, 0] += 1
        errorNN2 += 1
    elif (nnclass == 3):
        confNN3[1, 2] += 1
        errorNN2 += 1
        
    if (knnclass == 2):
        confKNN3[1,1] += 1
    elif (knnclass == 1):
        confKNN3[1, 0] += 1
        errorKNN2 += 1
    elif (knnclass == 3):
        confKNN3[1,2] += 1
        errorKNN2 ++ 1
        
for point in nnE:
    nnclass = knn_class(point, [testC, testD, testE], 1)
    knnclass = knn_class(point, [testC, testD, testE], 5)
    
    if (nnclass == 3):
        confNN3[2,2] += 1
    elif (nnclass == 1):
        confNN3[2, 0] += 1
        errorNN2 += 1
    elif (nnclass == 2):
        confNN3[2, 1] += 1
        errorNN2 += 1
        
    if (knnclass == 3):
        confKNN3[2,2] += 1
    elif (knnclass == 1):
        confKNN3[2, 0] += 1
        errorKNN2 += 1
    elif (knnclass == 2):
        confKNN3[2, 1] += 1
        errorKNN2 += 1
        
print confGED3
print confMED3
print confMAP3
print confNN3
print confKNN3


GEDClassification = []
MEDClassification = []
MAPClassification = []
NNClassification = []
KNNClassification = []

combined = []
combinedNN = []

for point in points:
    GEDClas, MEDClas, MAPClas = decision_classes_2(point)
    GEDClassification.append(GEDClas)
    combined.append(GEDClas)
    MEDClassification.append(MEDClas + 3)
    combined[len(combined) - 1] += 3 + MEDClas
    MAPClassification.append(MAPClas + 6)
    combined[len(combined) - 1] += 6 + MAPClas
    
    nnclass = knn_class(point, [testC, testD, testE], 1)
    knnclass = knn_class(point, [testC, testD, testE], 5)
    NNClassification.append(nnclass)
    KNNClassification.append(knnclass)
    combinedNN.append(nnclass)
    combinedNN[len(combinedNN) - 1] += 3 + knnclass



GEDClassification = numpy.reshape(GEDClassification, AA.shape)
MAPClassification = numpy.reshape(MAPClassification, AA.shape)
MEDClassification = numpy.reshape(MEDClassification, AA.shape)
NNClassification = numpy.reshape(NNClassification, AA.shape)
KNNClassification = numpy.reshape(KNNClassification, AA.shape)

combined = numpy.reshape(combined, AA.shape)
combinedNN = numpy.reshape(combinedNN, AA.shape)
plt.set_cmap(plt.cm.PuBuGn)

#CHANGE THIS TO CHANGE WHAT IS BEING PLOTTED
plt.pcolormesh(AA, BB, NNClassification) 


#errrrrors

print "Errors Case 1"
print errorGED1
print float(errorGED1) / (NA + NB)
print float(errorMED1) / (NA + NB)
print float(errorMAP1) / (NA + NB)
print float(errorNN1) / (NA + NB)
print float(errorKNN1) / (NA + NB)

print "Errors Case 2"
print float(errorGED2) / (NC + ND + NE)
print float(errorMED2) / (NC + ND + NE)
print float(errorMAP2) / (NC + ND + NE)
print float(errorNN2) / (NC + ND + NE)
print float(errorKNN2) / (NC + ND + NE)


