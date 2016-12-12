import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import csv
import warnings
import sys
import re

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from IPython.display import display, Math, Latex

#Funkcje	
def norm(X,y):
    return (X.T*X)**-1*X.T*y

def LatexMatrix(matrix):
    ltx = r'\left[\begin{array}'
    m, n = matrix.shape
    ltx += '{' + ("r" * n) + '}'
    for i in range(m):
        ltx += r" & ".join([('%.4f' % j.item()) for j in matrix[i]]) + r" \\ "
    ltx += r'\end{array}\right]'
    return ltx

def JMx(theta,X,y):
    m = len(y)
    J = 1.0/(2.0*m)*((X*theta-y).T*(X*theta-y))
    return J.item()

def hMx(theta, X):
    return X*theta
	
	
#Wczytanie danych treningowych

reader = csv.reader(open("../train/train.tsv"), delimiter="\t")

Param_Expected = [] 
Param_Rooms = [] 
Param_SqrMeters = [] 
Param_Floor = [] 

for i in reader:
    Param_Expected.append(float(i[0]))
    Param_Rooms.append(float(i[1]))
    Param_SqrMeters.append(float(i[2])) 
    Param_Floor.append(float(i[3])) 

m, np1 = len(Param_Rooms),4

Param_Expected1 = np.matrix(Param_Expected).reshape(len(Param_Rooms),1)
Param_SqrMeters1 = np.matrix(Param_SqrMeters).reshape(len(Param_Rooms),1)
Param_Floor1 = np.matrix(Param_Floor).reshape(len(Param_Rooms),1)
Param_Rooms1 = np.matrix(Param_Rooms).reshape(len(Param_Rooms),1)


XMx = np.matrix(np.concatenate((np.ones((len(Param_Rooms),1)), Param_Rooms1,Param_SqrMeters1,Param_Floor1),axis=1)).reshape(m,4)
yMx = Param_Expected1


thetaNorm = norm(XMx, yMx)
display(Math(r'\Large \theta = ' + LatexMatrix(thetaNorm)))

#Wczytanie danych testowych
reader = csv.reader(open("../test-A/in.tsv"), delimiter="\t")

Rooms = [] 
Sqmeters = [] 
Floor = [] 

for i in reader:
    Rooms.append(float(i[0]))
    Sqmeters.append(float(i[1])) 
    Floor.append(float(i[2])) 
  
Sqmeters1 = np.matrix(Sqmeters).reshape(len(Rooms),1)
Floor1 = np.matrix(Floor).reshape(len(Rooms),1)
Rooms1 = np.matrix(Rooms).reshape(len(Rooms),1)

wXMx = np.matrix(np.concatenate((np.ones((len(Rooms),1)), Rooms1,Sqmeters1,Floor1),axis=1)).reshape(len(Rooms),4)
    
whMx = hMx(thetaNorm,wXMx)

temp = []

# Wypisanie wynikow

out = open('../test-A/out.tsv', 'w+')

for i in range(len(whMx)):
    temp.append(float(whMx[i]))
datawriter = csv.writer(out, delimiter='\n')
datawriter.writerow(temp)