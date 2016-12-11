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

reader = csv.reader(open("../train/train.tsv"), delimiter="\t")

Param_Expected = [] 
Param_Rooms = [] 
Param_SqrMeters = [] 
Param_Floor = [] 
Param_Location = []
Param_Desc = []

for i in reader:
    Param_Expected.append(float(i[0]))
    Param_Rooms.append(float(i[1]))
    Param_SqrMeters.append(float(i[2])) 
    Param_Floor.append(float(i[3])) 
    Param_Location.append(i[4])
    Param_Desc.append(i[5])
	
	
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

def addFeauter(wdesc, wlocation):
    sciezka = r'kamienica|kamienicy|Kamienica|Kamienicy'
    sciezka2 = r'Centrum|centrum'
    wTenementHouse = []
    wCentrum = []
    for i in range (len(wdesc)):
        dopasowanie = re.search(sciezka, wdesc[i])
        if dopasowanie:
            wTenementHouse.append(float(1))
        else:
            wTenementHouse.append(float(0))
        dopasowanie2 = re.search(sciezka2, wdesc[i])
        dopasowanie3 = re.search(sciezka2, wlocation[i])
        if dopasowanie2 or dopasowanie3:
            wCentrum.append(float(1))
        else:
            wCentrum.append(float(0)) 
    return wTenementHouse, wCentrum

tenemenHouse = []
tentrum = []

tenemenHouse, centrum = addFeauter(Param_Desc, Param_Location)
m, np1 = len(Param_Rooms),4

Param_Expected1 = np.matrix(Param_Expected).reshape(m,1)
Param_SqrMeters1 = np.matrix(Param_SqrMeters).reshape(m,1)
Param_Floor1 = np.matrix(Param_Floor).reshape(m,1)
Param_Rooms1 = np.matrix(Param_Rooms).reshape(m,1)
tenemenHouse_m = np.matrix(tenemenHouse).reshape(m,1)
np1 = 5


XMx = np.matrix(np.concatenate((np.ones((m,1)), Param_Rooms1,Param_SqrMeters1,Param_Floor1, tenemenHouse_m ),axis=1)).reshape(m,np1)
yMx = Param_Expected1


thetaNorm = norm(XMx, yMx)
display(Math(r'\Large \theta = ' + LatexMatrix(thetaNorm)))


reader = csv.reader(open("../test-A/in.tsv"), delimiter="\t")


wrooms = [] 
wsqrmeters = [] 
wfloor = [] 
wlocation = []
wdesc = []

for i in reader:
    wrooms.append(float(i[0]))
    wsqrmeters.append(float(i[1])) 
    wfloor.append(float(i[2])) 
    wlocation.append(i[3])
    wdesc.append(i[4])
    
wtenemenHouse = []

wtenemenHouse, wcentrum = addFeauter(wdesc,wlocation)

    
w, np1 = len(wrooms),4  
wsqrmeters_m = np.matrix(wsqrmeters).reshape(w,1)
wfloor_m = np.matrix(wfloor).reshape(w,1)
wrooms_m = np.matrix(wrooms).reshape(w,1)
wtenemenHouse_m = np.matrix(wtenemenHouse).reshape(w,1)
np1=5

wXMx = np.matrix(np.concatenate((np.ones((w,1)), wrooms_m,wsqrmeters_m,wfloor_m, wtenemenHouse_m ),axis=1)).reshape(w,np1)
    
whMx = hMx(thetaNorm,wXMx)

temp = []
out = open('../test-A/out.tsv', 'w+')

for i in range(len(whMx)):
    temp.append(float(whMx[i]))
datawriter = csv.writer(out, delimiter='\n')
datawriter.writerow(temp)