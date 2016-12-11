import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import csv
import random

from numpy import matrix
from IPython.display import display, Math, Latex

def LatexMatrix(matrix):
    ltx = r'\left[\begin{array}'
    m, n = matrix.shape
    ltx += '{' + ("r" * n) + '}'
    for i in range(m):
        ltx += r" & ".join([('%.4f' % j.item()) for j in np.array(matrix[i]).reshape(-1)]) + r" \\ "
    ltx += r'\end{array}\right]'
    return ltx

def Standaryzacja(x):
    sr = []
    od = []
    ColumnCount = len(x) - 1
    RawCount = len(x[0])
    for a in range(ColumnCount):
        sr.append(np.average(x[a], axis=None, weights=None, returned=False))
        od.append(np.std(x[a]))
        for i in range(RawCount):
            x[a][i] = (x[a][i] - sr[a]) / od[a]
    return x

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xNew = np.zeros(shape=(len(x), 2))
    for i in range(0, len(x)):
        xNew[i][0] = 1
        xNew[i][1] = x[i]
    x = xNew
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

def GDMx(fJ, fdJ, theta, X, y, alpha=0.1, eps=10 ** -3, limit=-1):
    errorCurr = fJ(theta, X, y)
    errors = [[errorCurr, theta]]
    while (len(errors) < limit if limit != -1 else True):  # limit iterations if performance is really low
        theta = theta - alpha * fdJ(theta, X, y)  # implementacja wzoru
        errorCurr, errorPrev = fJ(theta, X, y), errorCurr
        if abs(errorPrev - errorCurr) <= eps:
            break
        errors.append([errorCurr, theta])
    return theta, errors

def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    xNorm = np.zeros(shape=numPoints)
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        xNorm[i] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y, xNorm

def regdots(x, y, labelx, lebely):
    fig = pl.figure(figsize=(16 * .6, 9 * .6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.scatter(x, y, c='r', s=80, label="Dane")

    ax.set_xlabel(labelx)
    ax.set_ylabel(lebely)
    ax.margins(.05, .05)
    pl.ylim(min(y) - 1, max(y) + 1)
    pl.xlim(min(x) - 1, max(x) + 1)
    return fig

def regline(fig, fun, theta, x):
    ax = fig.axes[0]
    x0, x1 = min(x), max(x)
    X = [x0, x1]
    Y = [fun(theta, x) for x in X]
    ax.plot(X, Y, linewidth='2',
            label=(r'$y=%.2f+%.2f x$' % (theta[0], theta[1])))

def h(theta, x):
    return theta[0] + theta[1]*x

##########  przygotowanie danych
dane = csv.reader(open("../train/train.tsv"), delimiter="\t")

Param_Expected = []
Param_Rooms = []
Param_SqrMeters = []
Param_Floor = []

trainingPart = []

for i in dane:
    Param_Expected.append(float(i[0]))
    Param_Rooms.append(float(i[1]))
    Param_SqrMeters.append(float(i[2]))
    Param_Floor.append(float(i[3]))

m = 4500
np1 = 4
trainingPart.append(Param_Expected[:m])
trainingPart.append(Param_Rooms[:m])
trainingPart.append(Param_SqrMeters[:m])
trainingPart.append(Param_Floor[:m])

Xn = np.matrix(trainingPart[0:np1]).T
XnNorm = np.matrix(trainingPart[0:np1]).T
XnSource = np.matrix(trainingPart[0:np1]).T


##########  normalizacja danych

print Xn.shape
XnSum = Xn.sum(axis=0)/m
XnStd = np.std(Xn, axis=0)
## normalizacja
n = 2
for j in range(0, np1):
    numberOfCutCollumn = 0
    for i in range(0, Xn.size/np1):
        if not((XnSum.item(j) - n * XnStd.item(j) < Xn.item(i, j)) and (Xn.item(i, j) < XnSum.item(j) + n * XnStd.item(j))):
            XnNorm = np.delete(XnNorm, i-numberOfCutCollumn, 0)
            numberOfCutCollumn += 1
    Xn = XnNorm


print Xn.shape


##########  tworzenie gradientu
y = matrix(XnNorm).transpose()[0].getA()[0]
x = matrix(XnNorm).transpose()[2].getA()[0]

xNew = np.zeros(shape=(len(x), 2))
for i in range(0, len(x)):
    xNew[i][0] = 1
    xNew[i][1] = x[i]
	
tsvfile = open('../test-A/out.tsv', 'w+')
datawriter = csv.writer(tsvfile, delimiter='\n')#,quotechar="", quoting=csv.QUOTE_ALL)
datawriter.writerow(y)

m, n = np.shape(xNew)
numIterations = 100
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print "Wykres dla danych znormalizowanych"
fig = regdots(x, y, "metraz", "cena")
regline(fig, h, [theta[0], theta[1]], x)
pl.show()