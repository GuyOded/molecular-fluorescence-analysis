import pandas
import numpy as np
from scipy.odr import *
import matplotlib.pyplot as plt

file_paths = ['Fe data/A.xlsx','R6G data/B.xlsx','RHB data/C.xlsx']
name_l = ['A','B','C']

def p_read (path, name):
    data = pandas.read_excel(path, sheet_name=name)
    return data

def  lin_func (p, x):
    a, b = p
    return (a*x)+b

lin_model = Model(lin_func)

for i in [0,1,2]:
    set=[]
    for i2 in [1,2,3,4,5,6,7,8,9,10]:
        d = p_read(file_paths[i], name_l[i] + str(i2))

        x=d.loc[:,'x']
        y=d.loc[:,'Avl']
        yn=d.loc[:,'Av']
        dy=1/255

        # init values and fixing None for y
        xdiff = (x[1]-x[0])/np.sqrt(12)

        x_err= np.array([xdiff for i in x])
        y_err= np.array([abs(dy/i) for i in yn])

        data = RealData(x,y, sx=x_err, sy=y_err)

        odr = ODR(data, lin_model, beta0=[0.,1.])

        out = odr.run()

        print('results for run ' + str(name_l[i]) + str(i2))
        out.pprint()
        xlen = len(x)
        x_fit=np.linspace(x[0],x[xlen-1],1000)
        y_fit = lin_func(out.beta, x_fit)

        set.append({
            'x':x,
            'y':y,
            'x_err':x_err,
            'y_err':y_err,
            'x_fit':x_fit,
            'y_fit':y_fit
        })

    plt.title('I(l) for different concentrations')
    for iplt in [1,2,3,4,5,6,7,8,9]:
        plt.subplot(3,3,iplt)
        plt.errorbar(set[iplt]['x'],set[iplt]['y'],xerr=set[iplt]['x_err'],yerr=set[iplt]['y_err'],linestyle='None',marker='.')
        plt.plot(set[iplt]['x_fit'],set[iplt]['y_fit'])
    plt.show()

    plt.errorbar(set[0]['x'], set[0]['y'], xerr=set[0]['x_err'], yerr=set[0]['y_err'], linestyle='None',marker='.')
    plt.plot(set[0]['x_fit'], set[0]['y_fit'])
    plt.title('Intensity ratio as a function of depth I(l) for c=0.1')
    plt.xlabel('position [cm]')
    plt.ylabel('Intensity ratio [N/a]')
    plt.show()

    plt.errorbar(set[9]['x'], set[9]['y'], xerr=set[9]['x_err'], yerr=set[9]['y_err'], linestyle='None',marker='.')
    plt.plot(set[9]['x_fit'], set[9]['y_fit'])
    plt.title('Intensity ratio as a function of depth I(l) for c=10^-4')
    plt.xlabel('position [cm]')
    plt.ylabel('Intensity ratio [N/a]')
    plt.show()
