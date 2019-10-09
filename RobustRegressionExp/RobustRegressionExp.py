if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.linspace(0, 10, 50)

    from Input import LinearSimulation
    from Input import Normalization
    from Input import ConstantValue
    from Input import SinSimulation
    from Input import splitdata
    from robustReg import RobustLinearRegressionCPU

    from robustReg import Model
    from robustReg import Linear
    from robustReg import Tanh
    from robustReg import *
    from scipy.stats import linregress

    from sklearn.linear_model import TheilSenRegressor, RANSACRegressor

    import time

    n = Normalization(mode='n')


    str = ['One side noise', 'Both side noise']

    error1 = []
    error2 = []


    #time_start = time.time()

    #for i in range(40):
    sim = LinearSimulation(b=1.0,w=0.4,normal_scale=0.3,bin_scale=10,oneside=True,bin_rate=0.3)
    y = sim.predict(x)
    y_normal = n.fit_predict(y)

    x_ = x.reshape([-1,1])
    y_ = y_normal.reshape([-1,1])
        
    a,b,r,p,std = linregress(x,y_normal)

    nn = [FCLayer(units=1,act=Linear(),w_init=np.ones(shape=[1,1]) * a, b_init=np.ones(shape=[1,1]) * b)]
    loss = TanhLoss()
    op = GradientDecentOptimizer(loss=loss, layers=nn, learnrate=0.1)
    model = Model(nn, op, onehot=False, debug=True)

    for i in range(40):
        y1_fit = model.predict(x_).reshape(-1)

        y1_norm = n.predict(y1_fit)

        fig = plt.figure()
        plt.plot(x, y, 'r.')
        plt.plot(x, y1_norm, 'b-')
        plt.savefig('./fig{}.png'.format(i))
        plt.close(fig)
        
        model.fit(x_, y_, epochs=20, batch_size=50, minideltaloss=None)

        #reg = RANSACRegressor(random_state=0).fit(x_,y_normal)
        #y1_fit2 = reg.predict(x_)

        #a,b,r,p,std = linregress(x,y_normal)

        #y1_fit2 = n.predict(a * x + b)

        #error1.append(np.mean(np.square(sim.baseline(x) - n.predict(y1_fit))))
    #    error2.append(np.mean(np.square(sim.baseline(x) - n.predict(y1_fit2))))

    #    #if error1[-1] > 1.0:
    #    #    plt.plot(x,y1_fit,'b-')
    #    #    plt.plot(x,y_normal,'r.')
    #    #    plt.show()

    #    #plt.subplot(121 + i)
    #    #plt.title(str[i])
    #    ##plt.plot(x,sim.baseline(x),'g-.')

    #    #plt.plot(x,y1_fit2,'y-')

    #    ##plt.legend(('Prediction robust', 'Prediction least square', 'Sample points'), loc='best')
    #    #plt.legend(('Prediction robust', 'Sample points'), loc='best')

    #time_end = time.time()
    #sim = ConstantValue()

    #y = sim.predict(x)
    #y_normal = n.fit_predict(y)

    #x = np.arange(0, len(y), 1)

    #model.fit(x.reshape([-1,1]), y_normal.reshape([-1,1]), epochs=500, batch_size=10)
    #y1_fit = n.predict(model.predict(x.reshape([-1,1]))).reshape(-1)

    #a,b,r,p,std = linregress(x,y_normal)
    #y1_fit2 = n.predict(a * x + b)

    #plt.title('Regression on real sensor value')
    #plt.plot(x,y1_fit,'b-')
    #plt.plot(x,y1_fit2,'y-')
    #plt.plot(x,y_normal,'r.')
    #plt.legend(('Prediction robust', 'Prediction least square', 'Sample points'), loc='best')

    

    #plt.subplot(122)
    #plt.title('Y=2*sinX')
    #plt.plot(x,y2,'b-')
    #plt.plot(x,y2_n,'r.')
    #plt.legend(('Base line', 'Sample points'), loc='best')
    #plt.show()
            
    #print(error1)
    #print(error2)
    #print('Time use:',(time_end-time_start) * 1000,'ms')
    #print(np.mean(error1))
    #print(np.mean(error2))

    #plt.show()
