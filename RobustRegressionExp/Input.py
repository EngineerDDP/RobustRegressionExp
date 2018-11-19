import numpy as np


class Normalization:
    """
        Normalization class
        Providing both z-score and min-max normalization
    """

    def __init__(self, scale=1.0, mode='z'):
        """
            mode 'z' for z-score
            mode 'm' for max-min
        """

        self.Scale = scale
        self.Mode = mode

    def fit_predict(self, x):
        """
            normalize give sequence x
        """

        if self.Mode == 'z':
            self.mean = np.mean(x)
            self.std = np.std(x)
            result = (x - self.mean) / self.std
        elif self.Mode == 'm':
            self.min = np.min(x)
            self.max = np.max(x)
            result = (x - self.min) / (self.max - self.min)
        else:
            result = x

        result = result * self.Scale
        return result

    def predict(self, x):
        """
            denormalize give sequence x with fited model
        """
        
        x = x / self.Scale

        if self.Mode == 'z':
            result = x * self.std + self.mean
        elif self.Mode == 'm':
            result = x * (self.max - self.min) + self.min
        else:
            result = x

        return result


class NoiseSimulation:

    def __init__(self, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.NScale = normal_scale
        self.BScale = bin_scale
        self.BRate = bin_rate
        self.Oneside = oneside

    def predict(self, x):
        # Gaussian noise
        n1 = np.random.normal(0.0, self.NScale, size=x.shape)
        # select points
        b1 = np.random.binomial(1, self.BRate, size=x.shape)
        if self.Oneside == False:
            s1 = b1[np.where(b1 == 1)].shape
            # select side 
            s1 = self.BScale * np.random.binomial(1, 0.5,size=s1)
            s1[np.where(s1 == 0)] = -1 * self.BScale
            # write back
            b1[np.where(b1 == 1)] = s1
        else:
            b1[np.where(b1 == 1)] = self.BScale

        return x + n1 + b1


class LinearSimulation:

    def __init__(self, w=1.0, b=0.0, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.W = w
        self.B = b

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):
        """
            Create samples with noise
        """

        return self.Noise.predict(self.W * x + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return self.W * x + self.B


class SinSimulation:

    def __init__(self, a=2.0, b=0.0, w=2*np.pi, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = sin(x * 2*pi/w) + b 
        """

        self.B = b
        self.W = w
        self.A = a

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):

        return self.Noise.predict(np.sin(x * 2 * np.pi / self.W) + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return self.A * np.sin(x * 2 * np.pi / self.W) + self.B


class ConstantValue:

    def __init__(self):
        self.Val = [39.23722839
                    ,
                    38.52582932
                    ,
                    100000
                    ,
                    38.94187164
                    ,
                    38.91136169
                    ,
                    38.47372818
                    ,
                    37.79745102
                    ,
                    38.86386108
                    ,
                    38.63282013
                    ,
                    37.83452988
                    ,
                    38.05027008
                    ,
                    37.84727859
                    ,
                    38.07490158
                    ,
                    38.04582977
                    ,
                    45.07487869
                    ,
                    44.9636116
                    ,
                    47.18169022
                    ,
                    -2715.316895
                    ,
                    -1365.258057
                    ,
                    -2017.706055
                    ,
                    -1768.574951
                    ,
                    -2522.504883
                    ,
                    -2086.041992
                    ,
                    36.65307999
                    ,
                    -2774.958008
                    ,
                    -2975.412109
                    ,
                    35.616539
                    ,
                    -3237
                    ,
                    -3079.115967
                    ,
                    -2609.628906
                    ,
                    -2734.167969
                    ,
                    36.22893143
                    ,
                    36.3425293
                    ,
                    36.40225983
                    ,
                    35.939991
                    ,
                    35.5184288
                    ,
                    34.92620087
                    ,
                    34.74095917
                    ,
                    34.10062027
                    ,
                    35.45132828
                    ,
                    34.94879913
                    ,
                    36.75865173
                    ,
                    35.59135056
                    ,
                    34.81093979
                    ,
                    35.25886154
                    ,
                    34.94363022
                    ,
                    33.81396103
                    ,
                    33.04745102
                    ,
                    36.17808914
                    ,
                    36.90132141
                    ,
                    35.33583069
                    ,
                    36.14868164
                    ,
                    35.57424927
                    ,
                    35.03118896
                    ,
                    33.58666992
                    ,
                    34.20589828
                    ,
                    33.94623947
                    ,
                    33.75194168
                    ,
                    33.87612152
                    ]

    def predict(self, x):
        return np.asarray(self.Val)


def splitdata(data, input_length, output_length=1, stride=1):
    '''
        Split data into batches
        for example:
        x --> data[0:input_length]
        y --> data[input_length:input_length+output_length]
        after each sampling , iteration will move forward as "stride"
    '''
    x = []
    y = []

    for k in range(0,len(data) - input_length - output_length + 1,stride):
        y.append(data[k + input_length:k + input_length + output_length])
        x.append(data[k:k + input_length])
    return x,y