import numpy as np
import pandas as pd

class OLS():

    name = 'Estimator' # class attribute

    def __init__(self):
        print('>> Constructor invoked')
        #self.estimator = 'Ordinary Least Squares' # Instance atribute
        self.__estimator = '' # Instance atribute

        self.__X = None
        self.__y = None

        self.__beta = None
        self.__y_hat = None

        self.__std_err = None
        self.__t_stat = None

    def setname(self, name):
        self.__name = name
    
    def getname(self):
        
        return self.__name

    estimator = property(setname, getname)



    def fit(self, X, y):

        self.__X = X 
        self.__y = y

        self.__beta = self.optimal(X, y)

        self.t_statistc()



    def moments(self):

        #print(self.__X.shape)
        x_mean = np.mean(self.__X, axis=0)
        x_var = np.sum((self.__X - x_mean)**2, axis=0)

        return x_mean, x_var


    def optimal(self, X, y, intercept=True): # Class method
        
        if intercept:
            print('Intercept added')
            X = np.insert(X, 0, 1, axis=1)
        
        beta = np.linalg.inv(X.T@X)@X.T@y

        return beta

    def predict(self, X):

        X = np.insert(X, 0, 1, axis=1)
        y_hat = X @ self.__beta
        self.__y_hat = y_hat

        return y_hat

    def rss(self, y, y_hat):

        err = y-y_hat
        rss = np.sum(err**2)

        return rss


    def standard_error(self):

        y_hat = self.predict(self.__X)
        
        n = len(self.__y)
        d = len(self.__beta) + 1
        rss = self.rss(self.__y, y_hat)
        sigma = np.sqrt(rss/(n-d)) # 2 degrees of freedom

        x_mean, x_var = self.moments()

        #TODO: beta zero standard error
        se_1 = 1

        se_x = sigma/x_var
        #print(se_x)
        se = np.insert(se_x, 0, se_1)
        #print(se)
        self.__std_err = se
        return se

    def t_statistc(self):

        se = self.standard_error()
        t_stat = self.__beta / se
        
        self.__t_stat = t_stat

        #return p_val


    def summary(self):

        summary_df = pd.DataFrame({'coef': self.__beta,
                    'std_err':self.__std_err,
                    't_stat':self.__t_stat,
                    })

        print(summary_df.round(4))

