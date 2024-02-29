from Plots import plot_auc, plot_learning_curve

from datetime import datetime
from csv import DictReader
from math import exp
from math import log
import collections
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
# parameters 
from sklearn.isotonic import IsotonicRegression

VERBOSE = True #Supplying information printing
CALIBRATION = False #Calibrating predictions by isotonic regression
NORMALIZATION = False #Normalizing features
INTERACTIONS = False  #Interactions mean using polynomials of features

#Define paths to files
train = 'train.csv'  #train file
test = 'test.csv'  #test file
submission = 'submission_FTRL.csv'  #Output submission to Kaggle testing system

#Parameters
alpha = 0.3  #Learning rate
beta = 1  #Parameter for adaptive learning rate
L1 = 0.1  #L1 regularization
L2 = 0.1  #L2 regularization
pow = 0.75 #Parameter p

mult_y = 20 #Multiplication of objects with positive y, 1 means initial datasets

#D is module for hashing if using nominal features
D = 2 ** 25

epoches = 3  #Learn training data by
holdout = 10  #Use every Nth object for validation set

#Class for calibrating predictions
class calibrate_predictions(object):
    def __init__(self, x, y, validation_set):
        self.x = x
        self.y = y
        self.validation_set = validation_set

#Main class of FTRL-algorithm
class ftrl_proximal(object):
    def __init__(self, alpha, beta, pow, L1, L2, D, interaction=False):
        self.total_count = 0
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.pow = pow
        self.D = D
        self.interaction = interaction

        '''
            model description
            n - squared sum of past gradients
            z,w - weights
        '''
        self.n = {}
        self.z = {}
        self.w = {}

        # calibrating predictions
        self.validation_set = []
        self.validation_set.append([])
        self.validation_set.append([])

        self.predictions = []
        for i in range(100):
            self.predictions.append([])

    def _indices(self, x):
        for i in x.keys():
            yield ({i: x[i]})

    def predict(self, x):
        #Parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        pow = self.pow
        #Model
        n = self.n
        z = self.z
        w = self.w

        #wTx = w*x
        wTx = 0.
        for pair in self._indices(x):
            feature = list(pair.keys())[0]

            if feature not in z:
                z[feature] = 0
                n[feature] = 0
                w[feature] = 0

            sign = -1. if z[feature] < 0 else 1.  #sgn function

            if sign * z[feature] <= L1:
                #w[i]  = 0 due to L1 regularization
                w[feature] = 0.
            else:
                w[feature] = (sign * L1 - z[feature]) / ((beta + n[feature] ** pow) / alpha + L2)

            wTx += w[feature] * float(x[feature])

        self.w = w
        #Probability prediction by bounded sigmoid function
        return 1. / (1. + exp(-max(min(wTx, 5.), -5.)))

    '''
        Updating model
        input: x - features, p - prediction of model, y - correct class
        this update modifies: self.n - squared gradient, self.z - model weights
    '''
    def update(self, x, p, y):

        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w

        #Updating z and n
        for pair in self._indices(x):
            feature = list(pair.keys())[0]
            g = (p - y) * float(x[feature])
            sigma = ((n[feature] + g * g) ** pow - n[feature] ** pow) / alpha

            z[feature] += g - sigma * w[feature]
            n[feature] += g * g


#Calculating LogLoss for one object
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    if y == 1.:
        return -log(p)
    else:
        return -log(1. - p)


#Calculating LogLoss for array of predictions
def logloss_arr(p, y):
    sum = 0.0
    for i in range(len(p)):
        p = max(min(p[i], 1. - 10e-15), 10e-15) #10e-15 value is needed to prevent calculating log(0)
        if y == 1.:
            sum += -log(p)
        else:
            sum += -log(1. - p)

    return sum

#Generator for processing data
def data_processing(path, D):
  for t, row in enumerate(DictReader(open(path))):
        #Process object id
        ID = row['']
        del row['']

        #Process y
        y = 0.0
        if 'SeriousDlqin2yrs' in row:
            try:
                y = float(row['SeriousDlqin2yrs'])
            except ValueError:
                y = 0.0
            del row['SeriousDlqin2yrs']

        row = collections.OrderedDict(sorted(row.items()))

        #Build x
        x = {}
        x['bias'] = 1

        for key in row:
            value = row[key]
            if value == 'NA':
                value = float(0)
            else:
                value = float(value)

            if value != 0:
                if NORMALIZATION:#Features normalization
                    if key=='age':
                        value/=100
                    elif key=='NumberOfTime30.59DaysPastDueNotWorse':
                        value/=100
                    elif key=='DebtRatio':
                        value/=1000000
                    elif key=='MonthlyIncome':
                        value/=10000000
                    elif key=='NumberOfOpenCreditLinesAndLoans':
                        value/=100
                    elif key=='RevolvingUtilizationOfUnsecuredLines':
                        value/=100000
                    else:
                        value/=100

            if key not in x:
                x[key] = 0
            x[key] += float(value)

            if learner.interaction == True:
                for key1 in row:
                    value1 = row[key1]

                    if value == 'NA' or value1 == 'NA':
                        value_new = float(0)
                    else:
                        if NORMALIZATION:#Features normalization
                            if key1=='age':
                                value/=100
                            elif key1=='NumberOfTime30.59DaysPastDueNotWorse':
                                value/=100
                            elif key1=='DebtRatio':
                                value/=1000000
                            elif key1=='MonthlyIncome':
                                value/=10000000
                            elif key1=='NumberOfOpenCreditLinesAndLoans':
                                value/=100
                            elif key1=='RevolvingUtilizationOfUnsecuredLines':
                                value/=100000
                            else:
                                value/=100

                        value_new = float(value) * float(value1)
                    x[key+'_'+key1] = value_new

        yield t, ID, x, y

#training
start = datetime.now()

best_loss = 10**10
predictions = list([])
real_values = list([])
train_losses = list([])
cv_losses = list([])
num_observations = list([])

for var in range(1): #This Loop was used for optimizing model parameters

    learner = ftrl_proximal(alpha, beta, pow, L1, L2, D, interaction=INTERACTIONS)

    #Training
    total = 0
    num_positive = 0
    num_negative = 0
    loss = 0.
    train_loss = 0.
    count = 0
    for e in range(epoches):
        num_positive = 0
        mean_pred = 0.0
        mean_actual = 0.0
        for t, ID, x, y in data_processing(train, D):
            multiplication = 1
            if y == 1:
                multiplication = mult_y #Multiply positive y

            for c in range(0, multiplication):
                p = learner.predict(x)
                learner.total_count += 1
                learner.validation_set[0].append(p)
                learner.validation_set[1].append(y)

                if t % holdout == 0 and t > 1:
                    #Item goes to validation
                    loss += logloss(p, y)

                    if e > 1:
                        predictions.append(p)
                        real_values.append(y)

                    if CALIBRATION:
                        learner.validation_set[0].append(p)
                        learner.validation_set[1].append(y)

                    count += 1

                else:
                    learner.update(x, p, y)
                    train_loss += logloss(p, y)

                if count > 0 and learner.total_count != count:
                    cv_losses.append(loss / count)
                    train_losses.append(train_loss / (learner.total_count - count))
                    num_observations.append(learner.total_count)

            d = p*5.0
            learner.predictions[int(d)].append([p, y])
            total += 1

            if t % 1000 == 0 and t > 1:
                if VERBOSE:
                    print(learner.w)
                    print(' %s\tencountered: %d\tcurrent cv_logloss: %f \tcurrent train_logloss: %f' % (
                        datetime.now(), t, loss / count, train_loss / (learner.total_count - count)))

    if loss/count < best_loss:
        best_loss = loss/count

    if CALIBRATION:
        #learner.predictions = sorted(zip(learner.predictions[0], learner.predictions[1]), key=itemgetter(0))
        #learner.predictions = zip(*learner.predictions)

        regression_x = [0.0]
        regression_y = [0.0]

        for i in range(len(learner.predictions)):
            sum_x = 0.0
            sum_y = 0.0
            for j in range(len(learner.predictions[i])):
                sum_x += learner.predictions[i][j][0]
                sum_y += learner.predictions[i][j][1]

            if sum_x > 0:
                regression_x.append(sum_x/max(1, len(learner.predictions[i])))
                regression_y.append(sum_y/max(1, len(learner.predictions[i])))
        regression_x.append(1.0)
        regression_y.append(1.0)

        ir = IsotonicRegression(increasing=True)

        fit = ir.fit(regression_x, regression_y)
        y_ = ir.predict(regression_x)
        plt.plot(regression_x, regression_y, 'g.', markersize=12)
        plt.plot(regression_x, y_, 'r-', markersize=5)
        plt.show()

        #learner.validation_set=zip(learner.validation_set)
        predictions_calibrated = ir.predict(learner.validation_set[0]).tolist()
        predictions_combined = learner.validation_set[0]

        for i in range(len(predictions_combined)):
            if 0.2 < predictions_combined[i] < 0.8:
                predictions_combined[i] = predictions_calibrated[i]

        loss_combined = logloss_arr(predictions_combined,learner.validation_set[1])/len(learner.validation_set[1])
        loss_calibrated = logloss_arr(predictions_calibrated,learner.validation_set[1])/len(learner.validation_set[1])

        print('calibrated logloss: %f' % loss_calibrated)
        print('combined logloss: %f' % loss_combined)

#Calculate losses and plot AUC
print('best_loss: %f\n' % (best_loss))
#print('best_pow: %f\n' % (best_pow))
plot_learning_curve(np.array(train_losses),np.array(cv_losses),np.array(num_observations))
plot_auc(np.array(real_values), np.array(predictions))

#Create Kaggle submission file
input = [0]
with open(submission, 'w') as outfile:
    outfile.write('id,Probability\n')
    for t, ID, x, y in data_processing(test, D):
        p = learner.predict(x)
        input[0] = p
        if CALIBRATION:
            p = ir.predict(input)[0]
        outfile.write('%s,%s\n' % (ID, str(p)))
