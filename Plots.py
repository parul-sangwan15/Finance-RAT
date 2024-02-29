__author__ = 'Arkady Dushatsky'
import pylab as pl
from sklearn.metrics import roc_curve, auc

def plot_auc(y_test, probas_):

    #Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, probas_)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    #Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('FTRL-Proximal ROC')
    pl.legend(loc="lower right")
    pl.show()

def plot_learning_curve(train_loss,cv_loss, n_obs):
    # Plot learning curve
    pl.clf()
    pl.plot(n_obs,cv_loss, label='Validation learning curve')
    pl.plot(n_obs,train_loss, label='Train learning curve')
    pl.xlabel('Number of Observations')
    pl.ylabel('Logarithmic Loss')
    pl.title('Learning curves')
    pl.legend(loc="lower right")
    pl.show()
