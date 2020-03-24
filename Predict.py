import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, make_scorer, classification_report, confusion_matrix
#from sklearn.preprocessing import OneHotEncoder

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

####################################################################
## options
####################################################################
doGrid = False

param_grid = {'max_depth': range(2,3,1),
              'min_samples_split': range(2, 203, 10),
              'n_estimators': range(100, 101, 100)}

# The scorers can be either be one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

n_estimators = 100
min_samples_split = 25
max_depth = 2

####################################################################
#read in data using pandas
####################################################################
data = pd.read_csv(filepath_or_buffer = 'SteliosData.csv', sep = ",", header = "infer", engine = "python", encoding='utf-8', error_bad_lines=False)
                   #encoding = "ISO-8859-1")

####################################################################
#data pre-processing
####################################################################
                   
#data = data[np.logical_not(np.isnan(data))]               

#check data has been read in properly
print data.head()

# plaque a float
data['PlaqueVolume'] = data['PlaqueVolume'].astype(float)

#treat empty plaque volumes: replace to NaN and then drop
data['PlaqueVolume'].replace('', np.nan, inplace=True)
data.dropna(subset=['PlaqueVolume'], inplace=True)


#data types
print data.dtypes


# keep entries with plaque
data = data[ (data.PlaqueVolume > 0)  ]

#print data["PlaqueVolume"].head()


print "data shape"
print data.shape

rows = len(data.index)

print "Rows", rows

data = data[ (data.PlaqueVolume > 0)  ]

print data.describe()


####################################################################
#select features and target
####################################################################

#target
target = ["Stroke"]

features = [
    "PlaqueVolume",
#    "PartialClabing",
#    "NoCalcification",
    "CABG",
    "AKE",
    "Age",
#    "MKE_MKR",
#    "AKE_CABG",
#    "CrossClampingNoTouchAorta",
#    "AK_MK_TK_CABG"
    ]

X_input = data[features]
y_input = data[target].astype(int)

strokes = data['Stroke'].value_counts()

print "Input X"
print X_input.head()

print "Input y"
print y_input.head()

#y_input = data[target].values


####################################################################
#split
####################################################################
#For very small classes it may happen that in one of the splits we do not get any, thus leading to errors.
#Use stratify which guarantees that in each split we have a constant amount of samples from each class.

X_train, X_test, y_train, y_test = train_test_split(X_input,
                                                    y_input,
                                                    test_size=0.5,
                                                    stratify = y_input,
                                                    random_state = 0)

strokes_train = y_train['Stroke'].value_counts()
strokes_test = y_test['Stroke'].value_counts()

print("Strokes", strokes)
print("Strokes test", strokes_test)
print("Strokes train", strokes_train)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

####################################################################
#classification
####################################################################

if doGrid:
    clf = RandomForestClassifier(random_state = 0)

    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    grid_clf = GridSearchCV(estimator = clf,
                            param_grid = param_grid,
                            scoring = scoring,
                            refit='AUC',
                            return_train_score=True)
        
    grid_clf.fit(X_train, y_train.values.ravel())
    print('Best score (AUC): ', grid_clf.best_score_)
    print('Best hyperparameters (max AUC): ', grid_clf.best_params_)
    print 'Best parameters set:'
    best_parameters = grid_clf.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

else:
    clf = RandomForestClassifier(
        n_estimators = n_estimators,
        min_samples_split = min_samples_split,
        max_depth = max_depth,        
        random_state = 0)
        
    clf.fit(X_train, y_train.values.ravel())
    print "Importances"
    print list(zip(X_train[features], clf.feature_importances_))
    
this_clf = grid_clf if doGrid else clf

####################################################################
#feature importances
####################################################################

#enable line if not interested in the scores of labels that were not predicted or
# explicitly specify the labels you are interested in
y_predictions = this_clf.predict(X_test)
print "Classification report:"
print classification_report(y_test,
                            y_predictions,
                            #labels=np.unique(y_predictions),
                            labels=[0, 1]
)

####################################################################
#feature importances
####################################################################
print "Feature Importances"
feature_imp = pd.Series(this_clf.feature_importances_,
                        index=features).sort_values(ascending=False)
print feature_imp

sns.set()
plt.figure(figsize=(15,8))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.margins(0.02)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features")
#plt.legend()
#plt.show()

####################################################################
#accuracy
####################################################################

print('Stroke dataset')
print('Accuracy of RF classifier on training set: {:.2f}'.format(this_clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(this_clf.score(X_test, y_test)))

y_prediction_class = this_clf.predict(X_test)
 
print "Train Accuracy :: ", accuracy_score(y_train, this_clf.predict(X_train))
print "Test Accuracy  :: ", accuracy_score(y_test, y_prediction_class)

####################################################################
#confusion matrix
####################################################################

#see https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/confusion_matrix_pretty_print.py

cm_train = confusion_matrix(y_train, y_prediction_class)
cm_test = confusion_matrix(y_test, y_prediction_class)
print " Confusion matrix train ", cm_train
print " Confusion matrix test ", cm_test

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
res = sns.heatmap(cm_train, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)
res.invert_yaxis()
plt.yticks([0.5,1.5], ["No Stroke", "Stroke"],va='center')
plt.title('Train Confusion Matrix')
#plt.show()

####################################################################
#Precision-Recall curve
####################################################################
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle

n_classes = 2

print "N classes", n_classes

y_score = y_prediction_class

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    print "--->"
    print y_test.shape
    print y_test.head()
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))
plt.show()
####################################################################
#ROC
####################################################################

y_prediction_prob = this_clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prediction_prob)

roc_auc_val = auc(fpr, tpr)
print "ROC-AUC", roc_auc_val

#https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
plt.figure()
plt.plot([0, 1], [0, 1],  color='navy', lw = 2,  linestyle = '--')
plt.plot(fpr, tpr, color='darkorange', label='ROC-AUC = %.3f'%(roc_auc_val) )
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()

if doGrid:
    results = grid_clf.cv_results_
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    #plt.show()





#prediction = this_clf.predict( [[1400, 1, 0]] )
#print( "Prediction %f"%(prediction) )

#data[['PlaqueVolume']].plot(kind='hist',bins=[0,200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], rwidth=0.8)
#plt.show()

#data.plot(kind='scatter', x='num_children',y='num_pets',color='red')
#plt.show()


#plot plaque
#hist_settings = {'bins': 100, 'density': True, 'alpha': 0.7}
#plt.figure(figsize=[15, 7])

#plt.hist(data[column])

#

#sns.lmplot("PlaqueVolume", "Age", data=data,
           #hue="gears",
#           fit_reg=False, col='Man', col_wrap=2)
#sns.show()

    
