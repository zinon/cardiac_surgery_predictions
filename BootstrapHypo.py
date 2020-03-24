"""
Ref: https://codingdisciple.com/bootstrap-hypothesis-testing.html

Bootstrap Confidence Intervals and Permutation Hypothesis Testing

"""
# Options
compare_plaque = False
print_head_respampling = False
CI_boostrap = False
permutations = 10000
CI_seaborn = False
hypo_testing = True

#-------------------------------------------
import Preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

data  = pp.get_data(verbose = True)


#Boolean filtering
plaque_stroke = data[data['Stroke']==1]['PlaqueVolume']
plaque_no_stroke = data[data['Stroke']==0]['PlaqueVolume']

#remove missing values
plaque_stroke = plaque_stroke.dropna()
plaque_no_stroke = plaque_no_stroke.dropna()

#max - min
maxval = max(plaque_stroke.max(), plaque_no_stroke.max())
minval = min(plaque_stroke.min(), plaque_no_stroke.min())
xmin = 0
xmax = 7e3

if compare_plaque:
    #Figure settings
    sns.set(font_scale=1.65)
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=.5)    
    
    #Plot top histogram
    ax = fig.add_subplot(2, 1, 1)
    ax = plaque_stroke.hist(range=(minval, maxval), bins=50)
    ax.set_xlim(xmin, xmax)
    plt.xlabel('Plaque Volume (Stroke)')
    plt.ylabel('Frequency')

    #Plot bottom histogram
    ax2 = fig.add_subplot(2, 1, 2)
    ax2 = plaque_no_stroke.hist(range=(minval, maxval), bins=50)
    ax2.set_xlim(xmin, xmax)

    plt.xlabel('Plaque Volume (No Stroke)')
    plt.ylabel('Frequency')

    print('Stroke mean: {}'.format(plaque_stroke.mean()))
    print('No Stroke mean: {}'.format(plaque_no_stroke.mean()))
    
    plt.show()


if print_head_respampling:
    print "Respample stroke"
    print plaque_stroke.head()
    print resample(plaque_stroke).head()

    print "Respample no stroke"
    print plaque_no_stroke.head()
    print resample(plaque_no_stroke).head()

if CI_boostrap:
    
    #generate N permutations with replacement of the N rows
    print "Bootstrap stroke"
    stroke_bootstrap = []
    print("Before %i"%(len(stroke_bootstrap)))
    for i in range(permutations):
        np.random.seed(i)
        stroke_bootstrap.append( (resample(plaque_stroke)) )
    print("After %i"%(len(stroke_bootstrap)))

    #generate N permutations with replacement of the N rows
    print "Bootstrap no stroke"
    no_stroke_bootstrap = []
    print("Before %i"%(len(no_stroke_bootstrap)))
    for i in range(permutations):
        np.random.seed(i)
        no_stroke_bootstrap.append( (resample(plaque_no_stroke)) )
    print("After %i"%(len(no_stroke_bootstrap)))

    #calculate the mean of every permutation of the N rows
    # get an numpy array
    stroke_bootstrap_means = np.mean(stroke_bootstrap, axis=1)
    no_stroke_bootstrap_means = np.mean(no_stroke_bootstrap, axis=1)

    # 95% confidence interval: cut off 2.5% of the tail on both sides of the distribution
    # if we take many samples and the 95% confidence interval was computed for each sample,
    # 95% of the intervals would contain the true population mean
    stroke_lower_bound = np.percentile(stroke_bootstrap_means, 2.5)
    stroke_upper_bound = np.percentile(stroke_bootstrap_means, 97.5)
    print "Stroke"
    print('Lower bound: {}'.format(stroke_lower_bound))
    print('Upper bound: {}'.format(stroke_upper_bound))

    no_stroke_lower_bound = np.percentile(no_stroke_bootstrap_means, 2.5)
    no_stroke_upper_bound = np.percentile(no_stroke_bootstrap_means, 97.5)
    print "No Stroke"
    print('Lower bound: {}'.format(no_stroke_lower_bound))
    print('Upper bound: {}'.format(no_stroke_upper_bound))

    fig = plt.figure(figsize=(10,5))
    ax = sns.kdeplot(stroke_bootstrap_means, color ='Red', label='Stroke', shade=True)
    ax = sns.kdeplot(no_stroke_bootstrap_means, color='Blue', label='No Stroke', shade=True)
    plt.yticks([])
    plt.title('Bootstrap Means')
    plt.ylabel('Frequency')
    plt.xlabel('Plaque Volume')
    plt.xlim(0, 5000)
    plt.show()

    #estimate the confidence interval for the true difference
    #95% confident that the true difference between paid groups and unpaid groups is between
    differences = stroke_bootstrap_means - no_stroke_bootstrap_means

    lower_bound = np.percentile(differences, 2.5)
    upper_bound = np.percentile(differences, 97.5)

    print "Difference between the means, 95% CI:"
    print('Lower bound: {}'.format(lower_bound))
    print('Upper bound: {}'.format(upper_bound))

    #under or equal 0 assuming an increase did not happen.
    negatives = differences[differences <= 0].shape[0]
    print "%i/%i events under or equal 0 assuming an increase did not happen"%(negatives, permutations)
    
    fig = plt.figure(figsize=(10,5))
    ax = plt.hist(differences, bins=30)
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.axvline(lower_bound, color='r')
    plt.axvline(upper_bound, color='r')
    plt.title('Bootstrapped Population (Difference Between 2 Groups)')
    plt.show()

    

if CI_seaborn:
    #Seaborn plots also uses bootstrapping for calculating the confidence intervals.
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Stroke', y='PlaqueVolume', data=data, capsize=0.05, ci=95)
    x = ['No Stroke', 'Stroke']

    plt.xticks([0, 1], x)
    plt.ylabel('Plaque Volume')
    plt.xlabel('')
    plt.show() 

if hypo_testing:
    #use bootstrapping for hypothesis testing
    #Null hypothesis: plauque not different between strokes and no-strokes
    #H0:mu1-mu0=0
    #H0:mu1-mu0=0
    #The alternative hypothesis would suggest that they are different
    #Ha:mu1-mu0>0

    #Simulate a very large bootstrapped population of data and then drawing the samples from this
    #bootstrapped population. Then we'll check the likelihood of getting observed difference in means.
    #If the likelihood is less than 0.05, we'll reject the null hypothesis at 95% CL.

    combined = np.concatenate((plaque_stroke, plaque_no_stroke), axis=0)

    perms_stroke = []
    perms_no_stroke = []

    for i in range(permutations):
        np.random.seed(i)
        #define Number of samples to generate equal to the length of arrays.
        perms_stroke.append(resample(combined, n_samples = len(plaque_stroke)))
        perms_no_stroke.append(resample(combined, n_samples = len(plaque_no_stroke)))
    
    dif_bootstrap_means = (np.mean(perms_stroke, axis=1)-np.mean(perms_no_stroke, axis=1))
    print "Bootstrap difference of means", dif_bootstrap_means

    fig = plt.figure(figsize=(10,5))
    ax = plt.hist(dif_bootstrap_means, bins=30)
    plt.xlabel('Difference of plaque between strokes and no strokes')
    plt.ylabel('Frequency')
    plt.title('Bootstrapped Population (Combined data)')
    plt.show()

    # calculate the observed difference in means from our actual data.
    obs_difs = (np.mean(plaque_stroke) - np.mean(plaque_no_stroke))
    print('Observed difference in means: {}'.format(obs_difs))

    #Use bootstrapped distribution and the observed difference to determine the likelihood
    # of getting a difference in means of obs_dif

    obs = dif_bootstrap_means[dif_bootstrap_means >= obs_difs].shape[0]
    p_value = obs/float(permutations)
    print('p-value: {}'.format(p_value))
    result = "This is not a very likely occurrence. As a result, we reject the null hypothesis" if p_value < 0.05 else "This is a very likely occurence. As a result, we will accept the null hypothesis at 95%% CL"
    
    fig = plt.figure(figsize=(10,5))
    ax = plt.hist(dif_bootstrap_means, bins=30)
    plt.xlabel('Difference in plaque volume')
    plt.ylabel('Frequency')
    plt.axvline(obs_difs, color='r')
    plt.show()

    print "Out of %i bootstrap samples, only %f of these samples had a difference in means of %f or higher shown by the red line, resulting in a p-value of %f."%(permutations, obs, obs_difs, p_value)
    
    print "--> %s"%(result)
