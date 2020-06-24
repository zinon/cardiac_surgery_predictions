"""
Two sampled T-test :-The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test.

H0: there is association between plaque volumes of strokes and no strokes



Two-sample Z test- In two sample z-test , similar to t-test here we are checking two independent data groups and deciding whether the sample mean of two groups is equal or not.

H0 : the mean of two groups is the same


Ref: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
"""
import Preprocessing as pp
import numpy as np
#
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import ks_2samp
from scipy.stats import brunnermunzel
#https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html
from statsmodels.stats import weightstats
from scipy.stats import chi2_contingency

def info_B_M():
  print('Explanation:')
  print("The Brunner-Munzel test is a nonparametric test of the null hypothesis that when values are taken one by one from each group, the probabilities of getting large values in both groups are equal. Unlike the Wilcoxon-Mann-Whitney’s U test, this does not require the assumption of equivariance of two groups. Note that this does not assume the distributions are same. This test works on two independent samples, which may have different sizes.. Brunner and Munzel recommended to estimate the p-value by t-distribution when the size of data is 50 or less. If the size is lower than 10, it would be better to use permuted Brunner Munzel test.")
  
def info_K_S():
  print('\nExplanation:')
  print('Computes the Kolmogorov-Smirnov statistic on 2 samples. This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution. The alternative hypothesis can be either two-sided (default), less or greater. This tests whether 2 samples are drawn from the same distribution. Note that, like in the case of the one-sample KS test, the distribution is assumed to be continuous. If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same. ')
  
def info_K_W():
  print('\nExplanation:')
  print('Tests whether the distributions of two or more independent samples are equal or not. Assumptions: 1. Observations in each sample are independent and identically distributed (iid). 2.Observations in each sample can be ranked. The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs. Post hoc comparisons between groups are required to determine which groups are different. Due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small. A typical rule is that each sample must have at least 5 measurements.')

def info_M_W():
  print('\nExplanation:')
  print("Use only when the number of observation in each sample is > 20 and you have 2 independent samples of ranks. Mann-Whitney U is significant if the u-obtained is LESS THAN or equal to the critical value of U. This test corrects for ties and by default uses a continuity correction.")
  
def info_Z():
  print('\nExplanation:')
  print("Test for mean based on normal distribution, one or two samples. In the case of two samples, the samples are assumed to be independent.")


def info_anova():
  print('\nExplanation:')
  print("The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid: 1.The samples are independent. 2.Each sample is from a normally distributed population. 3.The population standard deviations of the groups are all equal. This property is known as homoscedasticity. ")
  
def info_t_test():
  print('\nExplanation:')
  print( "Calculate the T-test for the means of two independent samples of scores. This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default. We can use this test, if we observe two independent samples from the same or different population, e.g. exam scores of boys and girls or of two ethnic groups. The test measures whether the average (expected) value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. If the p-value is smaller than the threshold, e.g. 1%%, 5%% or 10%%, then we reject the null hypothesis of equal averages.")

###############################################################################################
def print_hypo(p=0.):
  print('\nOutcome:')
  if p <0.05:
    print("--> we reject the null hypothesis H0 at 95 % CL")
  else:
    print("--> we accept null hypothesis H0 at 95 % CL")

def print_normality_test(p=0.):
  if p < 0.05:
    print("--> we reject the null hypothesis H0 = Normal Distribution, at 95 % CL")
  else:
    print("--> we accept null hypothesis H0 = Normal Distribution, at 95 % CL")

def print_stat_p(stat, p):
  print('stat=%.6f, p=%.6f' % (stat, p))

def hypotest(df = None, x = "", y = "", q1 = "", q2 = ""):
  print("\n--- HYPOTHESIS TESTS --- ")

  if y:
    medians = df.groupby(y)[x].median()
    means = df.groupby(y)[x].mean()
    stds = df.groupby(y)[x].std()

    #get np arrays
    x_y_true =  df.loc[ df[y] == 1][x].values
    x_y_false =  df.loc[ df[y] == 0][x].values
  elif q1 and q2:
    medians = df.query(q1)[x].median() , df.query(q2)[x].median()
    means = df.query(q1)[x].mean()     , df.query(q2)[x].mean()
    stds = df.query(q1)[x].std()       , df.query(q2)[x].std()

    #get np arrays
    x_y_true = df.query(q1)[x].values
    x_y_false =  df.query(q2)[x].values
    
  else:
    print("No condition in hypotest..")
    exit(1)

  print('~'*40)
  print("\nMean, Std & Median")
  print("(%s, %s)"%(q1, q2))
  print(80*"-")

  print("Mean values", means)
  print("Std deviations", stds)
  print("Median values", medians)
  
  print(80*"-")
  print("\nNormality tests (whether a data sample has a normal distribution)")
  print(80*"-")

  print("\nShapiro-Wilk:")
  print("H0: the sample for variable %s has a Gaussian distribution for positive %s"%(x, y))
  if x_y_true.size >= 3:
    stat, p = shapiro(x_y_true)
    print_stat_p(stat, p)
    print_normality_test(p)
  else:
    print("Cannot be performed because of size", x_y_true.size )
    
  print("\nH0: the sample for variable %s has a Gaussian distribution for negative %s"%(x, y))
  if x_y_false.size >= 3:
    stat, p = shapiro(x_y_false)
    print_stat_p(stat, p)
    print_normality_test(p)
  else:
    print("Cannot be performed because of size", x_y_false.size )


  # note: we need to avoid
  # ValueError: skewtest is not valid with less than 8 samples; 5 samples were given.
  print('~'*40)
  print("\nD'Agostino’s K^2 Test")
  print("H0: the sample for variable %s has a Gaussian distribution for positive %s"%(x, y))
  if x_y_true.size >= 8:
    stat, p = normaltest(x_y_true) 
    print_stat_p(stat, p)
    print_normality_test(p)
  else:
    print("Cannot be performed because of size", x_y_true.size )

  print("\nH0: the sample for variable %s has a Gaussian distribution for negative %s"%(x, y))
  if x_y_false.size >= 8:
    stat, p = normaltest(x_y_false) 
    print_stat_p(stat, p)
    print_normality_test(p)
  else:
    print("Cannot be performed because of size", x_y_false.size )

  print('~'*40)
  print("\nAnderson-Darling Test")
  print("H0: the sample for variable %s has a Gaussian distribution for positive %s"%(x, y))
  result = anderson(x_y_true)
  print('stat=%.6f' % (result.statistic))
  for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
      print('Probably Normal at the %.1f%% level' % (sl))
    else:
      print('Probably not Normal at the %.1f%% level' % (sl))

  print("\H0: the sample for variable %s has a Gaussian distribution for negative %s"%(x, y))
  result = anderson(x_y_false)
  print('stat=%.6f' % (result.statistic))
  for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
      print('Probably Normal at the %.1f%% CL' % (sl))
    else:
      print('Probably not Normal at the %.1f%% CL' % (sl))
  

  print(80*"-")
  print("\nNonparametric Statistical Hypothesis Tests")
  print(80*"-")

  print("\nMann-Whitney U (rank) test:")
  info_M_W()
  print("\nH0: the distributions of both samples for variable %s are equal (negative or positive %s)"%(x, y))
  stat, p = mannwhitneyu(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)

  print('~'*40)
  print("\nKruskal-Wallis H test:")
  info_K_W()
  print("\nH0: the distributions of all samples for variable %s are equal (negative or positive %s)"%(x, y))
  stat, p = kruskal(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)

  #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp
  print('~'*40)
  print("\nKolmogorov-Smirnov test:")
  info_K_S()
  print("\nH0: The 2 independent samples for the variable %s are drawn from the same continuous distribution (negative or positive %s)"%(x, y))
  stat, p = ks_2samp(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)


  print('~'*40)
  print("\nKolmogorov-Smirnov test using cumulative distributions:")
  print("\nH0: The 2 independent samples for the variable %s are drawn from the same continuous distribution (negative or positive %s)"%(x, y))
  x_y_true_cum = np.cumsum(x_y_true)
  x_y_false_cum = np.cumsum(x_y_false)
  stat, p = ks_2samp(x_y_true_cum, x_y_false_cum)
  print_stat_p(stat, p)
  print_hypo(p)
  
  
  #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.brunnermunzel.html#scipy.stats.brunnermunzel
  print('~'*40)
  print("\nBrunner-Munzel test:")
  info_B_M()
  print("\nH0: when values are taken one by one from each group of the variable %s, the probabilities of getting large values in both groups are equal (negative or positive %s)"%(x, y))
  stat, p = brunnermunzel(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)

  
  print(80*"-")
  print("\nParametric Statistical Hypothesis Tests")
  print(80*"-")

  #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
  print("\nStudent's t-test:")
  info_t_test()
  print("\nH0: there is association in variable %s for positive or negative %s"%(x, y))
  stat, p = ttest_ind(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)

  print('~'*40)
  print("\nStudent's t-test (two-sided for checking identical means):" )
  print("H0: the means of two distributions are identical")
  ttest, pval_t, dof  = weightstats.CompareMeans.from_data(data1= x_y_true,
                                                           data2 = x_y_false).ttest_ind(
                                                             alternative="two-sided",
                                                             usevar="pooled", value=0)
  
  print("t-test = %f p-value = %f DoF = %i"%(ttest, pval_t, dof))
  print_hypo(pval_t)

  #https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html
  print('~'*40)
  print("\nANOVA (one way):")
  info_anova()
  print("\nH0=the means of the %s samples are equal for positive or negative %s"%(x, y))
  stat, p = f_oneway(x_y_true, x_y_false)
  print_stat_p(stat, p)
  print_hypo(p)
  
  #https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html
  print('~'*40)
  print("\nZ-test: two-sided")
  info_Z()
  print("\nH0 : the mean of two independent groups is the same")
  stat, p = weightstats.ztest( x1= x_y_true,
                                     x2 = x_y_false,
                                     value = 0,
                                     alternative='two-sided')
  print_stat_p(stat, p)
  print_hypo(p)
  
                   
  #https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.CompareMeans.ztest_ind.html#statsmodels.stats.weightstats.CompareMeans.ztest_ind
  print('~'*40)
  print("\nZ-test: Two-sided test statistic for checking identical means.")
  print("\nH0: the means of two distributions are identical")
  stat, p  = weightstats.CompareMeans.from_data(data1= x_y_true,
                                                data2 = x_y_false).ztest_ind(alternative="two-sided",
                                                                             usevar="pooled", value=0)  
  print_stat_p(stat, p)
  print_hypo(p)


def chi2test(df = None,
             x1 = "",
             x2 = "",
             cond = ""):

  m1_1 = df[ df[x1] == 1].count()[x1]
  m1_0 = df[ df[x1] == 0].count()[x1]

  m2_1 = df[ df[x2] == 1].count()[x2]
  m2_0 = df[ df[x2] == 0].count()[x2]
  print("\nchi-squared test (Tests whether two categorical variables are related or independent):")
  table = [[m1_1, m1_0], [m2_1, m2_0]]
  print("Contingency table")
  print(x1, "true", "false")
  print(x2, "true", "false")
  print("=")
  print(table)
  print("Total",x1, m1_1 + m1_0)
  print("Total",x2, m2_1 + m2_0)
  stat, p, dof, expected = chi2_contingency(table)
  print("H0: the two samples are independent")
  print('chi2 test stat=%.3f, dof=%i, p=%.3f' % (stat, dof, p))
  print_hypo(p)
  
  
