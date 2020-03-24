"""
Two sampled T-test :-The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test.

H0: there is association between plaque volumes of strokes and no strokes



Two-sample Z test- In two sample z-test , similar to t-test here we are checking two independent data groups and deciding whether the sample mean of two groups is equal or not.

H0 : the mean of two groups is the same


Ref: 
"""
import Preprocessing as pp

#
from scipy.stats import ttest_ind, f_oneway

#https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html
from statsmodels.stats import weightstats
from scipy.stats import chi2_contingency
###############################################################################################
def print_hypo(p=0.):
  if p <0.05:
    print("we reject the null hypothesis at 95 % CL")
  else:
    print("we accept null hypothesis at 95 % CL")

def hypotest(df = None, x = "", y = "", q1 = "", q2 = ""):
  #df  = pp.get_df(verbose = True)
  print("\nHypothesis tests\n")
  if y:
    means = df.groupby(y)[x].mean()
    stds = df.groupby(y)[x].std()

    #get np arrays
    x_y_true =  df.loc[ df[y] == 1][x].values
    x_y_false =  df.loc[ df[y] == 0][x].values
  elif q1 and q2:
    means = df.query(q1)[x].mean(), df.query(q2)[x].mean()
    stds = df.query(q1)[x].std(), df.query(q2)[x].std()

    #get np arrays
    x_y_true = df.query(q1)[x].values
    x_y_false =  df.query(q2)[x].values
    
  else:
    print("No condition in hypotest..")
    exit(1)
    
  print("Mean values", means)
  print("Std deviations", stds)


  print("\nt-test:")
  print("H0: there is association in variable %s for positive or negative %s"%(x, y))
  ttest, pval_t = ttest_ind(x_y_true, x_y_false)
  print("t-test = %f p-value = %f"%(ttest, pval_t))
  print_hypo(pval_t)

  print("\nANOVA:")
  print("H0=the means of the %s samples are equal for positive or negative %s"%(x, y))
  stat, pval = f_oneway(x_y_true, x_y_false)
  print("test-stat = %f p-value = %f"%(stat, pval))
  print_hypo(pval)
  
  #https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html
  print("\nZ-test (for mean based on normal distribution, one or two samples):")
  print("H0 : the mean of two independent groups is the same")
  ztest, pval_Z = weightstats.ztest( x1= x_y_true,
                                     x2 = x_y_false,
                                     value = 0,
                                     alternative='two-sided')
  
  print("Z-test = %f p-value = %f"%(ztest, pval_Z))
  print_hypo(pval_Z)
  
                   
  #https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.CompareMeans.ztest_ind.html#statsmodels.stats.weightstats.CompareMeans.ztest_ind
  print("\nZ-test (two-sided test statistic for checking identical means):")
  print("H0: the means of two distributions are identical")
  ztest, pval_Z  = weightstats.CompareMeans.from_data(data1= x_y_true,
                                                      data2 = x_y_false).ztest_ind(alternative="two-sided",
                                                                                   usevar="pooled", value=0)
  
  print("Z-test = %f p-value = %f"%(ztest, pval_Z))
  print_hypo(pval_Z)

  
  print("\nt-test (two-sided for checking identical means):" )
  print("H0: the means of two distributions are identical")
  ttest, pval_t, dof  = weightstats.CompareMeans.from_data(data1= x_y_true, data2 = x_y_false).ttest_ind(alternative="two-sided", usevar="pooled", value=0)
  
  print("t-test = %f p-value = %f DoF = %i"%(ttest, pval_t, dof))
  print_hypo(pval_t)


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
  
  
