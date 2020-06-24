
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import Preprocessing as pp
import HypoTest as ht
import Cases as cs

def histo(df = None,
          x = None,
          title = "",
          query1 = None,
          label1 = "",
          query2 = None,
          label2 = "",
          y = "",
          xmin = None,
          xmax = None,
          binw = None,
          xlabel = "",
          bstyle = 2,
          folder = ".",
          name = "fig",
          display = False,
          use_queries_for_hypo = False ):

    #all data
    print("\nAll data")
    print(df.describe())

    
    #select
    if query1:
        df1 = df.query( query1 ).copy()
        print("\n", label1)
        print(df1.describe())
        
    if query2:
        df2 = df.query( query2 ).copy()
        print("\n", label2)
        print(df2.describe())

    #stats
    if y:
        columns_to_show = [x]
        print( "Statical information\n",
               df.groupby([y])[columns_to_show].agg([np.mean,
                                                     np.std,
                                                     np.min, 
                                                     np.max,
                                                     "count",
                                                     "median"]
    ) )


    #hypo-test
    if use_queries_for_hypo:
        ht.hypotest(df = df, x = x, y = y, q1 = query1, q2 = query2)
    else:
        ht.hypotest(df = df, x = x, y = y)
    
    
    #limits
    xmin = xmin if xmax else df[x].min()
    xmax = xmax if xmax else df[x].max()
    
    #default bin width
    if not binw:
        total = df[x].value_counts(dropna=True).sum()
        print("Sum of %s = %f"%(title, total))
        binw = round( math.sqrt( math.fabs(total)) )
        
    #bins 1
    bins_style1 = np.arange(start = xmin,
                            stop = xmax,
                            step = binw,
                            dtype = 'float')

    #bins 2
    list_bins = []
    xval = xmax
    while xval > xmin:
        list_bins.append(xval)
        xval -= binw

    bins_style2 = np.asarray( sorted(list_bins) )

    #bins
    bins = bins_style2 if bstyle else bins_style1
        
    if xmax:
        print("\nOverflow", xmax)
        bins = np.append(bins, bins[-1] + binw)

    nbins = bins.size
    print("Histogram range %f:%f binw=%.2f bins=%s"%(xmin,
                                                   xmax,                                            
                                                   binw,
                                                   np.array2string(bins, separator='|'),
    ))


    #numeric overflow
    #epsilon = (df[x].max() - df[x].min())/100.
    epsilon = binw/2.
    if xmax:
        df[x] = df[x].apply( lambda x :
                             xmax + epsilon if x > xmax else x)  
    #plot
    sns.set()
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(15,7))
    ax.set_title(title)
    
    ax1 = sns.distplot(df1[x],
                       bins=bins,
                       color = 'blue',
                       #hist_kws={"range": [xmin, xmax]},
                       label = label1,
                       kde=False,
                       rug=False,
                       ax = ax)

    
    ax2 = sns.distplot(df2[x],
                       bins=bins,
                       color = 'red',
                       #hist_kws={"range": [xmin, xmax]},
                       label = label2,
                       kde=False,
                       rug=False,
                       ax = ax)

    ax.set(xlabel=xlabel, ylabel='Entries')
    plt.tight_layout()
    plt.legend()

    plt.savefig('%s/%s.png'%(folder, name))

    if display:
        plt.show()

    plt.close('all')

############################################################################
def multiple(df = None, test = False, display = False):

    items = cs.items

    cases = cs.cases

    for c, item in enumerate(items):
        for case in cases: 
            word_item = item.replace("_", " ").title()
            if "plaque" in item:
                word_item = "Plaque Volume"
            word_case = case.replace("_", " ")
            if case == 'DGS':
                word_case = "Delirium"
            xlabel = word_item
            title = word_item + " and " + word_case
            if "plaque" in item:
                word_item+=" [$mm^3$]"
                xmin = None
                xmax = None
                binw = 100
            if "Age" in item:
                word_item+=" [years]"
                xmin = 20
                xmax = 100
                binw = 1

            print(80*"=")
            print("Case", title)
            print(80*"=")
            
            histo(df = df,
                  x = item,
                  title = title,
                  query1 = case+" == 0", label1 = "Negative "+word_case,
                  query2 = case+"== 1", label2 = "Positive "+word_case,
                  y = case,
                  xmin = xmin,
                  xmax = xmax,
                  binw = binw,
                  xlabel = word_item,
                  bstyle = 2,
                  folder = "plots",
                  name = item + "_" + case,
                  display = display)
            if test and c == 0:
                break
##########################################################################
def pie(title = '', percentages = [], labels = [], folder = 'pies'):
    color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9', 
                          '#C1F0F6', '#0099CC']

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#909090'
    plt.rcParams['axes.labelcolor']= '#909090'
    plt.rcParams['xtick.color'] = '#909090'
    plt.rcParams['ytick.color'] = '#909090'
    plt.rcParams['font.size'] = 25

    explode=(0.1,0)
    ax.pie(percentages, explode=explode, labels=labels,  
           colors=color_palette_list[0:2],
           autopct='%1.1f%%', 
           shadow=False,
           startangle=15,   
           pctdistance=1.2,
           labeldistance=1.42)
    ax.axis('equal')
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))
    name = title.replace(' ', '_')
    plt.savefig('%s/%s.png'%(folder, name))
    plt.show()

##########################################################################
def specific(df = None,
             q1 = "", l1 = "",
             q2 = "", l2 = "",
             title = "",
             xmin = None,
             xmax = None,
             binw = None, ):

    name = (l1 + "_" + l2).replace(" ", "_")
    histo(df = df,
          x = "volume_plaque",
          title = title,
          query1 = q1, label1 = l1,
          query2 = q2, label2 = l2,
          y = None,
          xmin = xmin,
          xmax = xmax,
          binw = binw,
          xlabel = "volume plaque [$mm^2$]",
          bstyle = 2,
          folder = "graphs",
          name = name,
          display = False,
          use_queries_for_hypo = True)
###########################################################################
def special(df = None):
    cases = cs.specific

    for c in cases:
        print(60*"#")
        tcase = c.replace("_", " ")
        print(tcase)
        print(60*"#")
        specific(df = df,
                 q1 = "Partial_clamping == 1 and %s == 1"%(c),
                 l1 = 'Partial Clamping + '+tcase,
                 q2 = "cross_clamping_no_touch_Aorta == 1 and %s == 1"%(c),
                 l2 = 'Cross Clamping + '+tcase,
                 title = "Cross & Partial Clamping for "+tcase)
        
        ht.chi2test(df = df,
                    x1 = "Partial_clamping",
                    x2 = "cross_clamping_no_touch_Aorta",
                    cond = c)
###########################################################################
def stats(df = None):
    cases = cs.cases

    total = df.shape[0]
    for c in cases:
        found_true = df[c].values.sum()
        found_false = (~df[c]).values.sum()
        print("%s: True=%i/%i %f %%, False=%i/%i %f %%"%(c,
                                                         found_true, total,
                                                         100.*found_true/total,
                                                         found_false, total,
                                                         100.*found_false/total)

        )
###########################################################################
def make_pie(df = None, title='', case = '', labels = []):
    total = df.shape[0]
    ntrue = df[case].values.sum() * 100. / total
    nfalse = (~df[case]).values.sum() * 100. / total

    pie(title = title, percentages = [ntrue, nfalse], labels = labels)


def make_histo_bars(df = None):
    techs = [
        "CABG",
        'Valve_AKE',
        'Valve_MKE_MKR',
        'Valve_TKR_TKE',
        'Valve_AKE_Umman',
        'Valve_KOMBI',
        'CABG_Valve_AKE_CABG',
        'CABG_Valve_MK_CABG',
        'CABG_Valve_TK_CABG',
        'CABG_Valve_AK_MK_TK_CABG',
        'CABG_Umma',
        'cross_clamping_no_touch_Aorta',
        'Partial_clamping',
        'No_calcification',
    ]

    probs = [
        'Stroke',
        #'No_Stroke',
        'In_hospital_mortality',
        'CIP_CIM',
        'DGS'
    ]
    for tech in techs:
        for prob in probs:
            qu = "%s == 1"%(prob)
    
            df1 = df.query( qu ).copy()
            counts = int(df1[tech].values.sum())
            print(f'{tech:30} {prob:30} {counts:3}')
        
###########################################################################
df  = pp.get_data(indata = 'data/new/TemplateSteliosDuplicates.csv',
                  verbose = True)

df.sort_values(by=['volume_plaque'], ascending=True, inplace=True)
print("10 Top Ranked values:")
print(df['volume_plaque'].head(10))
print("10 Bottom Ranked values:")
print(df['volume_plaque'].tail(10))

do_multiple = False
do_special = False
do_stats = False
pie_Mort = False
pie_Stroke = False
pie_cip_cim = False
pie_dgs = False
do_histo_bars = True

if do_multiple:
    multiple(df = df, test = False, display = False)

if do_special:
    special(df = df)

if do_stats:
    stats(df = df)

if pie_Mort:
    make_pie(df,
             title = 'In-hospital Mortality',
             case = 'In_hospital_mortality',
             labels = ['Positive', 'Negative'])

if pie_Stroke:
    make_pie(df,
             title = 'Stroke',
             case = 'Stroke',
             labels = ['Positive', 'Negative'])

if pie_cip_cim:
    make_pie(df,
             title = 'CIP CIM',
             case = 'CIP_CIM',
             labels = ['Positive', 'Negative'])

if pie_dgs:
    make_pie(df,
             title = 'Delirium',
             case = 'DGS',
             labels = ['Positive', 'Negative'])
    
if do_histo_bars:
    make_histo_bars(df)
