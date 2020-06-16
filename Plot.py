import Preprocessing as pp
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

data  = pp.get_data(indata = 'data/new/TemplateSteliosDuplicates.csv',
                    verbose = True)

stat_ages_gen = False
stat_ages = False
stackplot_stroke_per_gender = False
stackplot_stroke_per_clamping = False
stackplot_stroke_per_ake = False
stroke_no_stroke_hist = False
die_alive_marginal_plot = False
plaque_gender_hist = False
plaque_dgs_hist = True
scatter_matrix = False
plaque_cabg_hist = False

if stat_ages_gen:

    case = 'In_hospital_mortality'
    #case = 'Stroke'
    #case = 'DGS'
    #case = 'CIP_CIM'

    df = data.query( '%s == True'%(case) ) [ ['Age'] ]
    print(df.describe())

if stat_ages:
    #col = 'Women'
    col = 'Men'
    
    #case = 'In_hospital_mortality'
    #case = 'Stroke'
    #case = 'DGS'
    case = 'CIP_CIM'
    
    #df = data[ data[col]  == True].groupby('In_hospital_mortality')#[[col]].sum()
    #df = df.reset_index()
    #df = data[ data['Men']  == True].groupby('In_hospital_mortality').apply(
    #    lambda x: 100 * x / x.sum() )
    
    #

    #new_col = 'Mort%s'%(col)
    #df[new_col] = df[col]/total
    #df = df[ [col, new_col, 'Age'] ]
    

    df = data.query( '%s == True and %s == True'%(col, case) ) [ ['Age', col] ]
    total = df[col].sum()

    
    print("%s %s : %i"%(case, col, total))
    print(df.head())
    print(df.describe())

if stackplot_stroke_per_gender:
    plt.figure(figsize=(10,10)) 
    data.groupby(['Men', 'Stroke']).size().groupby(level=0).apply(
        lambda x: 100 * x / x.sum()
    ).unstack().plot(kind='bar',stacked=True)

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()


if stackplot_stroke_per_clamping:
    plt.figure(figsize=(10,10)) 
    data.groupby(['CrossClampingNoTouchAorta', 'Stroke']).size().groupby(level=0).apply(
        lambda x: 100 * x / x.sum()
    ).unstack().plot(kind='bar',stacked=True)

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    

if stackplot_stroke_per_ake:
    pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
    plt.figure(figsize=(10,10)) 
    data.groupby(['Valve_AKE', 'Stroke']).size().groupby(level=0).apply(
        lambda x: 100 * x / x.sum()
    ).unstack().plot(kind='bar', stacked=True, color=pal)

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    
if die_alive_marginal_plot:
    mort_plaque = data["PlaqueVolume"] #.where(data["InhospitalMortality"]==1).replace(np.nan, 0)
    no_mort_plaque = data["PlaqueVolume"] #.where(data["InhospitalMortality"]==0).replace(np.nan, 0)

    xmax = 15e3
        
    fig = plt.figure(figsize=(10, 5))    
    ax = sns.jointplot(x=mort_plaque.values,
                       y=no_mort_plaque.values,
                       kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)
 
    plt.title('Plaque Volume')
    plt.ylabel('No mortality')
    plt.xlabel('Mortality')
    plt.xlim(0, xmax)
    plt.show()

if stroke_no_stroke_hist:
    plaque_stroke = data["PlaqueVolume"].where(data["Stroke"]==1).replace(np.nan, 0)
    plaque_no_stroke = data["PlaqueVolume"].where(data["Stroke"]==0).replace(np.nan, 0)

    nbins = 25
    xmin = data["PlaqueVolume"].min()
    xmax = data["PlaqueVolume"].max()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    _ = ax.hist(plaque_stroke.values,
                color='blue',
                bins = nbins,
                range=[xmin, xmax],
                label = "Stroke")
    _ = ax.hist(plaque_no_stroke.values,
                color='red',
                alpha=.3,
                bins = nbins,
                range=[xmin, xmax],
                label = "No stroke")
    ax.set_xlabel("Plaque volume")
    ax.set_ylabel("Entries")
    
    #ax = plt.gca() # gca stands for 'get current axis'
    #data.plot(kind='line', x='PlaqueVolume', y='Stroke',ax=ax)
    #data.plot(kind='line', x='PlaqueVolume', y='NoStroke', color='red', ax=ax)
    #df4.plot.hist(alpha=0.5)
    #ax.hist([plaque_stroke, plaque_no_stroke],
    #        label=("Stroke", "No Stroke"),
    #        bins=25,
    #        range=[data["PlaqueVolume"].min(), data["PlaqueVolume"].max()])
    ax.legend()
    plt.show()

if plaque_dgs_hist:
    plaque_dgs_1 = data["volume_plaque"].where(data["DGS"]==1).replace(np.nan, 0)
    plaque_dgs_0 = data["volume_plaque"].where(data["DGS"]==0).replace(np.nan, 0)
    xmin = data["volume_plaque"].min()
    #xmax = data["PlaqueVolume"].max()
    xmax = 15e3
    fig = plt.figure(figsize=(10, 5))
    ax = sns.kdeplot(plaque_dgs_1, color = 'Red', label='Delirium', shade=True,  clip=(xmin, xmax))
    ax = sns.kdeplot(plaque_dgs_0, color = 'Blue', label='No Delirium', shade=True, clip=(xmin, xmax))
    #plt.yticks([])

    plt.title('Delirium')
    plt.ylabel('Normalized to unit area')
    plt.xlabel('Plaque volume [$mm^3$]')
    plt.xlim(0, xmax)
    plt.show()

if plaque_gender_hist:
    plaque_men = data["PlaqueVolume"].where(data["Man"]==1).replace(np.nan, 0)
    plaque_women = data["PlaqueVolume"].where(data["Man"]==0).replace(np.nan, 0)
    xmin = data["PlaqueVolume"].min()
    #xmax = data["PlaqueVolume"].max()
    xmax = 15e3
    fig = plt.figure(figsize=(10, 5))
    ax = sns.kdeplot(plaque_men, color = 'Orange', label='Men', shade=True,  clip=(xmin, xmax))
    ax = sns.kdeplot(plaque_women, color = 'Green', label='women', shade=True, clip=(xmin, xmax))
    #plt.yticks([])

    plt.title('Gender Plaque Volume')
    plt.ylabel('entries')
    plt.xlabel('plaque volume')
    plt.xlim(0, xmax)
    plt.show()

if plaque_cabg_hist:
    plaque_cabg = data["PlaqueVolume"].where(data["CABG"]==1).replace(np.nan, 0)
    plaque_cabg_null = data["PlaqueVolume"].where(data["CABG"]==0).replace(np.nan, 0)
    xmin = data["PlaqueVolume"].min()
    #xmax = data["PlaqueVolume"].max()
    xmax = 15e3
    fig = plt.figure(figsize=(10, 5))
    ax = sns.kdeplot(plaque_cabg, color = 'Yellow', label='CABG=1', shade=True,  clip=(xmin, xmax))
    ax = sns.kdeplot(plaque_cabg_null, color = 'Blue', label='CABG=0', shade=True, clip=(xmin, xmax))
    #plt.yticks([])

    plt.title('CABG Plaque Volume')
    plt.ylabel('entries')
    plt.xlabel('plaque volume')
    plt.xlim(0, xmax)
    plt.show()
    
if scatter_matrix:
    
    plt.style.use('ggplot')
    scatter_data = data[ [
        "Age",
        "CABG",
        "AKE",
        "MKE/MKR",
        "TKR/TKE",
        "AKE+Umman",
        "KOMBI",
        "AKE+CABG",
        "MK+CABG",
        "TK+CABG",
        "AK+MK/TK+CABG",
        "CABG+Umma",
        "CrossClampingNoTouchAorta",
        "PartialClamping",
        "NoCalcification",
        "PlaqueVolume",
        "Stroke",
        "NoStroke",
        "InhospitalMortality",
        "CIP/CIM",
        "DGS"
    ] ]
    axs = pd.plotting.scatter_matrix(scatter_data,
                                     alpha=0.2,
                                     figsize=(10, 10),
                                     diagonal='kde')
    
    n = len(scatter_data.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50

    plt.show()            



#
#bin_values = np.arange(start=0, stop=3000, step=50)
#rows_index = data['PlaqueVolume'].isin(['US','MQ']) # create index of flights from those airlines
#rows = data[rows_index] # select rows
#group_carriers = us_mq_airlines.groupby('Man')['PlaqueVolume'] # group values by carrier, select minutes delayed
#group_carriers.plot(kind='hist', bins=bin_values, figsize=[12,6], alpha=.4, legend=True) # alpha for transparency
    
