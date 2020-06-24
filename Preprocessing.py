import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype

def get_data(indata = '',
             verbose = False,
             drop_useless_cols = True):
    ####################################################################
    #read in data using pandas
    ####################################################################
    df = pd.read_csv(filepath_or_buffer = indata,
                       sep = ",",
                       header = "infer",
                       engine = "python",
                       encoding='utf-8',
                       error_bad_lines=False)
    #encoding = "ISO-8859-1")

    #check data has been read in properly
    if verbose:
        print("Loaded data:")
        print(df.head())
        print(df.tail())
        print("Plaque Volume:")
        print(df.loc[:, "volume_plaque"].head())
    
    #whitespaces in column names
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    #whitespaces in columns
    df = df.apply(lambda x: x.str.strip() if x.dtype is "object" else x)
    
    
    
    if verbose:
        print("Columns:")
        print(df.columns)

    #get rid of empty rows
    rows= len(df.index)
    df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
    print("Rows reduced from %i to %i after removing empty rows"%(rows, len(df.index)))

    
    #replace nan
    df.fillna(0, inplace = True)

    #check nans
    nans = df.isnull().sum().sum()

    if nans:
        print("Warning: %i NaNs detected"%(nans))
        exit(1)


    # plaque a float
    #df['volume_plaque'] = df['volume_plaque'].astype(float)
    #df['volume_plaque'] = df['volume_plaque'].apply(pd.to_numeric, errors='coerce')

    #to integers
    #df[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)

    #new columns
    df['Elder'] = np.where(df['Age']>=60, 1, 0)

    #to ints
    int_cols = ['Age']

    #to booleans
    bool_cols = ['Elder',
                 'Men',
                 'Women',
                 'CABG',
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
                 'Stroke',
                 'No_Stroke',
                 'In_hospital_mortality',
                 'CIP_CIM',
                 'DGS']

    #to float
    float_cols = [
        'volume_plaque'
    ]
    
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    
    df[bool_cols] = df[bool_cols].astype(bool)

    df[int_cols] = df[int_cols].astype(int)
    
    if df['volume_plaque'].dtype != np.float64:
        print("Error: volume plaque not float!")
        exit(1)
    else:
        print("Info: volume plaque type",  df['volume_plaque'].dtype)

    #data types
    if verbose:
        print("Data types:")
        print(df.dtypes)
    
    #rank the dataframe in descending order of score by plaque. So, ranking is done by Stroke wise
    if set(['Stroke', 'volume_plaque']).issubset(df.columns):
        df["Rank"] = df.groupby("Stroke")["volume_plaque"].rank(ascending=True, method = "average")
        print("Info: ranked per plaque")
    else:
        print ("Error: missing columns. Bye!")
        exit(1)

    # keep entries with plaque
    df = df[ (df['volume_plaque'] >= 0)  ]
        
    #duplicates
    print("Duplicates:")
    dupes = df[df.duplicated(['Name'], keep=False)]
    print(dupes)
    
    rows= len(df.index)
    df.drop_duplicates(subset=['Name'], keep = "first", inplace= True)
    print("Rows reduced from %i to %i after removing duplicated entries"%(rows, len(df.index)))

    #get rid of useless data
    if drop_useless_cols:
        df.drop(["Name", "ID"], axis=1, inplace=True)
    
    #final check
    if verbose:
        print(df.head())

    if verbose:
        print("Df shape:")
        print(df.shape)

        print("Df description:")
        print(df.describe())
        
        rows = len(df.index)
        print("Final Rows", rows)


        

    
    #treat NaNs
    #nans = df.isnull().sum().sum()

   # if nans:
   #     print("Warning: %i NaNs detected"%(nans))
   #     print("NaNs in each row of a DataFrame") 
   #     for i in range(len(data.index)) :
   #         cnan = data.iloc[i].isnull().sum()
        
    #data = data[np.logical_not(np.isnan(data))]               

        


    #treat empty plaque volumes: replace to NaN and then drop
    if nans:
        #rows= len(data.index)
        #data['volume_plaque'].replace('', np.nan, inplace=True)
        #data.dropna(subset=['volume_plaque'], inplace=True)
        #print("Rows reduced from %i to %i after removing NaNs"%(rows, len(data.index)))
        df['volume_plaque'].replace('', 0, inplace=True)


    



    #print data["PlaqueVolume"].head()


    return df
