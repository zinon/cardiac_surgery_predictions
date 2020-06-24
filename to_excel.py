import Preprocessing as pp
import pandas as pd

df  = pp.get_data(indata = 'data/new/TemplateSteliosDuplicates.csv',
                  verbose = True,
                  drop_useless_cols = False)

df.to_excel('data/excel/database_with_names.xlsx')
