import Preprocessing as pp
import pandas as pd

df  = pp.get_data(indata = 'data/new/TemplateSteliosDuplicates.csv',
                  verbose = True)

df.to_excel('excel/database.xlsx')
