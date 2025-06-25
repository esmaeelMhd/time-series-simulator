import pandas as pd
import pickle

df = pd.read_csv('./datasets/wastewater.csv')

df = df.set_index(["date"])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
# df = df['2021-09-01 00:00:00+00:00':]
df.reset_index(drop = False, inplace = True)

columns = [
    'IN_METAL_Q',
    'BYPASS_Q',
    'IN_Q',
    'IN_Q_QCF',
    'METAL_Q',
    'T1_NH4',
    'T1_NH4NO3',
    'T1_NH4NO3_NCF',
    'T1_NO3',
    'T1_O2',
    'T1_PROCESSPHASE',
    'T1_STATE_BIOPFOCUS',
    'T1_STATE_BIOPSAFE',
    'T2_NH4NO3_NCF',
    'T2_O2',
    'T2_PROCESSPHASE',
    'SCADAWD_AIRSUPPLY',
    'SCADAWD_PHASEOXYGEN',
    'SS',
    'TEMPERATURE',
    'INLET_PH',
    'OUTLET_PO4',
    'T1_PO4']

use_columns = ['date',
               'IN_METAL_Q',
               'T1_NH4',
               'T1_NO3',
               'OUTLET_PO4',
               'T1_PO4']

with open('df_info.pkl', 'rb') as file:
    df_info = pickle.load(file)
    
for col in df.columns:
    if col not in use_columns:
        df = df.drop([col], axis=1)
        df_info = df_info.drop([col], axis=1)
        
swap_list = ['date', 'IN_METAL_Q', 'T1_NH4', 'T1_NO3', 'OUTLET_PO4', 'T1_PO4']
df = df.reindex(columns=swap_list)

with open('df_info_New_IP.pkl', 'wb') as file:          
    pickle.dump(df_info, file)      

df.to_csv('./datasets/New_IP_June_2023.csv', encoding = 'iso-8859-1', index = False)

