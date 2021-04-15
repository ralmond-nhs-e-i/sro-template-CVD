import pandas as pd
import os
import json
import numpy as np


patients_total = {}
for file in os.listdir('output'):

        if file.startswith('input_2'):
            date = file.split('_')[-1][:-4]

            df = pd.read_csv(os.path.join('output', file))
            
            patients = np.unique(df['patient_id'][df['event']==1])

            patients_total[date] =[int(x) for x in patients]



with open('output/patient_count.json', 'w') as f:
    json.dump({"num_patients": patients_total}, f)
