import json 
import pandas as pd 
import os 
from datetime import datetime 
with open('prediction.json') as f: 
    data = json.load(f)
    log_entry = {'timestamp': datetime.now().isoformat(),'features': [5.1, 3.5, 1.4, 0.2],'prediction': data['prediction'] } 
    log_df = pd.DataFrame([log_entry]) 
    log_df.to_csv('logs/predictions.csv', index=False)