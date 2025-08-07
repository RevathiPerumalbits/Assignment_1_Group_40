import pandas as pd
import datetime
import os


def log_prediction(features, prediction):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
    }
    log_df = pd.DataFrame([log_entry])

    # Append to CSV log
    log_file = "logs/predictions.csv"
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
