import requests
import numpy as np
import pandas as pd
from scipy.stats import kurtosis,skew
def extract_features(samples):
    samples=samples.astype(np.float32)
    amplitude=np.abs(samples)
    phase=np.angle(samples+1j)
    return [np.mean(amplitude),np.std(amplitude),kurtosis(amplitude),skew(amplitude),np.mean(phase),np.std(phase)]
def get_file_size(url):
    r=requests.head(url)
    return int(r.headers["Content-Length"])
def extract_features_from_url(url,dataset_name,label,window_size=100000,percentage=0.2):
    file_size=get_file_size(url)
    target_bytes=int(file_size*percentage)
    maximum_windows=target_bytes//window_size
    print(f"{dataset_name}: using {percentage*100:.1f}% Of Data "
          f"({maximum_windows} windows)")
    response=requests.get(url,stream=True)
    buffer=b''
    rows=[]
    processed_windows=0
    for chunk in response.iter_content(chunk_size=1024*1024):
        buffer+=chunk
        while len(buffer)>=window_size:
            segment=np.frombuffer(buffer[:window_size],dtype=np.int8)
            buffer=buffer[window_size:]
            features=extract_features(segment)
            rows.append([dataset_name,label,*features])
            processed_windows+=1
            if processed_windows>=maximum_windows:
                return rows
    return rows
datasets={
    "CLEAN_STATIC":{
        "url":"https://rnl-data.ae.utexas.edu/datastore/texbat/cleanStatic.bin",
        "label":0
    },
    "CLEAN_DYNAMIC":{
        "url":"https://rnl-data.ae.utexas.edu/datastore/texbat/cleanDynamic.bin",
        "label":0
    }
}
all_rows=[]
for name,info in datasets.items():
    print(f"\nProcessing {name}...")
    rows=extract_features_from_url(url=info["url"],dataset_name=name,label=info["label"],window_size=100000,percentage=0.2)
    all_rows.extend(rows)
columns=["Dataset","Label","Mean Of Amplitude","Standard Deviation Of Amplitude","Kurtosis Of Amplitude","Skewness Of Amplitude","Mean Of Phase","Standard Deviation Of Phase"]
data_frame=pd.DataFrame(all_rows,columns=columns)
data_frame.to_csv("Samples.csv",index=False)
print("\nFile Is Saved As: Samples.csv")