import pandas as pd
import json
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


df = pd.DataFrame(data)


df.to_csv('test_dataset.csv', index=False)