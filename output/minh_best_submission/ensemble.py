import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.stats import rankdata

all_csv_paths = [
    "./submission_927.csv",
    "./submission_931.csv",
    "./submission_yamnet.csv",
]

test_csv = pd.read_csv("./submission_927.csv")

all_preds = []

for i in tqdm(range(len(all_csv_paths))):
    all_preds.append(pd.read_csv(all_csv_paths[i]))

preds = np.zeros(shape=[1992, 24], dtype=np.float32)

for i in tqdm(range(1992)):
    l = []
    for k in range(len(all_preds)):
        l.append(all_preds[k].iloc[i][1:].tolist())
    preds[i] += np.mean(l, axis=0)

for i in range(24):
    test_csv["s" + str(i)] = preds[:, i]
    if i == 3:
        test_csv["s" + str(i)] = all_preds[0]["s" + str(i)].tolist()
    # if i == 18:
    #     test_csv["s" + str(i)] = 0.5 * np.array(
    #         all_preds[0]["s" + str(i)].tolist()
    #     ) + 0.5 * np.array(all_preds[1]["s" + str(i)].tolist())

test_csv.to_csv("./ensemble.csv", index=False)

argmaxs = []
for i in range(len(test_csv)):
    pred_i = test_csv.iloc[i][1:].tolist()
    argmaxs.append(np.argmax(pred_i))

print(Counter(argmaxs))
