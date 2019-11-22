import os
import glob
import pandas as pd
import numpy as np

results = glob.glob('results/*/training_history.csv')
results = [pd.read_csv(res) for res in results]
results = pd.concat(results)
best_loss = results.groupby('dir').max()[['val_custom_dice_score']].sort_values('val_custom_dice_score').reset_index()
for _, model in best_loss.iterrows():
    try:
        os.rename(model.dir + '/best_model.h5', model.dir + '/best_model_{}.h5'.format(np.round(model.val_custom_dice_score, 5)))
        print(model.dir)
    except Exception as e:
        pass
