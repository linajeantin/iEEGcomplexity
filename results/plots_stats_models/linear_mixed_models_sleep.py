""" 
Created on 2025.04.25
Updated on 2026.02.17
Author: Lina Jeantin

Linear mixed models to study the variation in the values of EEG markers of complexity across sleep stages
Taking into account the variable number of recordings between patients
(patient ID set as random effect)

Requirements:
   df_full: dataframe containing the results of a given marker of EEG complexity across sleep stages
            (see extract_results_df.py)

"""

import pandas as pd
import statsmodels.formula.api as smf

# %%


df_full = pd.read_csv("/path/to/save/dataframes/df_full.csv")

df_avg = df_full.groupby('recording_id', as_index=False).agg({
    'value': 'mean',                  # Mean per recording
    'sleep_stage': 'first',           
    'patient_id': 'first',
})


# %%

# LMM: considering all values per epoch

df_full['patient_id'] = df_full['patient_id'].astype('category')
df_full['sleep_stage'] = df_full['sleep_stage'].astype('category')
df_full['sleep_stage'] = df_full['sleep_stage'].cat.set_categories(['W', 'N2', 'N3', 'REM'], ordered=True)

model = smf.mixedlm("value ~ sleep_stage", df_full, groups=df_full["patient_id"])
result = model.fit()
print(result.summary())



# LMM: considering one value per recording

df_avg['patient_id'] = df_avg['patient_id'].astype('category')
df_avg['sleep_stage'] = df_avg['sleep_stage'].astype('category')
df_avg['sleep_stage'] = df_avg['sleep_stage'].cat.set_categories(['W', 'N2', 'N3', 'REM'], ordered=True)

model = smf.mixedlm("value ~ sleep_stage", df_avg, groups=df_avg["patient_id"])
result = model.fit()
print(result.summary())

# %%
