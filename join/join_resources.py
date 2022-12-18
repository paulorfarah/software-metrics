import pandas as pd

pd.set_option('display.max_columns', None)
df_static_metrics = pd.read_csv('..\\results\\static_features.csv')
print(df_static_metrics.head())
df_resources = pd.read_csv('C:\\Users\\paulo\\ufpr\\datasets\\bcel\\merged.csv')
print(df_resources.head())
df_resources['project_name'] = 'commons-bcel'

df_all = pd.merge(left=df_static_metrics, right=df_resources, left_on=['project_name', 'commit_hash', 'class'],
                  right_on=['project_name', 'commit_hash', 'class_name'])

print(df_all.head())