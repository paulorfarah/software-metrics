import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

projects = ['csv', 'easymock', 'gson', 'jgit', 'openfire']

for project_name in projects:
    print(project_name)

    #all
    df_all = pd.read_csv('../static_metrics/results/changedistiller/changedistiller_all.csv')
    # df_all.columns = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'committer_date']
    df = df_all[['PROJECT_NAME' == project_name]]
    sns.scatterplot(data=df_all, x="CURRENT_COMMIT", y='TOTAL_CHANGES', hue='method_name', legend=False).set(title=project_name)
    plt.show()
