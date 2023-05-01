import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

projects = ['csv', 'easymock', 'gson', 'jgit', 'openfire']

for project_name in projects:
    print(project_name)
    #t-test diff
    # df_perf = pd.read_csv('results/commons-csv/commons-csv-method-performance-diff_filtered.csv')

    # # median
    # df_median = pd.read_csv('data/median/own_dur_trace-method-median-' + project_name + '.csv')
    # sns.lineplot(data=df_median, x='commit_hash', y='own_duration', hue='method_name', legend=False).set(title=project_name)
    # plt.show()

    #all
    df_all = pd.read_csv('data/median/own_dur_trace-all-' + project_name + '.csv', sep=';')
    df_all.columns = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'committer_date']
    sns.scatterplot(data=df_all, x="commit_hash", y='own_duration', hue='method_name', legend=False).set(title=project_name)
    plt.show()
