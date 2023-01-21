import pandas as pd

df = pd.read_csv('results/ck/ck_all.csv')

projects = ['commons-bcel', 'commons-text', 'commons-csv', 'easymock', 'jgit', 'gson', 'openfire']

for prj in projects:
    dft = df.loc[df['project_name'] == prj]
    dft.to_csv('results/ck/' + prj + 'ck_2.csv', index=False)
    print(prj + ': ' + str(dft.shape))
