import pandas as pd

df = pd.read_csv('results/ck/ck_all.csv')

projects = ['commons-bcel', 'commons-text', 'commons-csv', 'easymock', 'jgit', 'gson', 'Openfire']
#projects = ['Openfire']
for prj in projects:
    dft = df.loc[df['project_name'] == prj]
    dft.to_csv('results/ck/' + prj + 'ck_2.csv', index=False)
    print(prj + ': ' + str(dft.shape))


# commons-bcel: (73868, 87)
# commons-text: (28502, 87)
# commons-csv: (2525, 87)
# easymock: (29713, 87)
# jgit: (0, 87)
# gson: (11461, 87)
# openfire: (0, 87)



