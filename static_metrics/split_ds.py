import pandas as pd

df = pd.read_csv('results/und/und_all.csv')

projects = ['commons-bcel', 'commons-text', 'commons-csv', 'easymock', 'jgit', 'gson', 'Openfire']
#projects = ['Openfire']
for prj in projects:
    dft = df.loc[df['project_name'] == prj]
    dft.to_csv('results/und/' + prj + '-und_2.csv', index=False)
    print(prj + ': ' + str(dft.shape))

# ck:
# commons-bcel: (73868, 87)
# commons-text: (28502, 87)
# commons-csv: (2525, 87)
# easymock: (29713, 87)
# jgit: (0, 87)
# gson: (11461, 87)
# Openfire: (1020902, 87)

# und
# commons-bcel: (91960, 66)
# commons-text: (1070198, 66)
# commons-csv: (24131, 66)
# easymock: (40459, 66)
# jgit: (0, 66)
# gson: (26095, 66)
# Openfire: (1797604, 66)


