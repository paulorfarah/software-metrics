import os
import subprocess

from pydriller import Git

path1 = 'jgit1'
path2 = 'jgit2'
commits = ['73a432514936fcee1386b37a0d60dcd913706bd1', 'e7142a3937825074ec68e4bacba30f9c962bd1e4',
           'df479df17676148bf6391401a704a7d7265c45fa', '399d2929ee0912bfeda8a4ef125f1b96bb7fd144',
           'f3b8653021830d8a503c5e7a9d33cb82b16db739', 'f2c58503d76be1dfb8072f6a7d592f88133708e1',
           '41b194c718d50763a79951029787cca70a5804a5', '1447159b926076c9222a96b3abbe17571953a74f',
           '4f0daa3bb2a5fa28286f1973deb9d13996cc73cc', 'bf32c9102fb1b5fdfa7a26a120b5d9a6b428dd2f']

output = "jgit-changedistiller-results.csv"

def runJar(projectName, path1, path2, currentCommit, previousCommit, output):

    files1 = [x for x in path1.files() if x.endswith('.java')]
    files2 = [x for x in path2.files() if x.endswith('.java')]
    dir = os.getcwd()
    try:
        f = open(output, "x")
    except:
        print("file exists")
    for file in files1:
        file_temp = file.replace(dir + "projectA", '')
        if any(file_temp in s for s in files2):
            file2 = file_temp
            # classPreviousCommit classCurrentCommit csvPath projectName currentCommit previousCommit
            print('java -jar ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar ' + file2 + ' ' + file + ' ' + output)
            subprocess.call(
                ['java', '-jar', 'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar', file2, file, output,
                 projectName, currentCommit, previousCommit])


if __name__ == "__main__":
    print('starting...')

    gr1 = Git(path1)
    gr1.clear()

    gr2 = Git(path1)
    gr2.clear()

    first = True
    cur_com = ''
    prev_com = ''

    for hash in reversed(commits):
        if first:
            cur_com = hash
            first = False
        else:
            prev_com = cur_com
            cur_com = hash
            gr1.checkout(prev_com)

            gr2.checkout(cur_com)
            runJar('jgit', gr1, gr1, prev_com, cur_com, output)
            print('---')
