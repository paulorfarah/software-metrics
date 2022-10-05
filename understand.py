import pydriller
import shutil
import subprocess
import os
import time
import sys


def check(chash):
    if os.path.exists(chash + '.csv'):
        print("Commit", chash, "already collected, skipping...")
        return True
    else:
        print("Commit", chash, "not found, collecting...")
        return False


gr = pydriller.Git("commons-bcel")
commits = ['a9c13ede0e565fae0593c1fde3b774d93abf3f71', 'bebe70de81f2f8912857ddb33e82d3ccc146a24e',
           'bbaf623d750f030d186ed026dc201995145c63ec', 'fa271c5d7ed94dd1d9ef31c52f32d1746d5636dc',
           'dce57b377d1ad6711ff613639303683e90f7bcc8', '9174edf0d530540c9f6df76b4d786c5a6ad78a5d',
           '3aecd517ad0ac4c83828a5f89b6b062acb6f4f6a', 'f38847e90714fbefc33042912d1282cc4fb7d43e',
           '893d9bbcdbd5ce764db1a38eccd73af150e5d34d', '9cd000cc265bfd0997e0277363dbe28ed8a28714',
           'f70742fde892c2fac4f0862f8f0b00e121f7d16e', 'a303928530ee61e45b4523cd5894c9a8bdb9deaa',
           '647c723ba1262e1ffce520524692b366a7fde45a', 'fe98b6f098069607955a68f1d695031c011f6452',
           '5bfa4baa2b7b2cc3dc4cc2600bbcd5d74df7451c', '8fb97bd21c565e0da8300d6a87a95d0fe812bca8',
           'daada0977098e6633de09fe4c73643ddd8331f06']

# for commit in gr.get_list_commits():
#     commits.append(commit.hash)

for commit in commits:
    if check(commit):
        continue
    
    gr.clear()
    print("git checkout on commit", commit + "...")
    gr.checkout(commit)
    if os.path.exists(commit + '.und'):
        print("deleting possibly corrupt project files...")
        shutil.rmtree(commit + '.und')
    print("creating the project", commit + ".udb ...")
    subprocess.run(['und', 'create', '-db', commit + '.und', '-languages', 'java'])
    os.makedirs(commit + '.und/local')
    print("adding java files to project...")
    subprocess.run(['und', '-db', commit + '.und', 'add', '.'])
    print("analyzing source code for commit", commit + "...")
    subprocess.run(['und', 'analyze', '-db', commit + '.und'])
    print("adding the metrics to the project and setting up the environment...")
    subprocess.run(
        ['und', 'settings', '-metricmetricsAdd', 'AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict',
         'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'CountClassBase',
         'CountClassCoupled', 'CountClassDerived', 'CountDeclClass', 'CountDeclClassMethod',
         'CountDeclClassVariable', 'CountDeclFile', 'CountDeclFunction', 'CountDeclInstanceMethod',
         'CountDeclInstanceVariable', 'CountDeclMethod', 'CountDeclMethodAll', 'CountDeclMethodDefault',
         'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic', 'CountInput',
         'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
         'CountLineComment', 'CountOutput', 'CountPath', 'CountSemicolon', 'CountStmt', 'CountStmtDecl',
         'CountStmtExe', 'Cyclomatic', 'CyclomaticModified', 'CyclomaticStrict', 'Essential',
         'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential',
         'MaxInheritanceTree', 'MaxNesting', 'PercentLackOfCohesion', 'RatioCommentToCode', 'SumCyclomatic',
         'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential', commit + '.und'])
    subprocess.run(['und', 'settings', '-MetricFileNameDisplayMode', 'RelativePath', commit + '.und'])
    subprocess.run(['und', 'settings', '-MetricDeclaredInFileDisplayMode', 'RelativePath', commit + '.und'])
    subprocess.run(['und', 'settings', '-MetricShowDeclaredInFile', 'on', commit + '.und'])
    subprocess.run(['und', 'settings', '-MetricShowFunctionParameterTypes', 'on', commit + '.und'])
    print("calculating metrics for", commit + "...")
    subprocess.run(['und', 'metrics', commit + '.und'])
    print("deleting", commit + '.und')
    if os.path.exists(commit + '.und'):
        shutil.rmtree(commit + '.und')
    if os.path.exists("/users/rogerioc/.local/share/Scitools/Db/"+commit):
        shutil.rmtree("/users/rogerioc/.local/share/Scitools/Db/"+commit) 
    print("resetting repository", commit + "...")
    try:
        gr.reset()
    except:
        print(sys.exc_info())
    print("waiting for 3 seconds, it is safe to ctrl+c here...", flush=True)
    for i in range(300):
        time.sleep(0.01)