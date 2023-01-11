import pydriller
import argparse
from csv import reader
import shutil
import subprocess

import git
from pydriller import Repository


def runJar(path1, path2, hash1, hash2):
    repo1 = pydriller.Git(path1)
    repo1.checkout(hash1)
    files1 = [x for x in repo1.files() if x.endswith('.java')]

    repo2 = pydriller.Git(path2)
    repo2.checkout(hash2)
    files2 = [x for x in repo2.files() if x.endswith('.java')]

    csvPath = args.absolutePath + args.projectName + "-results.csv"
    try:
        f = open(csvPath, "x")
    except:
        print("file exists")
    for file in files1:
        file_temp = file.replace(args.absolutePath + "projectA", '')
        if any(file_temp in s for s in files2):
            file2 = args.absolutePath + "projectB" + file_temp
            # classPreviousCommit classCurrentCommit csvPath projectName currentCommit previousCommit
            print('java -jar ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar ' + file2 + ' ' + file + ' ' + csvPath)
            subprocess.call(
                ['java', '-jar', 'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar', file2, file, csvPath,
                 args.projectName, hashA, hashB])


if __name__ == "__main__":
    print('starting...')
    ap = argparse.ArgumentParser(description='Extractor for changeDistiller')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--pathB', required=True)
    ap.add_argument('--commits', required=True, help='csv with list of commits to compare commitA and commitB')
    ap.add_argument('--projectName', required=True)
    ap.add_argument('--absolutePath', required=True)
    ap.add_argument('--mode', required=True, help='mode - tag for commits with tag, csv - for csv of commits')
    args = ap.parse_args()

    # folder with repo: projectA and projectB
    print(args.pathA)
    pathA = pydriller.Git(args.pathA)
    pathB = pydriller.Git(args.pathB)

    repo = git.Repo(args.pathA)
    tags = repo.tags

    i = 0
    commit_A = ''
    commit_B = ''
    print(args.mode)
    if (args.mode == 'tag'):
        for tag in tags:
            if (i == 0):
                commit_A = tag
                i += 1
            else:
                print('else')
                hashA = pathA.get_commit_from_tag(commit_A.name).hash
                hashB = pathB.get_commit_from_tag(tag.name).hash
                pathA.checkout(hashA)
                pathB.checkout(hashB)
                runJar(pathA, pathB, str(hashA), str(hashB))
                commit_A = tag
    else:
        print(args.commits)
        first = True
        cur_com = ''
        prev_com = ''
        for line in reversed(list(open(args.commits))):
            # print(line)
            if first:
                cur_com = line
                first = False
            else:
                prev_com = cur_com
                cur_com = line
                # print(prev_com)
                # pathA.checkout(prev_com)
                # print(cur_com)
                # pathB.checkout(cur_com)
                runJar(args.pathA, args.pathB, prev_com, cur_com)
                print('---')

