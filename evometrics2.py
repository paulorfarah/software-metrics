import os
import csv
import argparse
import pydriller
from pydriller import Repository

def check(chash):
    if os.path.exists(chash + '.csv'):
        print("Commit", chash, "already collected, skipping...")
        return True
    else:
        print("Commit", chash, "not found, collecting...")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract evolutionary metrics')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--pathB', required=True)
    ap.add_argument('--commits', required=True, help='csv with list of commits to compare commitA and commitB')
    ap.add_argument('--project_name', required=True)
    # ap.add_argument('--absolute_path', required=True)
    ap.add_argument('--mode', required=True,help='mode - tag for commits with tag, csv - for csv of commits')
    args = ap.parse_args()

    absolute_path = os.getcwd()
    # folder with repo: projectA and projectB
    # print(args.pathA)
    pathA = pydriller.Git(args.pathA)
    pathB = pydriller.Git(args.pathB)

    release = 1
    tag_previous = ''
    # commit_B = ''
    bocArray = {}
    fchArray = {}
    frchArray = {}
    wcdArray = {}
    wfrArray = {}
    lcaArray = {}
    lcdArray = {}
    csbArray = {}
    csbsArray = {}
    acdfArray = {}

    csvPath = absolute_path + "/results/" + args.project_name + "-results-processMetrics.csv"
    f = open(csvPath, "w")
    writer = csv.writer(f)

    # gr = pydriller.Git("commons-bcel")
    # commits = ['a9c13ede0e565fae0593c1fde3b774d93abf3f71', 'bebe70de81f2f8912857ddb33e82d3ccc146a24e',
    #            'bbaf623d750f030d186ed026dc201995145c63ec', 'fa271c5d7ed94dd1d9ef31c52f32d1746d5636dc',
    #            'dce57b377d1ad6711ff613639303683e90f7bcc8', '9174edf0d530540c9f6df76b4d786c5a6ad78a5d',
    #            '3aecd517ad0ac4c83828a5f89b6b062acb6f4f6a', 'f38847e90714fbefc33042912d1282cc4fb7d43e',
    #            '893d9bbcdbd5ce764db1a38eccd73af150e5d34d', '9cd000cc265bfd0997e0277363dbe28ed8a28714',
    #            'f70742fde892c2fac4f0862f8f0b00e121f7d16e', 'a303928530ee61e45b4523cd5894c9a8bdb9deaa',
    #            '647c723ba1262e1ffce520524692b366a7fde45a', 'fe98b6f098069607955a68f1d695031c011f6452',
    #            '5bfa4baa2b7b2cc3dc4cc2600bbcd5d74df7451c', '8fb97bd21c565e0da8300d6a87a95d0fe812bca8',
    #            'daada0977098e6633de09fe4c73643ddd8331f06']

    versions = []
    with open(args.commits) as f:
        versions = f.read().splitlines()
        versions.reverse()


        version_prev = versions[0]
        row = ['project', 'commit', 'commitprevious', 'class', 'release', 'BOC', 'TACH', 'FCH', 'LCH',
               'CHO', 'FRCH', 'CHD', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF']

        pathA.checkout(version_prev)
        print('checkout hashCurrent: ' + version_prev)
        filesCur = [x for x in pathA.files() if x.endswith('.java')]
        for file in filesCur:
            if (file not in bocArray):
                bocArray[file] = release
                fchArray[file] = 0
        writer.writerow(row)

        for version in versions[1:]:
            pathA.checkout(version)
            print('checkout hashCurrent: ' + version)
            filesCur = [x for x in pathA.files() if x.endswith('.java')]

            boc = release
            tach = 0
            fch = 0
            lch = release
            cho = 0
            frch = 0
            chd = 0
            wcd = 0
            wfr = 0
            ataf = 0
            lca = 0
            lcd = 0
            csb = 0
            csbs = 0
            acdf = 0
            # hashPrevious = pathA.get_commit_from_tag(tag_previous.name).hash
            pathB.checkout(version_prev)
            print('checkout hashPrevious: ' + version_prev)
            # filesPrev = pathB.files()
            # filesPrev = [x for x in filesPrev if x.endswith('.java')]
            for file in filesCur:
                if (file not in bocArray):
                    bocArray[file] = release
                    boc = release
                else:
                    boc = bocArray.get(file)
                if (file not in fchArray):
                    fchArray[file] = 0
                if (file not in frchArray):
                    frchArray[file] = 0
                if (file not in wcdArray):
                    wcdArray[file] = 0
                if (file not in wfrArray):
                    wfrArray[file] = 0
                if (file not in lcaArray):
                    lcaArray[file] = 0
                if (file not in lcdArray):
                    lcdArray[file] = 0
                if (file not in csbArray):
                    csbArray[file] = 0
                if (file not in csbsArray):
                    csbsArray[file] = 0
                if (file not in acdfArray):
                    acdfArray[file] = 0
                # get all commits from release n-1 to n, the goal is to find the total amount of changes on a file
                commits_touching_path = Repository(args.pathA, from_commit=version_prev,
                                                   to_commit=version).traverse_commits()
                proj_path = absolute_path + '/' + args.pathA #+ "/" + args.project_name + "/"
                file_temp = file.replace(proj_path, '')
                added_lines = 0
                removed_lines = 0
                loc = 0
                for cc in commits_touching_path:
                    modifiedFiles = [x for x in cc.modified_files if x.filename.endswith('.java')]
                    for m in modifiedFiles:
                        if m.change_type.name == 'ADD' and (m.new_path == file_temp or m.old_path == file_temp):
                            # size of
                            csbsArray[file] = m.nloc
                        if m.change_type.name == 'MODIFY' and (m.new_path == file_temp or m.old_path == file_temp):
                            loc = m.nloc
                            added_lines += m.added_lines
                            removed_lines += m.deleted_lines
                            # first time change
                            if fchArray[file] == 0:
                                fchArray[file] = release
                                fch = release
                            # last time change, the lastest released analayzed
                            lhc = release
                            # if changes have occurred
                            cho = 1
                            # frequency of change
                            frchArray[file] += 1
                            frch = frchArray[file]

                # total amount change, added lines + deleted lines (changed lines are already counted twice )
                tach = added_lines + removed_lines
                if (tach > 0):
                    chd = tach / loc
                    # cumulative weight of change
                    wcdArray[file] += tach * pow(2, boc - release)
                    # sum of change density, to normalize later
                    acdfArray[file] += chd
                    # agregate change size, normalized by frequency change
                    if (frch > 0):
                        ataf = tach / frch
                        # agregate change density normalized by frch
                        acdf = acdfArray[file] / frch
                    # last amount of change
                    lcaArray[file] = tach
                    # last change density
                    lcdArray[file] = chd
                    csbArray[file] += tach

                wcd = wcdArray[file]
                wch = wcd * pow(2, boc - release)
                # cumultive weight frequecy
                wfrArray[file] += (release - 1) * cho
                wfr = wfrArray[file]
                lca = lcaArray[file]
                lcd = lcdArray[file]
                csb = csbArray[file]
                if (csb > 0):
                    csbs = csbsArray[file] / csb

                row = [args.project_name, version, version_prev, file, release, boc, tach, fch, lch, cho, frch, chd,
                       wch, wfr, ataf, lca, lcd, csb, csbs, acdf]
                writer.writerow(row)
            version_prev = version
            release += 1
