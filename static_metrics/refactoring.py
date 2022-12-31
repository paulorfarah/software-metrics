from asyncore import write
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git
import os
import json


def parse_json(path, refactoring_dict, commit_sha1):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)

        for json_commit in json_data["commits"]:
            # commit_sha1 = json_commit['sha1']
            # refactoring_dict = {commit_sha1: {}}
            for json_refactoring in json_commit["refactorings"]:
                print(json_refactoring["type"] + ' (parse_json)')
                refactoring_type = json_refactoring["type"]
                if refactoring_type not in refactoring_dict[commit_sha1]:
                    refactoring_dict[commit_sha1][refactoring_type] = {}

                for json_left in json_refactoring["leftSideLocations"]:
                    file_path = json_left["filePath"]
                    code_element_type = json_left["codeElementType"]
                    code_element = json_left["codeElement"]
                    if file_path not in refactoring_dict[commit_sha1][refactoring_type]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path] = {}

                    if code_element_type not in refactoring_dict[commit_sha1][refactoring_type][file_path]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type] = {'source': 1}
                    elif 'source' not in refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]['source'] = 1
                    else:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]['source'] += 1
                    # else:
                    #     refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]['source'] += 1

                for json_left in json_refactoring["rightSideLocations"]:
                    file_path = json_left["filePath"]
                    code_element_type = json_left["codeElementType"]
                    code_element = json_left["codeElement"]
                    if file_path not in refactoring_dict[commit_sha1][refactoring_type]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path] = {}

                    if code_element_type not in refactoring_dict[commit_sha1][refactoring_type][file_path]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type] = {
                            'destination': 1}
                    elif 'destination' not in refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]['destination'] = 1
                    else:
                        refactoring_dict[commit_sha1][refactoring_type][file_path][code_element_type]['destination'] += 1
    return refactoring_dict

def convertJson(writer, ref_dict, commit_sha1):
    for type_key, type_val in ref_dict[commit_sha1].items():
        if isinstance(type_val, dict):
            for file_key, file_val in type_val.items():
                if isinstance(file_val, dict):
                    for elem_type_key, elem_type_val in file_val.items():
                        source = 0
                        destination = 0
                        if isinstance(elem_type_val, dict):
                            if 'source' in elem_type_val:
                                source = elem_type_val['source']
                            if 'destination' in elem_type_val:
                                destination = elem_type_val['destination']
                        row = [commit_sha1, type_key, file_key, elem_type_key, source, destination]
                        writer.writerow(row)
                        f.flush()
                else:
                    row = [commit_sha1, type_key, file_key, file_val, 0, 0]
                    writer.writerow(row)
                    f.flush()
        else:
            print('>>> Not a dictionary: ' + type_key)


if __name__ == "__main__":

    print('starting refactoring miner...')
    ap = argparse.ArgumentParser(description='Extract refactorings')
    ap.add_argument('--project_name', required=True)
    ap.add_argument('--git_repo_folder_A', required=True)
    ap.add_argument('--git_repo_folder_B', required=True)
    ap.add_argument('--commits', required=True, help='csv with list of commits to compare commitA and commitB')
    ap.add_argument('--mode', required=True, help='mode - tag for commits with tag, csv - for csv of commits')
    args = ap.parse_args()

    # folder with repo: projectA and projectB
    pathA = pydriller.Git(args.git_repo_folder_A)
    pathB = pydriller.Git(args.git_repo_folder_B)

    release = 1
    tag_previous = ''
    refactoring_list = ["Extract Method", "Inline Method", "Rename Method", "Move Method", "Move Attribute",
                        "Pull Up Method", "Pull Up Attribute", "Push Down Method", "Push Down Attribute",
                        "Extract Superclass", "Extract Interface", "Move Class", "Rename Class",
                        "Extract and Move Method", "Rename Package", "Move and Rename Class", "Extract Class",
                        "Extract Subclass", "Extract Variable", "Inline Variable", "Parameterize Variable",
                        "Rename Variable", "Rename Parameter", "Rename Attribute", "Move and Rename Attribute",
                        "Replace Variable with Attribute", "Replace Attribute (with Attribute)", "Merge Variable",
                        "Merge Parameter", "Merge Attribute", "Split Variable", "Split Parameter", "Split Attribute",
                        "Change Variable Type", "Change Parameter Type", "Change Return Type", "Change Attribute Type",
                        "Extract Attribute", "Move and Rename Method", "Move and Inline Method",
                        "Add Method Annotation",
                        "Remove Method Annotation", "Modify Method Annotation", "Add Attribute Annotation",
                        "Remove Attribute Annotation",
                        "Modify Attribute Annotation", "Add Class Annotation", "Remove Class Annotation",
                        "Modify Class Annotation", "Add Parameter Annotation", "Remove Parameter Annotation",
                        "Modify Parameter Annotation", "Add Variable Annotation", "Remove Variable Annotation",
                        "Modify Variable Annotation", "Add Parameter", "Remove Parameter", "Reorder Parameter",
                        "Add Thrown Exception Type", "Remove Thrown Exception Type", "Change Thrown Exception Type",
                        "Change Method Access Modifier", "Change Attribute Access Modifier", "Encapsulate Attribute",
                        "Parameterize Attribute", "Replace Attribute with Variable",
                        "Add Method Modifier (final, static, abstract, synchronized)",
                        "Remove Method Modifier (final, static, abstract, synchronized)",
                        "Add Attribute Modifier (final, static, transient, volatile)",
                        "Remove Attribute Modifier (final, static, transient, volatile)",
                        "Add Variable Modifier (final)", "Add Parameter Modifier (final)",
                        "Remove Variable Modifier (final)", "Remove Parameter Modifier (final)",
                        "Change Class Access Modifier",
                        "Add Class Modifier (final, static, abstract)",
                        "Remove Class Modifier (final, static, abstract)",
                        "Move Package", "Split Package", "Merge Package", "Localize Parameter",
                        "Change Type Declaration Kind (class, interface, enum)", "Collapse Hierarchy",
                        "Replace Loop with Pipeline", "Replace Anonymous with Lambda", "Merge Class",
                        "Inline Attribute",
                        "Replace Pipeline with Loop", "Split Class", "Split Conditional"]

    # class_dict = {}

    csvPath = "results/refactoring/" + args.project_name + "-results-refactoring-metrics.csv"
    with open(csvPath, "w") as f:
        writer = csv.writer(f)

        if (args.mode == 'tag'):
            repo = git.Repo(args.git_repo_folder_A)
            tags = repo.tags

            for tag_current in tags:
                hashCurrent = pathB.get_commit_from_tag(tag_current.name).hash
                print('######################## hashCurrent #############################', hashCurrent)
                pathA.checkout(hashCurrent)
                refactoring_dict = {hashCurrent: {}}
                # rows = []
                if release == 1:
                    tag_previous = tag_current
                    row = ['commit', 'refactoring', 'filePath', 'codeElementType', 'source', 'destination']# + refactoring_list
                else:
                    hashPrevious = pathA.get_commit_from_tag(tag_previous.name).hash
                    pathB.checkout(hashPrevious)
                    commits_range = Repository(args.git_repo_folder_A, from_commit=hashPrevious, to_commit=hashCurrent).traverse_commits()
                    for cc in commits_range:
                        print(cc.hash)
                        path_to_json_file = 'results/refactoring/json/' + args.project_name + '_' + cc.hash + '-refactoringminer.json'
                        out = os.popen(
                            'RefactoringMiner-2.3.2/bin/RefactoringMiner -c ' + args.git_repo_folder_A + ' ' + cc.hash + ' -json ' + path_to_json_file).read()
                        refactoring_dict = parse_json(path_to_json_file, refactoring_dict, hashCurrent)

                    convertJson(writer, refactoring_dict, hashCurrent)
                # for row in rows:

                tag_previous = tag_current
                release += 1

        else:
            print('csv mode.')
            versions = []
            with open(args.commits) as f:
                versions = f.read().splitlines()
                versions.reverse()

                version_prev = versions[0]
                # row = ['project', 'commit', 'commitprevious', 'class', 'release', 'BOC', 'TACH', 'FCH', 'LCH',
                #        'CHO', 'FRCH', 'CHD', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF']
                row = refactoring_list

                pathA.checkout(version_prev)
                print('checkout hashCurrent: ' + version_prev)
                repo = git.Repo(args.git_repo_folder_A)

                for version in versions[1:]:
                    print('######################## hashCurrent #############################', version)
                    pathA.checkout(version)
                    refactoring_dict = {version: {}}
                    if release == 1:
                        row = ['commit', 'refactoring', 'filePath', 'codeElementType', 'source',
                               'destination']
                    else:
                        # hashPrevious = pathA.get_commit_from_tag(tag_previous.name).hash
                        pathB.checkout(version_prev)
                        commits_range = Repository(args.git_repo_folder_A, from_commit=version_prev,
                                                   to_commit=version).traverse_commits()
                        for cc in commits_range:
                            print(cc.hash)
                            path_to_json_file = 'results/refactoring/json/' + args.project_name + '_' + cc.hash + '-refactoringminer.json'
                            out = os.popen(
                                'RefactoringMiner-2.3.2/bin/RefactoringMiner -c ' + args.git_repo_folder_A + ' ' + cc.hash + ' -json ' + path_to_json_file).read()
                            refactoring_dict = parse_json(path_to_json_file, refactoring_dict, version)

                        convertJson(writer, refactoring_dict, version)
                    version_prev = version
                    release += 1

    print(args.project_name + ' ended.')
