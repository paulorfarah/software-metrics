import glob
import os
from asyncore import write
from functools import reduce
from importlib.resources import path
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git
import pandas as pd
import numpy as np


def parse_ck_results(file_path, project_name):
    #
    # list all csv files only
    csv_files = glob.glob(file_path + '*_class.{}'.format('csv'))
    # print(csv_files)
    df_all = pd.DataFrame()
    try:
        # names = ['project_name', 'commit_hash', 'class_name',
        #            'PublicFieldCount', 'IsAbstract', 'ClassLinesOfCode',
        #            'WeighOfClass', 'FANIN', 'TightClassCohesion', 'FANOUT',
        #            'OverrideRatio', 'LCOM3', 'WeightedMethodCount', 'LCOM2',
        #            'NumberOfAccessorMethods', 'LazyClass', 'DataClass',
        #            'ComplexClass', 'SpaghettiCode', 'SpeculativeGenerality',
        #            'GodClass', 'RefusedBequest', 'ClassDataShouldBePrivate',
        #            'BrainClass', 'TotalClass', 'LongParameterList', 'LongMethod',
        #            'FeatureEnvy', 'DispersedCoupling', 'MessageChain',
        #            'IntensiveCoupling', 'ShotgunSurgery', 'BrainMethod',
        #            'TotalMethod', 'TotalClassMethod', 'DiversityTotal',
        #            'DiversityMethod', 'DiversityClass']
        df_all = pd.read_csv('results/ck/ck_all.csv')  # , dtype={"user_id": int, "username": "string"})

    except:
        pass
    for file in csv_files:
        # print(file)
        hash = file.split('/')[-1][:-4]
        hash = hash.replace(project_name + '_', '')
        hash = hash.replace('_class', '')
        # print(hash)
        df1 = pd.read_csv(file)
        df2 = pd.read_csv(file[:-10] + '_method.csv')
        # df_temp = df1.append(df2)
        df_temp = pd.merge(df1, df2, on=['file', 'class'], how='left').reset_index()
        df_temp['commit_hash'] = hash
        df_temp['project_name'] = project_name
        # df_temp.to_csv(file[:-10] + '.csv')
        df_all = df_all.append(df_temp)

    df_all = df_all.rename(columns={'oldName2': 'newName2'})
    df_all.to_csv('results/ck/ck_all.csv', index=False)


def parse_understand_results(file_path, project_name):
    names = []
    csv_files = glob.glob(file_path + '*.{}'.format('csv'))
    df_all = pd.DataFrame()
    try:
        df_all = pd.read_csv('results/understand/und_all.csv', index_col=False)
    except:
        pass
    for file in csv_files:
        hash = file.split('/')[-1][:-4]
        df = pd.read_csv(file)
        df['commit_hash'] = hash
        df['project_name'] = project_name
        df_all = df_all.append(df)
    df_all.to_csv('results/understand/und_all.csv')


def merge_csv_files(file_path, tool):
    # list all csv files only
    csv_files = glob.glob(file_path + '*.{}'.format('csv'))
    df_append = pd.DataFrame()

    for file in csv_files:
        # files.append(file)
        if not '_all.csv' in file:
            if tool == 'changedistiller':
                chageDistillerMetrics = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT",
                                         "CLASS_PREVIOUSCOMMIT", "STATEMENT_DELETE", "STATEMENT_INSERT",
                                         "STATEMENT_ORDERING_CHANGE",
                                         "STATEMENT_PARENT_CHANGE", "STATEMENT_UPDATE", "TOTAL_STATEMENTLEVELCHANGES",
                                         "PARENT_CLASS_CHANGE", "PARENT_CLASS_DELETE", "PARENT_CLASS_INSERT",
                                         "CLASS_RENAMING",
                                         "TOTAL_CLASSDECLARATIONCHANGES", "RETURN_TYPE_CHANGE", "RETURN_TYPE_DELETE",
                                         "RETURN_TYPE_INSERT", "METHOD_RENAMING", "PARAMETER_DELETE",
                                         "PARAMETER_INSERT",
                                         "PARAMETER_ORDERING_CHANGE", "PARAMETER_RENAMING", "PARAMETER_TYPE_CHANGE",
                                         "TOTAL_METHODDECLARATIONSCHANGES", "ATTRIBUTE_RENAMING",
                                         "ATTRIBUTE_TYPE_CHANGE",
                                         "TOTAL_ATTRIBUTEDECLARATIONCHANGES", "ADDING_ATTRIBUTE_MODIFIABILITY",
                                         "REMOVING_ATTRIBUTE_MODIFIABILITY", "REMOVING_CLASS_DERIVABILITY",
                                         "REMOVING_METHOD_OVERRIDABILITY", "ADDING_CLASS_DERIVABILITY",
                                         "ADDING_CLASS_DERIVABILITY2", "ADDING_METHOD_OVERRIDABILITY",
                                         "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES"]

                df_temp = pd.read_csv(file,
                                      names=chageDistillerMetrics,
                                      dtype={"PROJECT_NAME": "string", "CURRENT_COMMIT": "string",
                                             "PREVIOUS_COMMIT": "string", "CLASS_CURRENTCOMMIT": "string",
                                             "CLASS_PREVIOUSCOMMIT": "string", "STATEMENT_DELETE": int,
                                             "STATEMENT_INSERT": int,
                                             "STATEMENT_ORDERING_CHANGE": int,
                                             "STATEMENT_PARENT_CHANGE": int, "STATEMENT_UPDATE": int,
                                             "TOTAL_STATEMENTLEVELCHANGES": int,
                                             "PARENT_CLASS_CHANGE": int, "PARENT_CLASS_DELETE": int,
                                             "PARENT_CLASS_INSERT": int, "CLASS_RENAMING": int,
                                             "TOTAL_CLASSDECLARATIONCHANGES": int, "RETURN_TYPE_CHANGE": int,
                                             "RETURN_TYPE_DELETE": int,
                                             "RETURN_TYPE_INSERT": int, "METHOD_RENAMING": int,
                                             "PARAMETER_DELETE": int, "PARAMETER_INSERT": int,
                                             "PARAMETER_ORDERING_CHANGE": int, "PARAMETER_RENAMING": int,
                                             "PARAMETER_TYPE_CHANGE": int,
                                             "TOTAL_METHODDECLARATIONSCHANGES": int, "ATTRIBUTE_RENAMING": int,
                                             "ATTRIBUTE_TYPE_CHANGE": int,
                                             "TOTAL_ATTRIBUTEDECLARATIONCHANGES": int,
                                             "ADDING_ATTRIBUTE_MODIFIABILITY": int,
                                             "REMOVING_ATTRIBUTE_MODIFIABILITY": int,
                                             "REMOVING_CLASS_DERIVABILITY": int,
                                             "REMOVING_METHOD_OVERRIDABILITY": int,
                                             "ADDING_CLASS_DERIVABILITY": int,
                                             "ADDING_CLASS_DERIVABILITY2": int,
                                             "ADDING_METHOD_OVERRIDABILITY": int,
                                             "TOTAL_DECLARATIONPARTCHANGES": int, "TOTAL_CHANGES": int
                                             })
            else:
                df_temp = pd.read_csv(file, index_col=False)
            df_append = df_append.append(df_temp)
    # df_append = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df_append.to_csv(file_path + tool + '_all.csv', index=False)
    return df_append


def join_all_metrics():
    print('join all metrics...')

    ck_metrics = ['index', 'file', 'class', 'type', 'cbo_x', 'cboModified_x', 'fanin_x', 'fanout_x', 'wmc_x', 'dit',
                  'noc', 'rfc_x', 'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty', 'staticMethodsQty',
                  'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty',
                  'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty',
                  'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
                  'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'loc_x', 'returnQty',
                  'loopQty_x', 'comparisonsQty_x', 'tryCatchQty_x', 'parenthesizedExpsQty_x', 'stringLiteralsQty_x',
                  'numbersQty_x', 'assignmentsQty_x', 'mathOperationsQty_x', 'variablesQty_x', 'maxNestedBlocksQty_x',
                  'anonymousClassesQty_x', 'innerClassesQty_x', 'lambdasQty_x', 'uniqueWordsQty_x', 'modifiers_x',
                  'logStatementsQty_x', 'method', 'constructor', 'line', 'cbo_y', 'cboModified_y', 'fanin_y',
                  'fanout_y', 'wmc_y', 'rfc_y', 'loc_y', 'returnsQty', 'variablesQty_y', 'parametersQty',
                  'methodsInvokedQty', 'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty', 'loopQty_y',
                  'comparisonsQty_y', 'tryCatchQty_y', 'parenthesizedExpsQty_y', 'stringLiteralsQty_y', 'numbersQty_y',
                  'assignmentsQty_y', 'mathOperationsQty_y', 'maxNestedBlocksQty_y', 'anonymousClassesQty_y',
                  'innerClassesQty_y', 'lambdasQty_y', 'uniqueWordsQty_y', 'modifiers_y', 'logStatementsQty_y',
                  'hasJavaDoc', 'commit_hash', 'project_name']

    ck_values = pd.read_csv('results/ck/ck_all.csv', usecols=ck_metrics, sep=',', index_col=False)
    # print("CK ")
    # print(ck_values.shape[0])

    understand_metrics = ["Kind", "Name", "File", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
                          "AvgEssential", "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "CountClassBase",
                          "CountClassCoupled", "CountClassDerived", "CountDeclClass", "CountDeclClassMethod",
                          "CountDeclClassVariable", "CountDeclFile", "CountDeclFunction", "CountDeclInstanceMethod",
                          "CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodAll",
                          "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected",
                          "CountDeclMethodPublic", "CountInput", "CountLine", "CountLineBlank", "CountLineCode",
                          "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment", "CountOutput", "CountPath",
                          "CountSemicolon", "CountStmt", "CountStmtDecl", "CountStmtExe", "Cyclomatic",
                          "CyclomaticModified", "CyclomaticStrict", "Essential", "MaxCyclomatic",
                          "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxInheritanceTree",
                          "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode", "SumCyclomatic",
                          "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential"]
    und_values = pd.read_csv('results/understand/und_all.csv', usecols=understand_metrics, sep=',',
                             engine='python', index_col=False)

    evo_metrics = ["project", "commit", "commitprevious", "class", "BOC", "TACH", "FCH", "LCH", "CHO", "FRCH",
                   "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF"]
    evo_values = pd.read_csv('results/evometrics/evometrics_all.csv', usecols=evo_metrics, sep=',', index_col=False)

    changedistiller_metrics = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT",
                               "CLASS_PREVIOUSCOMMIT",
                               "STATEMENT_DELETE", "STATEMENT_INSERT", "STATEMENT_ORDERING_CHANGE",
                               "STATEMENT_PARENT_CHANGE", "STATEMENT_UPDATE", "TOTAL_STATEMENTLEVELCHANGES",
                               "PARENT_CLASS_CHANGE", "PARENT_CLASS_DELETE", "PARENT_CLASS_INSERT", "CLASS_RENAMING",
                               "TOTAL_CLASSDECLARATIONCHANGES",
                               "RETURN_TYPE_CHANGE", "RETURN_TYPE_DELETE", "RETURN_TYPE_INSERT", "METHOD_RENAMING",
                               "PARAMETER_DELETE", "PARAMETER_INSERT", "PARAMETER_ORDERING_CHANGE",
                               "PARAMETER_RENAMING",
                               "PARAMETER_TYPE_CHANGE", "TOTAL_METHODDECLARATIONSCHANGES",
                               "ATTRIBUTE_RENAMING", "ATTRIBUTE_TYPE_CHANGE", "TOTAL_ATTRIBUTEDECLARATIONCHANGES",
                               "ADDING_ATTRIBUTE_MODIFIABILITY", "REMOVING_ATTRIBUTE_MODIFIABILITY",
                               "REMOVING_CLASS_DERIVABILITY", "REMOVING_METHOD_OVERRIDABILITY",
                               "ADDING_CLASS_DERIVABILITY", "ADDING_CLASS_DERIVABILITY", "ADDING_METHOD_OVERRIDABILITY",
                               "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES"]

    changedistiller_values = pd.read_csv('results/changedistiller/changedistiller_all.csv',
                                         usecols=changedistiller_metrics, sep=',', index_col=False)

    organic_metrics = ["projectName", "commitNumber", "fullyQualifiedName",
                       "PublicFieldCount", "IsAbstract", "ClassLinesOfCode", "WeighOfClass",
                       "FANIN", "TightClassCohesion", "FANOUT", "OverrideRatio", "LCOM3",
                       "WeightedMethodCount", "LCOM2", "NumberOfAccessorMethods",
                       'LazyClass', 'DataClass', 'ComplexClass', 'SpaghettiCode',
                       'SpeculativeGenerality', 'GodClass', 'RefusedBequest',
                       'ClassDataShouldBePrivate', 'BrainClass', 'TotalClass',
                       'LongParameterList', 'LongMethod', 'FeatureEnvy',
                       'DispersedCoupling', 'MessageChain', 'IntensiveCoupling',
                       'ShotgunSurgery', 'BrainMethod', 'TotalMethod', 'TotalClassMethod',
                       "DiversityTotal", "DiversityMethod", "DiversityClass"]

    organic_values = pd.read_csv('results/organic/organic_all.csv', usecols=organic_metrics, sep=',', index_col=False)

    all_metrics = pd.merge(left=ck_values, right=und_values, left_on='class', right_on='Name')
    all_metrics = pd.merge(left=all_metrics, right=und_values, left_on='class', right_on='Name')


def join_static_features():
    df_ck = pd.read_table('results/ck/ck_all.csv', sep=',')
    df_und = pd.read_table('results/understand/und_all.csv', sep=',')
    df_evo = pd.read_table('results/evometrics/evometrics_all.csv', sep=',')
    df_cd = pd.read_table('results/changedistiller/changedistiller_all.csv', sep=',')
    df_org = pd.read_table('results/organic/organic_all.csv', sep=',')
    df_ref = pd.read_table('results/refactoring/refactoring_all.csv', sep=',')
    # data_frames = [df_ck, df_und, df_evo, df_cd, df_org, df_ref]

    df_all = reduce(lambda left, right: pd.merge(left, right, left_on=['project_name', 'commit_hash', 'class'],
                                                 right_on=['project_name', 'commit_hash', 'File'], how='outer'),
                    [df_ck, df_und])
    df_all.to_csv('results/static_features.csv', sep=',', index=False)


if __name__ == "__main__":
    # projects = ['commons-bcel', 'commons-csv', 'easymock', 'gson', 'Openfire', 'commons-text', 'commons-io', 'pdfbox', 'dubbo']
    # result_df = pd.DataFrame()
    # for project_name in projects:
    # # ck
    # # file, class, commit_hash, project_name
    # path = 'results/ck/' + project_name + '/'
    # parse_ck_results(path, project_name)

    # # understand
    # # Kind, Name, File
    # path = 'results/understand/' + project_name + '/'
    # parse_understand_results(path, project_name)

    # # refactoring miner
    # path = 'results/refactoring/' + project_name + '-results-refactoring-metrics.csv'
    # df = pd.read_csv(path, names=['commit_hash', 'refactoring', 'class_name', 'type', 'source', 'destination'])
    # df = df[['commit_hash', 'refactoring', 'class_name', 'source', 'destination']]
    # df_pivot = pd.pivot_table(df, values=['source', 'destination'],
    #                               index=['commit_hash', 'class_name'],
    #                               columns='refactoring',
    #                               aggfunc=np.sum)
    #
    # df_pivot['projectName'] = project_name
    # df_pivot.reset_index(inplace=True)
    # result_df = result_df.append(df_pivot)

    # # refactoring miner
    # result_df.reset_index()
    # result_df.columns = [' '.join(col).strip() for col in result_df.columns.values]
    # result_df.to_csv('results/refactoring/refactoring_all.csv',
    #                  index=False) #columns=['commit_hash', 'refactoring', 'file',
    #                                                                      #'refactoring_type', 'source', 'destination'],
    #
    # tools = ['changedistiller', 'organic', 'evometrics']
    # for tool in tools:
    #     path = 'results/' + tool + '/'
    #     result_df = merge_csv_files(path, tool)
    #     result_df.to_csv(path + tool + '_all.csv', index=False)

    join_all_metrics()
################################################# ROGERIO
# print(result_df.head())
# #folder with repo: projectA and projectB
# pathCK = args.project_name #"/Volumes/backup-geni/projects-smells/results/ck/junit4/repo/junit4"
# csvPathCK = "results/ck/" + args.project_name + "" #"/Volumes/backup-geni/projects-smells/results/ck/junit4/junit4-all/"
#
# csvPathUndestand = "/Volumes/backup-geni/projects-smells/results/understand/junit4/"
# csvPathProcessMetrics = "/Volumes/backup-geni/projects-smells/results/processMetrics/junit4-results-processMetrics.csv"
# csvPathChangeDistiller = "/Volumes/backup-geni/projects-smells/results/ChangeDistiller/junit4-results.csv"
# csvOrganic =  "/Volumes/backup-geni/projects-smells/results/organic/junit4.csv"
# csvResults = "/Volumes/backup-geni/projects-smells/results/junit4-all-releases.csv"
#
# ckRepo = pydriller.Git(pathCK)
# #understandRepo = pydriller.Git(csvPathUndestand)
# repo = git.Repo(pathCK)
# tags = repo.tags
# release = 1
# #REMOVED FROM CK - "file"
# ckClassMetricsAll = ["class","type","cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty"]
#
# understandMetrics = ["Kind","Name","File","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","CountClassBase","CountClassCoupled","CountClassDerived","CountDeclClass","CountDeclClassMethod","CountDeclClassVariable","CountDeclFile","CountDeclFunction","CountDeclInstanceMethod","CountDeclInstanceVariable","CountDeclMethod","CountDeclMethodAll","CountDeclMethodDefault","CountDeclMethodPrivate","CountDeclMethodProtected","CountDeclMethodPublic","CountInput","CountLine","CountLineBlank","CountLineCode","CountLineCodeDecl","CountLineCodeExe","CountLineComment","CountOutput","CountPath","CountSemicolon","CountStmt","CountStmtDecl","CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential"]
#
# #rename "class" to "className", remove "release"
# processMetrics = ["project","commit","commitprevious","className","BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF"]
#
# chageDistillerMetrics = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT","CLASS_PREVIOUSCOMMIT",
# 	    		"STATEMENT_DELETE", "STATEMENT_INSERT", "STATEMENT_ORDERING_CHANGE","STATEMENT_PARENT_CHANGE","STATEMENT_UPDATE","TOTAL_STATEMENTLEVELCHANGES",
# 	    		"PARENT_CLASS_CHANGE", "PARENT_CLASS_DELETE", "PARENT_CLASS_INSERT","CLASS_RENAMING","TOTAL_CLASSDECLARATIONCHANGES",
# 	    		"RETURN_TYPE_CHANGE","RETURN_TYPE_DELETE","RETURN_TYPE_INSERT","METHOD_RENAMING","PARAMETER_DELETE","PARAMETER_INSERT","PARAMETER_ORDERING_CHANGE","PARAMETER_RENAMING","PARAMETER_TYPE_CHANGE","TOTAL_METHODDECLARATIONSCHANGES",
# 	    		"ATTRIBUTE_RENAMING","ATTRIBUTE_TYPE_CHANGE","TOTAL_ATTRIBUTEDECLARATIONCHANGES",
# 	    		"ADDING_ATTRIBUTE_MODIFIABILITY","REMOVING_ATTRIBUTE_MODIFIABILITY","REMOVING_CLASS_DERIVABILITY","REMOVING_METHOD_OVERRIDABILITY","ADDING_CLASS_DERIVABILITY","ADDING_CLASS_DERIVABILITY","ADDING_METHOD_OVERRIDABILITY", "TOTAL_DECLARATIONPARTCHANGES","TOTAL_CHANGES"]
# organicMetrics = ["projectName","commitNumber","fullyQualifiedName",
#                 "PublicFieldCount","IsAbstract","ClassLinesOfCode","WeighOfClass",
#                 "FANIN","TightClassCohesion","FANOUT","OverrideRatio","LCOM3",
#                 "WeightedMethodCount","LCOM2","NumberOfAccessorMethods",
#                 'LazyClass', 'DataClass', 'ComplexClass', 'SpaghettiCode',
#                 'SpeculativeGenerality', 'GodClass', 'RefusedBequest',
#                 'ClassDataShouldBePrivate', 'BrainClass', 'TotalClass',
#                 'LongParameterList', 'LongMethod', 'FeatureEnvy',
#                     'DispersedCoupling', 'MessageChain', 'IntensiveCoupling',
#                     'ShotgunSurgery', 'BrainMethod', 'TotalMethod', 'TotalClassMethod',
#                      "DiversityTotal","DiversityMethod","DiversityClass"]
# #f = open(csvPath, "w")
# #writer = csv.writer(f)
# missing = []
# for tag in tags:
#     hashCurrent = ckRepo.get_commit_from_tag(tag.name).hash
#
#     try:
#
#
#         releaseUnderstand = pd.read_csv(csvPathUndestand + hashCurrent+'.csv', usecols=understandMetrics, sep=',',engine='python', index_col=False)
#
#         print("Understand ")
#         print(releaseUnderstand.shape[0])
#
#         releaseCK = pd.read_csv(csvPathCK + hashCurrent+'-class.csv', usecols=ckClassMetricsAll, sep=',', index_col=False)
#
#         print("CK ")
#         print(releaseCK.shape[0])
#
#
#         releaseOrganicMetrics = pd.read_csv(csvOrganic, usecols=organicMetrics, sep=',',engine='python', index_col=False)
#         releaseOrganicMetrics = releaseOrganicMetrics[(releaseOrganicMetrics['commitNumber'] == hashCurrent)]
#
#         print("Organic ")
#         print(releaseOrganicMetrics.shape[0])
#
#         releaseChangeDistillerMetrics = pd.read_csv(csvPathChangeDistiller, usecols=chageDistillerMetrics, sep=',',engine='python', index_col=False)
#         releaseChangeDistillerMetrics = releaseChangeDistillerMetrics[(releaseChangeDistillerMetrics['CURRENT_COMMIT'] == hashCurrent)]
#
#
#         print("Change distiller ")
#         print(releaseChangeDistillerMetrics.shape[0])
#
#         releaseProcessMetrics = pd.read_csv(csvPathProcessMetrics, usecols=processMetrics, sep=',', engine='python', index_col=False)
#         releaseProcessMetrics = releaseProcessMetrics[(releaseProcessMetrics['commit'] == hashCurrent)]
#
#         print("Process ")
#         print(releaseProcessMetrics.shape[0])
#
#
#         #para cada release procurar as classes correspondentes e agregar em um só dataframe se "name" = "class"
#         ck_understand = pd.merge(left=releaseCK, right=releaseUnderstand, left_on='class', right_on='Name')
#         ck_understand_process = pd.merge(left=ck_understand, right=releaseProcessMetrics, left_on='class', right_on='className')
#         ck_understand_process_organic = pd.merge(left=ck_understand_process, right=releaseOrganicMetrics, left_on='class', right_on='fullyQualifiedName')
#
#         merged_full = pd.merge(left=ck_understand_process_organic, right=releaseChangeDistillerMetrics, left_on='class', right_on='CLASS_PREVIOUSCOMMIT')
#
#         #merged_full.loc[:,'class_frequency'] = 1
#         merged_full.loc[:,'will_change'] = 0
#         #merged_full.loc[:,'number_of_changes'] = 0
#         merged_full.loc[:,'release'] = release
#         medianChanges = merged_full['TOTAL_CHANGES'].median()
#         merged_full['will_change'] = np.where(merged_full['TOTAL_CHANGES'] > medianChanges, 1,0)
#         if(release == 1):
#             merged_full.to_csv(csvResults, index=False)
#         else:
#              merged_full.to_csv(csvResults,mode="a", header=False, index=False)
#
#         release += 1
#     except Exception as e:
#         print(e)
#        # print(hashCurrent)
#         missing.append(hashCurrent)
#
# print(missing)
