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
        df_all = pd.read_csv('../results/ck/ck_all.csv')  # , dtype={"user_id": int, "username": "string"})

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
        df_all = pd.read_csv('../results/understand/und_all.csv', index_col=False)
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

    ck_values = pd.read_csv('../results/ck/ck_all.csv', usecols=ck_metrics, sep=',', index_col=False)

    # clean file system path
    spl_word = "commons-"
    file_str = ck_values[ck_values['file'].str.contains(spl_word)].iloc[0]
    res = file_str['file'].split(spl_word, 1)
    splitString = res[0]
    ck_values['file'] = ck_values.file.str.replace(splitString, '')

    und_metrics = ["index1","index2","index3","index4","index5","index6","index7","Kind", "Name", "File", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
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
                          "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential", 'unknown', 'commit_hash', 'project_name']
    und_values = pd.read_csv('../results/understand/und_all.csv', sep=',',
                             engine='python', names=und_metrics)#, nrows=5)
    und_values = und_values[und_values.columns[8:]]
    und_values['class'] = und_values['Name']

    # merge ck and understand
    df_all = pd.merge(left=ck_values, right=und_values, left_on=['project_name', 'commit_hash', 'class'],
                      right_on=['project_name', 'commit_hash', 'Name'], how='outer')
    df_all.to_csv('results/static_features_ck_und.csv', sep=',', index=False)


    evo_metrics = ["project", "commit", "commitprevious", "class", "BOC", "TACH", "FCH", "LCH", "CHO", "FRCH",
                   "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF"]
    evo_values = pd.read_csv('../results/evometrics/evometrics_all.csv', usecols=evo_metrics, sep=',', index_col=False)

    #clean file system path
    spl_word = "commons-"
    class_str = evo_values[evo_values['class'].str.contains(spl_word)].iloc[0]
    res = class_str['class'].split(spl_word, 1)
    splitString = res[0]
    evo_values['file'] = evo_values['class'].str.replace(splitString, '')

    # merge ck+und and evometrics
    df_all = pd.merge(left=df_all, right=evo_values, left_on=['project_name', 'commit_hash', 'file'],
                      right_on=['project', 'commit', 'file'], how='outer')
    df_all.to_csv('results/static_features_evo.csv', sep=',', index=False)
    
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

    cd_values = pd.read_csv('../results/changedistiller/changedistiller_all.csv', usecols=changedistiller_metrics,
                            sep=',', index_col=False)

    # merge ck+und+evo and change distiller
    df_all = pd.merge(left=df_all, right=cd_values, left_on=['project_name', 'commit_hash', 'class'],
                      right_on=['PROJECT_NAME', 'PREVIOUS_COMMIT', 'CLASS_PREVIOUSCOMMIT'], how='outer')
    df_all.to_csv('results/static_features_cd.csv', sep=',', index=False)

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

    organic_values = pd.read_csv('../results/organic/organic_all.csv', usecols=organic_metrics, sep=',', index_col=False)

    df_all = pd.merge(left=df_all, right=organic_values, left_on=['project_name', 'commit_hash', 'class'],
                      right_on=['projectName', 'commitNumber', 'fullyQualifiedName'], how='outer')
    df_all.to_csv('results/static_features_org.csv', sep=',', index=False)

    # rm_metrics = ['class_name', 'commit_hash', 'destination Add Attribute Annotation', 'destination Add Attribute Modifier',
    #               'destination Add Class Annotation', 'destination Add Class Modifier', 'destination Add Method Annotation',
    #               'destination Add Method Modifier', 'destination Add Parameter', 'destination Add Parameter Annotation',
    #               'destination Add Parameter Modifier', 'destination Add Thrown Exception Type', 'destination Add Variable Annotation',
    #               'destination Add Variable Modifier', 'destination Change Attribute Access Modifier', 'destination Change Attribute Type',
    #               'destination Change Class Access Modifier', 'destination Change Method Access Modifier',
    #               'destination Change Parameter Type', 'destination Change Return Type', 'destination Change Thrown Exception Type',
    #               'destination Change Type Declaration Kind', 'destination Change Variable Type', 'destination Collapse Hierarchy', 'destination Encapsulate Attribute',
    #               'destination Extract And Move Method', 'destination Extract Attribute', 'destination Extract Class',
    #               'destination Extract Interface', 'destination Extract Method', 'destination Extract Subclass',
    #               'destination Extract Superclass', 'destination Extract Variable', 'destination Inline Attribute',
    #               'destination Inline Method', 'destination Inline Variable', 'destination Localize Parameter',
    #               'destination Merge Attribute', 'destination Merge Class', 'destination Merge Package',
    #               'destination Merge Parameter', 'destination Merge Variable', 'destination Modify Attribute Annotation',
    #               'destination Modify Class Annotation', 'destination Modify Method Annotation',
    #               'destination Modify Variable Annotation', 'destination Move And Inline Method',
    #               'destination Move And Rename Attribute', 'destination Move And Rename Class',
    #               'destination Move And Rename Method', 'destination Move Attribute', 'destination Move Class',
    #               'destination Move Method', 'destination Move Package', 'destination Move Source Folder',
    #               'destination Parameterize Attribute', 'destination Parameterize Variable', 'destination Pull Up Attribute',
    #               'destination Pull Up Method', 'destination Push Down Attribute', 'destination Push Down Method',
    #               'destination Remove Attribute Annotation', 'destination Remove Attribute Modifier',
    #               'destination Remove Class Annotation', 'destination Remove Class Modifier',
    #               'destination Remove Method Annotation', 'destination Remove Method Modifier', 'destination Remove Parameter',
    #               'destination Remove Parameter Annotation', 'destination Remove Parameter Modifier',
    #               'destination Remove Thrown Exception Type', 'destination Remove Variable Annotation',
    #               'destination Remove Variable Modifier', 'destination Rename Attribute', 'destination Rename Class',
    #               'destination Rename Method', 'destination Rename Package', 'destination Rename Parameter',
    #               'destination Rename Variable', 'destination Reorder Parameter', 'destination Replace Anonymous With Lambda',
    #               'destination Replace Attribute', 'destination Replace Attribute With Variable',
    #               'destination Replace Loop With Pipeline', 'destination Replace Pipeline With Loop',
    #               'destination Replace Variable With Attribute', 'destination Split Attribute',
    #               'destination Split Class', 'destination Split Conditional', 'destination Split Package',
    #               'destination Split Parameter', 'destination Split Variable', 'projectName', 'source Add Attribute Annotation',
    #               'source Add Attribute Modifier', 'source Add Class Annotation', 'source Add Class Modifier',
    #               'source Add Method Annotation', 'source Add Method Modifier', 'source Add Parameter',
    #               'source Add Parameter Annotation', 'source Add Parameter Modifier', 'source Add Thrown Exception Type',
    #               'source Add Variable Annotation', 'source Add Variable Modifier', 'source Change Attribute Access Modifier',
    #               'source Change Attribute Type', 'source Change Class Access Modifier', 'source Change Method Access Modifier',
    #               'source Change Parameter Type', 'source Change Return Type', 'source Change Thrown Exception Type',
    #               'source Change Type Declaration Kind', 'source Change Variable Type', 'source Collapse Hierarchy',
    #               'source Encapsulate Attribute', 'source Extract And Move Method', 'source Extract Attribute',
    #               'source Extract Class', 'source Extract Interface', 'source Extract Method', 'source Extract Subclass',
    #               'source Extract Superclass', 'source Extract Variable', 'source Inline Attribute', 'source Inline Method',
    #               'source Inline Variable', 'source Localize Parameter', 'source Merge Attribute', 'source Merge Class',
    #               'source Merge Package', 'source Merge Parameter', 'source Merge Variable', 'source Modify Attribute Annotation',
    #               'source Modify Class Annotation', 'source Modify Method Annotation', 'source Modify Variable Annotation',
    #               'source Move And Inline Method', 'source Move And Rename Attribute', 'source Move And Rename Class',
    #               'source Move And Rename Method', 'source Move Attribute', 'source Move Class', 'source Move Method',
    #               'source Move Package', 'source Move Source Folder', 'source Parameterize Attribute', 'source Parameterize Variable',
    #               'source Pull Up Attribute', 'source Pull Up Method', 'source Push Down Attribute', 'source Push Down Method',
    #               'source Remove Attribute Annotation', 'source Remove Attribute Modifier', 'source Remove Class Annotation',
    #               'source Remove Class Modifier', 'source Remove Method Annotation', 'source Remove Method Modifier',
    #               'source Remove Parameter', 'source Remove Parameter Annotation', 'source Remove Parameter Modifier',
    #               'source Remove Thrown Exception Type', 'source Remove Variable Annotation', 'source Remove Variable Modifier',
    #               'source Rename Attribute', 'source Rename Class', 'source Rename Method', 'source Rename Package',
    #               'source Rename Parameter', 'source Rename Variable', 'source Reorder Parameter', 'source Replace Anonymous With Lambda',
    #               'source Replace Attribute', 'source Replace Attribute With Variable', 'source Replace Loop With Pipeline',
    #               'source Replace Pipeline With Loop', 'source Replace Variable With Attribute', 'source Split Attribute',
    #               'source Split Class', 'source Split Conditional', 'source Split Package', 'source Split Parameter',
    #               'source Split Variable']

    rm_metrics = ['class_name', 'commit_hash', 'destinationAddAttributeAnnotation', 'destinationAddAttributeModifier',
    'destinationAddClassAnnotation', 'destinationAddClassModifier', 'destinationAddMethodAnnotation',
    'destinationAddMethodModifier', 'destinationAddParameter', 'destinationAddParameterAnnotation',
    'destinationAddParameterModifier', 'destinationAddThrownExceptionType', 'destinationAddVariableAnnotation',
    'destinationAddVariableModifier', 'destinationChangeAttributeAccessModifier', 'destinationChangeAttributeType',
    'destinationChangeClassAccessModifier', 'destinationChangeMethodAccessModifier',
    'destinationChangeParameterType', 'destinationChangeReturnType', 'destinationChangeThrownExceptionType',
    'destinationChangeTypeDeclarationKind', 'destinationChangeVariableType', 'destinationCollapseHierarchy', 'destinationEncapsulateAttribute',
    'destinationExtractAndMoveMethod', 'destinationExtractAttribute', 'destinationExtractClass',
    'destinationExtractInterface', 'destinationExtractMethod', 'destinationExtractSubclass',
    'destinationExtractSuperclass', 'destinationExtractVariable', 'destinationInlineAttribute',
    'destinationInlineMethod', 'destinationInlineVariable', 'destinationLocalizeParameter',
    'destinationMergeAttribute', 'destinationMergeClass', 'destinationMergePackage',
    'destinationMergeParameter', 'destinationMergeVariable', 'destinationModifyAttributeAnnotation',
    'destinationModifyClassAnnotation', 'destinationModifyMethodAnnotation',
    'destinationModifyVariableAnnotation', 'destinationMoveAndInlineMethod',
    'destinationMoveAndRenameAttribute', 'destinationMoveAndRenameClass',
    'destinationMoveAndRenameMethod', 'destinationMoveAttribute', 'destinationMoveClass',
    'destinationMoveMethod', 'destinationMovePackage', 'destinationMoveSourceFolder',
    'destinationParameterizeAttribute', 'destinationParameterizeVariable', 'destinationPullUpAttribute',
    'destinationPullUpMethod', 'destinationPushDownAttribute', 'destinationPushDownMethod',
    'destinationRemoveAttributeAnnotation', 'destinationRemoveAttributeModifier',
    'destinationRemoveClassAnnotation', 'destinationRemoveClassModifier',
    'destinationRemoveMethodAnnotation', 'destinationRemoveMethodModifier', 'destinationRemoveParameter',
    'destinationRemoveParameterAnnotation', 'destinationRemoveParameterModifier',
    'destinationRemoveThrownExceptionType', 'destinationRemoveVariableAnnotation',
    'destinationRemoveVariableModifier', 'destinationRenameAttribute', 'destinationRenameClass',
    'destinationRenameMethod', 'destinationRenamePackage', 'destinationRenameParameter',
    'destinationRenameVariable', 'destinationReorderParameter', 'destinationReplaceAnonymousWithLambda',
    'destinationReplaceAttribute', 'destinationReplaceAttributeWithVariable',
    'destinationReplaceLoopWithPipeline', 'destinationReplacePipelineWithLoop',
    'destinationReplaceVariableWithAttribute', 'destinationSplitAttribute',
    'destinationSplitClass', 'destinationSplitConditional', 'destinationSplitPackage',
    'destinationSplitParameter', 'destinationSplitVariable', 'projectName', 'sourceAddAttributeAnnotation',
    'sourceAddAttributeModifier', 'sourceAddClassAnnotation', 'sourceAddClassModifier',
    'sourceAddMethodAnnotation', 'sourceAddMethodModifier', 'sourceAddParameter',
    'sourceAddParameterAnnotation', 'sourceAddParameterModifier', 'sourceAddThrownExceptionType',
    'sourceAddVariableAnnotation', 'sourceAddVariableModifier', 'sourceChangeAttributeAccessModifier',
    'sourceChangeAttributeType', 'sourceChangeClassAccessModifier', 'sourceChangeMethodAccessModifier',
    'sourceChangeParameterType', 'sourceChangeReturnType', 'sourceChangeThrownExceptionType',
    'sourceChangeTypeDeclarationKind', 'sourceChangeVariableType', 'sourceCollapseHierarchy',
    'sourceEncapsulateAttribute', 'sourceExtractAndMoveMethod', 'sourceExtractAttribute',
    'sourceExtractClass', 'sourceExtractInterface', 'sourceExtractMethod', 'sourceExtractSubclass',
    'sourceExtractSuperclass', 'sourceExtractVariable', 'sourceInlineAttribute', 'sourceInlineMethod',
    'sourceInlineVariable', 'sourceLocalizeParameter', 'sourceMergeAttribute', 'sourceMergeClass',
    'sourceMergePackage', 'sourceMergeParameter', 'sourceMergeVariable', 'sourceModifyAttributeAnnotation',
    'sourceModifyClassAnnotation', 'sourceModifyMethodAnnotation', 'sourceModifyVariableAnnotation',
    'sourceMoveAndInlineMethod', 'sourceMoveAndRenameAttribute', 'sourceMoveAndRenameClass',
    'sourceMoveAndRenameMethod', 'sourceMoveAttribute', 'sourceMoveClass', 'sourceMoveMethod',
    'sourceMovePackage', 'sourceMoveSourceFolder', 'sourceParameterizeAttribute', 'sourceParameterizeVariable',
    'sourcePullUpAttribute', 'sourcePullUpMethod', 'sourcePushDownAttribute', 'sourcePushDownMethod',
    'sourceRemoveAttributeAnnotation', 'sourceRemoveAttributeModifier', 'sourceRemoveClassAnnotation',
    'sourceRemoveClassModifier', 'sourceRemoveMethodAnnotation', 'sourceRemoveMethodModifier',
    'sourceRemoveParameter', 'sourceRemoveParameterAnnotation', 'sourceRemoveParameterModifier',
    'sourceRemoveThrownExceptionType', 'sourceRemoveVariableAnnotation', 'sourceRemoveVariableModifier',
    'sourceRenameAttribute', 'sourceRenameClass', 'sourceRenameMethod', 'sourceRenamePackage',
    'sourceRenameParameter', 'sourceRenameVariable', 'sourceReorderParameter', 'sourceReplaceAnonymousWithLambda',
    'sourceReplaceAttribute', 'sourceReplaceAttributeWithVariable', 'sourceReplaceLoopWithPipeline',
    'sourceReplacePipelineWithLoop', 'sourceReplaceVariableWithAttribute', 'sourceSplitAttribute',
    'sourceSplitClass', 'sourceSplitConditional', 'sourceSplitPackage', 'sourceSplitParameter',
    'sourceSplitVariable']

    rm_values = pd.read_csv('../results/refactoring/refactoring_all.csv', names=rm_metrics, sep=',', index_col=False)

    rm_values['file'] = rm_values['projectName'] + '/' + rm_values['class_name']

    df_all = pd.merge(left=df_all, right=rm_values, left_on=['project_name', 'commit_hash', 'class'],
                      right_on=['projectName', 'commit_hash', 'file'], how='outer')

    df_all.to_csv('results/static_features_ref.csv', sep=',', index=False)

# def
# ():
#     print('join static features...')
    # df_ck = pd.read_csv('results/ck/ck_all.csv', sep=',')
    # df_und = pd.read_table('results/understand/und_all.csv', sep=',', index_col=False)
    # df_evo = pd.read_table('results/evometrics/evometrics_all.csv', sep=',')
    # df_cd = pd.read_table('results/changedistiller/changedistiller_all.csv', sep=',')
    # df_org = pd.read_table('results/organic/organic_all.csv', sep=',')
    # df_ref = pd.read_table('results/refactoring/refactoring_all.csv', sep=',')
    # data_frames = [df_ck, df_und, df_evo, df_cd, df_org, df_ref]

    # df_all = reduce(lambda left, right: pd.merge(left, right, left_on=['project_name', 'commit_hash', 'class'],
    #                                              right_on=['project_name', 'commit_hash', 'File'], how='outer'),
    #                 [df_ck, df_und])

    # df_all = pd.merge(left=df_ck, right=df_und, left_on=['project_name', 'commit_hash', 'class'], #
    #                                              right_on=['project_name', 'commit_hash', 'File'], how='outer') #, 'commit_hash'
    # df_all.to_csv('results/static_features.csv', sep=',', index=False)

    # print(df_und.columns)


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
    # join_static_features()
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
