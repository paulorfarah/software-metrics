# summarize the number of unique values for each column using numpy
import pandas as pd
from numpy import loadtxt, percentile, nan
from numpy import unique
# load the dataset

ignored_cols = {'ck': ['file', 'class', 'type', 'method', 'constructor', 'commit_hash', 'project_name'],
                'evometrics': ['project', 'commit', 'commitprevious', 'class', 'release'],
                'organic': ['projectName', 'commitNumber' ,'fullyQualifiedName'],
                'refactoring': ['class_name', 'commit_hash', 'projectName'],
                'und': ['Kind', 'Name' ,'File', 'commit_hash', 'project_name'],
                'changedistiller': ['PROJECT_NAME', 'CURRENT_COMMIT', 'PREVIOUS_COMMIT', 'CLASS_CURRENTCOMMIT', 'CLASS_PREVIOUSCOMMIT']}
######################################################
# local = True
# if local:
#     metric = 'ck_100'
#     df = pd.read_csv('results/ck_all_100.csv')
#######################################################
# metrics = ['ck', 'evometrics', 'organic', 'refactoring', 'und', 'changedistiller']
metrics = ['refactoring']
for metric in metrics:

    if not metric == 'und':
        df = pd.read_csv('results/' + metric + '_all.csv')
    else:
        und_metrics = ["index1", "index2", "index3", "index4", "index5", "index6", "index7", "Kind", "Name", "File",
                       "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
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
                       "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential", 'unknown', 'commit_hash',
                       'project_name']
        df = pd.read_csv('results/und_all.csv', sep=',', engine='python', names=und_metrics)
        df = df[df.columns[8:]]

    # summarize the dataset
    df.describe().to_csv(metric + '_describe.csv')

    # summarize the number of unique values in each column (1%)
    # print(df.shape[1])
    # Writing to file
    with open("results/variability_" + metric + ".csv", "w") as file1:
        # Writing data to a file
        # file1.write("Hello \n")
        # file1.writelines(L)

        # df_class = df[:, 'cbo_x':'logStatementsQty_x']
        # print(df_class.head())
        for i in range(df.shape[1]):
            if not df.columns[i] in ignored_cols[metric]:
                num = len(unique(df.iloc[:, i]))
                percentage = float(num) / df.shape[0] * 100
                if percentage < 1.0:
                    print('%s, %d, %.2f%%' % (df.columns[i], num, percentage))
                    file1.write('%s, %d, %.1f%%\n' % (df.columns[i], num, percentage))



    # removing columns with low variance
    # get number of unique values for each column
    counts = df.nunique()
    # record columns to delete
    to_del = [df.columns[i] for i, v in enumerate(counts) if ((float(v)/df.shape[0]*100) < 1) and (df.columns[i] not in ignored_cols[metric])]

    with open("results/dropped_cols_" + metric + ".csv", "w") as file1:
        for col in to_del:
            file1.write(col + '\n')
    # drop useless columns
    df.drop(to_del, axis=1, inplace=True)
    df.to_csv('results/' + metric + '_clean.csv')

# def evaluate_features_threshold():
#     # split data into inputs and outputs
#     data = df.values
#     X = data[:, :-1]
#     y = data[:, -1]
#     print(X.shape, y.shape)
#     # define thresholds to check
#     thresholds = arange(0.0, 0.55, 0.05)
#     # apply transform with each threshold
#     results = list()
#     for t in thresholds:
#     # define the transform
#     transform = VarianceThreshold(threshold=t)
#     # transform the input data
#     X_sel = transform.fit_transform(X)
#     # determine the number of input features
#     n_features = X_sel.shape[1]
#     print('>Threshold=%.2f, Features=%d' % (t, n_features))
#     # store the result
#     results.append(n_features)
#     # plot the threshold vs the number of selected features
#     pyplot.plot(thresholds, results)
#     pyplot.show()
#
# def delete_duplicate_rows(df):
#     print(df.shape)
#     # delete duplicate rows
#     df.drop_duplicates(inplace=True)
#     print(df.shape)
#
# def iqr_outliers(data):
#     # calculate interquartile range
#     q25, q75 = percentile(data, 25), percentile(data, 75)
#     iqr = q75 - q25
#     print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
#     # calculate the outlier cutoff
#     cut_off = iqr * 1.5
#     lower, upper = q25 - cut_off, q75 + cut_off
#     # identify outliers
#     outliers = [x for x in data if x < lower or x > upper]
#     print('Identified outliers: %d' % len(outliers))
#     # remove outliers
#     outliers_removed = [x for x in data if x >= lower and x <= upper]
#     print('Non-outlier observations: %d' % len(outliers_removed))
#
#
# def count_missing_data(df, cols):
#     num_missing = (df[cols] == 0).sum()
#     # report the results
#     print(num_missing)
#
# def identify_missing_data(df, cols):
#     # replace '0' values with 'nan'
#     df[cols] = df[cols].replace(0, nan)
#     # count the number of nan values in each column
#     print(df.isnull().sum())
#     return df

print('end')