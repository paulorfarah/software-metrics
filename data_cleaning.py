# summarize the number of unique values for each column using numpy
import pandas as pd
from numpy import loadtxt, percentile, nan
from numpy import unique

# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# load the dataset

ignored_cols = {'ck': ['file', 'class', 'type', 'method', 'constructor', 'commit_hash', 'project_name'],
                'evometrics': ['project', 'commit', 'commitprevious', 'class', 'release'],
                'organic': ['projectName', 'commitNumber' ,'fullyQualifiedName'],
                'refactoring': ['class_name', 'commit_hash', 'projectName'],
                'und': ['Kind', 'Name' ,'File', 'commit_hash', 'project_name'],
                'changedistiller': ['PROJECT_NAME', 'CURRENT_COMMIT', 'PREVIOUS_COMMIT', 'CLASS_CURRENTCOMMIT', 'CLASS_PREVIOUSCOMMIT'],
                'perform': ['class_name', 'method_name', 'commit_hash', 'committer_date_x']}
######################################################
# local = True
# if local:
#     metric = 'ck_100'
#     df = pd.read_csv('results/ck_all_100.csv')
#######################################################
# metrics = ['ck', 'evometrics', 'organic', 'refactoring', 'und', 'changedistiller']
metrics = ['perform']


def format_understand():
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
    df = pd.read_csv('results/und_all.csv', sep=',', engine='python', names=und_metrics, skiprows=1)
    df = df[df.columns[7:]]
    return df

def format_resources_avg():
    res_avg_metrics = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name',
                       'method_started_at', 'method_ended_at', 'methods.caller_id', 'own_duration', 'cumulative_duration',
                       'AVG(active)', 'AVG(available)', 'AVG(buffers)', 'AVG(cached) ', 'AVG(child_major_faults)',
                       'AVG(child_minor_faults)', 'AVG(commit_limit)', 'AVG(committed_as)', 'AVG(cpu_percent)',
                       'AVG(data)', 'AVG(dirty)', 'AVG(free)', 'AVG(high_free)', 'AVG(high_total)', 'AVG(huge_pages_total)',
                       'AVG(huge_pages_free)', 'AVG(huge_pages_total)', 'AVG(hwm)', 'AVG(inactive)', 'AVG(laundry)',
                       'AVG(load1)', 'AVG(load5)', 'AVG(load15)', 'AVG(locked)', 'AVG(low_free)', 'AVG(low_total)',
                       'AVG(major_faults)', 'AVG(mapped)', 'AVG(mem_percent)', 'AVG(minor_faults)', 'AVG(page_tables)',
                       'AVG(pg_fault)', 'AVG(pg_in)', 'AVG(pg_maj_faults)', 'AVG(pg_out)', 'AVG(read_bytes)',
                       'AVG(read_count)', 'AVG(rss)', 'AVG(shared)', 'AVG(sin)', 'AVG(slab)', 'AVG(sout)',
                       'AVG(sreclaimable)', 'AVG(stack)', 'AVG(sunreclaim)', 'AVG(swap)', 'AVG(swap_cached)',
                       'AVG(swap_free)', 'AVG(swap_total)', 'AVG(swap_used)', 'AVG(swap_used_percent) ', 'AVG(total)',
                       'AVG(used)', 'AVG(used_percent)', 'AVG(vm_s)', 'AVG(vmalloc_chunk)', 'AVG(vmalloc_total)',
                       'AVG(vmalloc_used)', 'AVG(wired)', 'AVG(write_back)', 'AVG(write_back_tmp)', 'AVG(write_bytes)',
                       'AVG(write_count)']

    df = pd.read_csv('res_avg.csv', sep=',', names=res_avg_metrics)
    return df

for metric in metrics:
    print('reading dataset: ' + metric)
    if metric == 'und':
        df = format_understand()
    elif metric == 'res_avg':
        df = format_resources_avg()
    else:
        df = pd.read_csv('results/' + metric + '_all.csv')


    # summarize the dataset
    df.describe().to_csv(metric + '_describe.csv')

    # summarize the number of unique values in each column (1%)
    with open("results/variability_" + metric + ".csv", "w") as file1:
        for i in range(df.shape[1]):
            if not df.columns[i] in ignored_cols[metric]:
                num = len(unique(df.iloc[:, i]))
                percentage = float(num) / df.shape[0] * 100
                if percentage < 1.0:
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
