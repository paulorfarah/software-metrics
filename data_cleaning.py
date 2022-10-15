# summarize the number of unique values for each column using numpy
import pandas as pd
from numpy import loadtxt, percentile, nan
from numpy import unique
# load the dataset
df = pd.read_csv('results/ck_all.csv')

# summarize the dataset
df.describe().to_csv('ck_describe.csv')

# summarize the number of unique values in each column (1%)
print(df.shape[1])
# Writing to file
with open("unique.csv", "w") as file1:
    # Writing data to a file
    # file1.write("Hello \n")
    # file1.writelines(L)


    for i in range(df[:, 'cbo_x':'logStatementsQty_x'].shape[1]):
        num = len(unique(df.iloc[:, i]))
        percentage = float(num) / df.shape[0] * 100
        if percentage < 1:
            # print('%d, %d, %.1f%%' % (i, num, percentage))
            file1.write('%d, %d, %.1f%%' % (i, num, percentage))

#
# # removing columns with low variance
# # get number of unique values for each column
# counts = df.nunique()
# # record columns to delete
# to_del = [i for i, v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
# print(to_del)
# # drop useless columns
# df.drop(to_del, axis=1, inplace=True)
# df.to_csv('ck_drop.csv')
#
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
