import pandas as pd
from matplotlib import pyplot as plt, pyplot
from numpy import mean, std, array
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

pd.set_option('display.max_columns', None)

path1 = '/groups/ilabt-imec-be/software-performance/changedistiller/projectA/commons-bcel/'
path2 = '/groups/ilabt-imec-be/software-performance/changedistiller/projectB/commons-bcel/'

def read_dataset():
    # read dataset
    res_cols = ['commit_hash', 'class_name', 'active', 'available', 'cached ', 'cpu_percent', ' free',
                'inactive', 'mem_percent', 'minor_faults', 'read_count', 'rss', 'used', 'used_percent']
    df_res = pd.read_csv('../data/resources/bcel-resources-avg.csv', sep=';', header=None)
    df_res.columns = res_cols

    cd_cols = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT",
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
    df_cd = pd.read_csv('../data/changedistiller/commons-bcel-results.csv', sep=',')
    df_cd.columns = cd_cols

    df_cd['class_name'] = df_cd['CLASS_PREVIOUSCOMMIT'].str.replace(path1, '')
    df_cd['class_name'] = df_cd['class_name'].str.replace(path2, '')
    df_cd['changed'] = 0
    df_cd.loc[df_cd['TOTAL_CHANGES'] > 0, 'changed'] = 1

    # tracing
    trace_cols = ['committer_date', 'commit_hash', 'class_name', 'num_methods', 'num_calls_sum', 'num_calls_avg',
                  'own_duration_avg_sum', 'own_duration_avg_avg', 'own_duration_avg_min', 'own_duration_avg_max', 'own_duration_avg_stddev']
    df_trace = pd.read_csv('../data/trace/own_dur_trace-2.csv', sep=';', header=None)
    df_trace.columns = trace_cols

    df = pd.merge(left=df_trace, right=df_res, on=['commit_hash', 'class_name'], how='left')
    df = pd.merge(left=df, right=df_cd, left_on=['commit_hash', 'class_name'],
                  right_on=['PREVIOUS_COMMIT', 'class_name'], how='left')
    # df.to_csv('dataset1.csv')
    # df['changed'] = df['changed'].fillna(0)
    # df.drop_duplicates(inplace=True)
    df = df[df['changed'].notna()]
    #
    #f1: 0.932 / 0.925
    #f2: 0 / 0
    # roc2: 49.882 / 50
    # f3: 54.545 / 31.250
    # roc3: 73.135 / 61.06
    print(df[df['changed'].isna()])

    return df


def knearest_neighbor(dataset):
    print('knn')
    data = dataset.values
    # separate into input and output columns
    features = ['active', 'available', 'cached ', 'cpu_percent', ' free',
                'inactive', 'mem_percent', 'minor_faults', 'read_count', 'rss', 'used', 'used_percent']
    X, y = data[:, 3:22], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('int'))
    # define and configure the model
    model = KNeighborsClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report model performance
    print( ' Accuracy: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))
    n_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    # report model performance
    print(' F1: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))

    # precision = metrics.precision_score(y_test, yhat)
    # print('precision: %.3f' % (precision * 100))
    #
    # recall = metrics.recall_score(y_test, yhat)
    # print('recall: %.3f' % (recall * 100))
    #
    # roc_auc = metrics.roc_auc_score(y_test, yhat)
    # print('roc_auc: %.3f' % (roc_auc * 100))

def logistic_regression(dataset):
    print('logistic regression')
    data = dataset.values
    X, y = data[:, 3:22], data[:, -1]

    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('int'))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define the scaler
    scaler = MinMaxScaler()
    # fit on the training dataset
    scaler.fit(X_train)
    # scale the training dataset
    X_train = scaler.transform(X_train)
    # scale the test dataset
    X_test = scaler.transform(X_test)
    # fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % (accuracy*100))

    f1 = metrics.f1_score(y_test, yhat)
    print('f1: %.3f' % (f1 * 100))

    precision = metrics.precision_score(y_test, yhat)
    print('precision: %.3f' % (precision * 100))

    recall = metrics.recall_score(y_test, yhat)
    print('recall: %.3f' % (recall * 100))

    roc_auc = metrics.roc_auc_score(y_test, yhat)
    print('roc_auc: %.3f' % (roc_auc * 100))

def random_forest(dataset):
    print('random forest')
    data = dataset.values
    X, y = data[:, 3:22], data[:, -1]

    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('int'))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define the scaler
    scaler = MinMaxScaler()
    # fit on the training dataset
    scaler.fit(X_train)
    # scale the training dataset
    X_train = scaler.transform(X_train)
    # scale the test dataset
    X_test = scaler.transform(X_test)
    # fit the model
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % (accuracy * 100))

    f1 = metrics.f1_score(y_test, yhat)
    print('f1: %.3f' % (f1 * 100))

    precision = metrics.precision_score(y_test, yhat)
    print('precision: %.3f' % (precision * 100))

    recall = metrics.recall_score(y_test, yhat)
    print('recall: %.3f' % (recall * 100))

    roc_auc = metrics.roc_auc_score(y_test, yhat)
    print('roc_auc: %.3f' % (roc_auc * 100))

def gradient_boosting(dataset):
    print('gradient_boosting')
    data = dataset.values
    X, y = data[:, 3:22], data[:, -1]

    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('int'))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define the scaler
    scaler = MinMaxScaler()
    # fit on the training dataset
    scaler.fit(X_train)
    # scale the training dataset
    X_train = scaler.transform(X_train)
    # scale the test dataset
    X_test = scaler.transform(X_test)
    # fit the model
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % (accuracy * 100))

    f1 = metrics.f1_score(y_test, yhat)
    print('f1: %.3f' % (f1 * 100))

    precision = metrics.precision_score(y_test, yhat)
    print('precision: %.3f' % (precision * 100))

    recall = metrics.recall_score(y_test, yhat)
    print('recall: %.3f' % (recall * 100))

    roc_auc = metrics.roc_auc_score(y_test, yhat)
    print('roc_auc: %.3f' % (roc_auc * 100))


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def main():
    print('starting...')

    df = read_dataset()
    print(df['changed'].unique())
    # print(df.head())
    #normalize
    # scaler = MinMaxScaler()
    # # transform data
    # scaled = scaler.fit_transform(df)

    #standardize

    fig = df.hist(xlabelsize=4, ylabelsize=4)
    [x.title.set_size(4) for x in fig.ravel()]
    plt.show()

    # knearest_neighbor(df)
    # logistic_regression(df)
    # random_forest(df)
    gradient_boosting(df)
if __name__ == "__main__":
    main()