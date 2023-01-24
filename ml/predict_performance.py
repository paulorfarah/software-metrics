from collections import Counter

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from numpy import where, mean, std, isnan
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, \
    precision_recall_curve, PrecisionRecallDisplay, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from shap import TreeExplainer
from shap import summary_plot
from xgboost import XGBClassifier

def read_ck(project_name, file):
    ck_metrics_class_and_methods = ['index', 'file', 'class', 'type', 'cbo_x', 'cboModified_x', 'fanin_x', 'fanout_x',
                                    'wmc_x', 'dit',
                                    'noc', 'rfc_x', 'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty',
                                    'staticMethodsQty',
                                    'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty',
                                    'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty',
                                    'synchronizedMethodsQty',
                                    'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty',
                                    'protectedFieldsQty',
                                    'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'loc_x',
                                    'returnQty',
                                    'loopQty_x', 'comparisonsQty_x', 'tryCatchQty_x', 'parenthesizedExpsQty_x',
                                    'stringLiteralsQty_x',
                                    'numbersQty_x', 'assignmentsQty_x', 'mathOperationsQty_x', 'variablesQty_x',
                                    'maxNestedBlocksQty_x',
                                    'anonymousClassesQty_x', 'innerClassesQty_x', 'lambdasQty_x', 'uniqueWordsQty_x',
                                    'modifiers_x',
                                    'logStatementsQty_x', 'method', 'constructor', 'line', 'cbo_y', 'cboModified_y',
                                    'fanin_y',
                                    'fanout_y', 'wmc_y', 'rfc_y', 'loc_y', 'returnsQty', 'variablesQty_y',
                                    'parametersQty',
                                    'methodsInvokedQty', 'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty',
                                    'loopQty_y',
                                    'comparisonsQty_y', 'tryCatchQty_y', 'parenthesizedExpsQty_y',
                                    'stringLiteralsQty_y', 'numbersQty_y',
                                    'assignmentsQty_y', 'mathOperationsQty_y', 'maxNestedBlocksQty_y',
                                    'anonymousClassesQty_y',
                                    'innerClassesQty_y', 'lambdasQty_y', 'uniqueWordsQty_y', 'modifiers_y',
                                    'logStatementsQty_y',
                                    'hasJavaDoc', 'commit_hash', 'project_name']
    # ck_metrics_class = ['file', 'class', 'type', 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc',
    #                     'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty',
    #                     'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty',
    #                     'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty',
    #                     'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
    #                     'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'loc', 'returnQty',
    #                     'loopQty', 'comparisonsQty', 'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty',
    #                     'numbersQty', 'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty',
    #                     'anonymousClassesQty', 'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers',
    #                     'logStatementsQty', 'commit_hash', 'project_name']

    print('comecou a ler o arquivo df_ck')
    df_ck = pd.read_csv(file, usecols=ck_metrics_class_and_methods, sep=',',
                        index_col=False)
    print('leu arquivo df_ck')
    df_ck = df_ck[df_ck['project_name'] == project_name]
    # path1 = '/mnt/sda4/software-metrics/static_metrics/' + project_name + '/'
    path1 = '/groups/ilabt-imec-be/software-performance/ck/' + project_name + '/'

    # df_ck = df_ck.loc[df_ck['project_name'] == project_name]
    # path1 = '/groups/ilabt-imec-be/software-performance/ck/' + project_name + '/'

    df_ck['file'] = df_ck['file'].str.replace(path1, '')
    # try:
    #    spl_word = "commons-"
    #    file_str = df_ck[df_ck['file'].str.contains(spl_word)].iloc[0]
    #    res = file_str['file'].split(spl_word, 1)
    #    splitString = res[0]
    #    df_ck['file'] = df_ck.file.str.replace(splitString, '')
    # except:
    #    pass
    print(df_ck.head())
    return df_ck

def read_own_duration_change(file):
    df_pd = pd.read_csv(file, sep=',')
    df_pd['project_name'] = project_name
    df = pd.DataFrame().assign(project_name=df_pd['project_name'], commit_hash=df_pd['commit'],
                               class_name=df_pd['class_name'], own_duration=df_pd['avg1'], pvalue=df_pd['pvalue'])
    df = df.merge(df_pd[['project_name', 'prevcommit', 'class_name', 'avg2']].rename(
        columns={'prevcommit': 'commit_hash', 'avg2': 'own_duration'}), how='outer')

    df['perf_change'] = np.where(df['pvalue'] < 0.05, 1, 0)
    return df




def read_dataset(project_name, ck_file, trace_file):

    df_ck = read_ck(project_name, ck_file)
    df_target = read_own_duration_change(trace_file)

    df = pd.merge(left=df_ck, right=df_target, left_on=['project_name', 'commit_hash', 'file'],
                  right_on=['project_name', 'commit_hash', 'class_name'], how='inner')

    df = df[df['perf_change'].notna()]
    # df = df.notna()
    return df



def knearest_neighbor(dataset):
    print('knn')
    data = dataset.values
    # separate into input and output columns
    features = ['active', 'available', 'cached ', 'cpu_percent', ' free',
                'inactive', 'mem_percent', 'minor_faults', 'read_count', 'rss', 'used', 'used_percent']
    X, y = data[:, 3:53], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('int'))
    # define and configure the model
    model = KNeighborsClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report model performance
    print(' Accuracy: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))
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


def logistic_regression(X_train, X_test, y_train, y_test):
    print('logistic regression')
    # data = dataset.values
    # X, y = data[:, 3:53], data[:, -1]
    #
    # # ensure inputs are floats and output is an integer label
    # X = X.astype('float32')
    # y = LabelEncoder().fit_transform(y.astype('int'))
    #
    # # split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # # define the scaler
    # scaler = MinMaxScaler()
    # # fit on the training dataset
    # scaler.fit(X_train)
    # # scale the training dataset
    # X_train = scaler.transform(X_train)
    # # scale the test dataset
    # X_test = scaler.transform(X_test)
    # fit the model
    model = LogisticRegression()
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


def random_forest(X, y, cols):
    print('random forest')

    # # evaluate each strategy on the dataset
    # results = list()
    # strategies = ['mean', 'median', 'most_frequent', 'constant']
    # for s in strategies:
    #     # create the modeling pipeline
    #     pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)),
    #                                ("selector", MySelectKBest(f_classif, k=30000)),
    #                                ('m', RandomForestClassifier())])
    #     print(pipeline.steps[2][1].feature_importances_)
    #
    #     # evaluate the model
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     scores = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)
    #     # store results
    #     results.append(scores)
    #     print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
    # # plot model performance for comparison
    # pyplot.boxplot(results, labels=strategies, showmeans=True)
    # pyplot.show()

    # summarize total missing
    print('Missing: %d' % sum(isnan(X).flatten()))
    # define imputer
    # imputer = SimpleImputer(strategy='mean')
    # # fit on the dataset
    # imputer.fit(X)
    # # transform the dataset
    # Xtrans = imputer.transform(X)
    #
    # # transform the dataset
    # oversample = SMOTE()
    # X, y = oversample.fit_resample(Xtrans, y)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # configure to select all features
    # fs = SelectKBest(score_func=f_classif, k='all')
    # # learn relationship from training data
    # fs.fit(X_train, y_train)
    # # transform train input data
    # X_train_fs = fs.transform(X_train)
    # # transform test input data
    # X_test_fs = fs.transform(X_test)
    # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d-%s: %f' % (i, cols[i], fs.scores_[i]))

    # # plot the scores
    # pyplot.figure(figsize=(12, 8))
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_, tick_label=cols)
    # pyplot.xticks(rotation=90)
    # pyplot.subplots_adjust(bottom=0.4)
    # plt.title(project_name + ' - Random Forest')
    # pyplot.show()
    #
    # # lolipop chart of features
    # list1 = fs.scores_
    # list2 = cols
    # list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2), key=lambda pair: pair[0])))
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.hlines(y=list2, xmin=0, xmax=list1)
    # ax.plot(list1, list2, 'o')
    # # adjust chart margin and layout
    # fig.tight_layout()
    # # show chart
    # plt.title(project_name)
    # plt.show()

    # define the scaler
    scaler = MinMaxScaler()
    # fit on the training dataset
    scaler.fit(X_train)
    # scale the training dataset
    X_train = scaler.transform(X_train)
    # scale the test dataset
    # X_test = scaler.transform(X_test)
    # fit the model
    model = RandomForestClassifier(max_depth=10, random_state=0)
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, yhat)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.title(project_name + ' - Random Forest')
    plt.show()

    # roc curve
    # y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, yhat, pos_label=model.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    # precision recall curve
    prec, recall, _ = precision_recall_curve(y_test, yhat, pos_label= model.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.title(project_name + ' - Random Forest')
    plt.show()

    #SHAP
    explainer = TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(X_train))
    # print(shap_values.shape)

    shap_values_ = shap_values.transpose((1, 0, 2))

    np.allclose(
        model.predict_proba(X_train),
        shap_values_.sum(2) + explainer.expected_value
    )
    # print(shap_values)
    summary_plot(shap_values[0], X_train, feature_names=cols)
    summary_plot(shap_values[1], X_train, feature_names=cols)
    plt.title(project_name + ' - Random Forest')
    plt.show()

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


def hist_gradient_boosting(X, y):
    print('gradient_boosting')
    # data = dataset.values
    # X, y = data[:, 3:53], data[:, -1]
    #
    # # ensure inputs are floats and output is an integer label
    # X = X.astype('float32')
    # y = LabelEncoder().fit_transform(y.astype('int'))
    #
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
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

    print(classification_report(yhat, y_test))
    # plot_confusion_matrix(model, X_test, y_test)
    cm = confusion_matrix(y_test, yhat)

    cm_display = ConfusionMatrixDisplay(cm).plot()

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


def xgboost(X, y):
    print('xgboost')
    model = XGBClassifier()
    # define grid
    weights = [1, 10, 25, 50, 75, 99, 100, 1000]
    param_grid = dict(scale_pos_weight=weights)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":

    projects = ['commons-bcel', 'commons-text', 'easymock', 'jgit', 'Openfire']
    # projects = ['Openfire']
    trace_type = 'diff'

    for project_name in projects:
        # trace_file = '../dynamic_metrics/results/' + project_name + '-performance-diff.csv'
        # trace_type = 'diff'
        trace_file = '../dynamic_metrics/results/' + project_name + '-class-performance-' + trace_type + '2.csv'
        ck_file = '../static_metrics/results/ck/' + project_name + 'ck_2.csv'

        df = read_dataset(project_name, ck_file, trace_file)

        # delete duplicate rows
        df.drop_duplicates(inplace=True)

        # # Correlation matrix
        # print(df.columns)
        # ax = sns.heatmap(df[:, [6:53]], linewidth=0.5, cmap='coolwarm')
        # # sns.heatmap(data=df.corr(), annot=True, linewidths=1.5, fmt='.1g', cmap=plt.cm.Reds)
        # plt.show()

        # print(df.head())
        # normalize
        # scaler = MinMaxScaler()
        # # transform data
        # scaled = scaler.fit_transform(df)

        # standardize

        fig = df.hist(xlabelsize=4, ylabelsize=4)
        [x.title.set_size(4) for x in fig.ravel()]
        plt.title(project_name)
        plt.show()
        data = df.values
        X, y = data[:, 4:53], data[:, -1]

        cols = df.columns[4:53]

        # ensure inputs are floats and output is an integer label
        X = X.astype('float32')
        y = LabelEncoder().fit_transform(y.astype('int'))

        # summarize class distribution

        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.title(project_name)
        pyplot.show()

        v_counts = df.perf_change.value_counts()
        fig = plt.figure()
        plt.pie(v_counts, labels=v_counts.index, autopct='%.1f')
        plt.title(project_name)
        plt.show()

        # df.changed.value_counts().plot(kind='pie')
        # pyplot.show()

        # knearest_neighbor(X, y, cols)
        # logistic_regression(X, y, cols)
        random_forest(X, y, cols)
        # hist_gradient_boosting(X, y)
        # xgboost(X, y)


    print('end')