import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

pd.set_option('display.max_columns', None)

path1 = '/groups/ilabt-imec-be/software-performance/changedistiller/projectA/commons-bcel/'
path2 = '/groups/ilabt-imec-be/software-performance/changedistiller/projectB/commons-bcel/'

def read_dataset():
    # read dataset
    res_cols = ['commit_hash', 'class_name', 'active', 'available', 'cached ', 'cpu_percent', ' free',
                'inactive', 'mem_percent', 'minor_faults', 'read_count', 'rss', 'used', 'used_percent']
    df_res = pd.read_csv('../data/bcel-resources-avg.csv', sep=';', header=None)
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
    df_cd = pd.read_csv('../data/commons-bcel-change_distiller.csv', sep=',')
    df_cd.columns = cd_cols

    df_cd['class_name'] = df_cd['CLASS_PREVIOUSCOMMIT'].str.replace(path1, '')
    df_cd['class_name'] = df_cd['class_name'].str.replace(path2, '')
    df_cd['changed'] = 0
    df_cd.loc[df_cd['TOTAL_CHANGES'] >0 , 'changed'] = 1

    df = pd.merge(left=df_res, right=df_cd, left_on=['commit_hash', 'class_name'],
                  right_on=['PREVIOUS_COMMIT', 'class_name'], how='left')
    # df.to_csv('dataset1.csv')

    return df


def knearest_neighbor(dataset):
    data = dataset.values
    # separate into input and output columns
    features = ['active', 'available', 'cached ', 'cpu_percent', ' free',
                'inactive', 'mem_percent', 'minor_faults', 'read_count', 'rss', 'used', 'used_percent']
    X, y = data[:, 2:13], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))
    # define and configure the model
    model = KNeighborsClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # # report model performance
    # print( ' Accuracy: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))
    n_scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
    # report model performance
    print(' F1: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))

def main():
    print('starting...')

    df = read_dataset()
    print(df.head())
    #normalize
    # scaler = MinMaxScaler()
    # # transform data
    # scaled = scaler.fit_transform(df)

    #standardize

    fig = df.hist(xlabelsize=4, ylabelsize=4)
    [x.title.set_size(4) for x in fig.ravel()]
    plt.show()

    knearest_neighbor(df)

if __name__ == "__main__":
    main()
