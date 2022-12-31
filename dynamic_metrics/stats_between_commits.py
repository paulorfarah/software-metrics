import argparse
import collections
import csv
import sys
from functools import reduce

import pandas as pd
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)

val_cols = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'cumulative_duration', 'active',
            'available', 'buffers', 'cached ', 'child_major_faults', 'child_minor_faults', 'commit_limit',
            'committed_as', 'cpu_percent', 'data', 'dirty', 'free', 'high_free', 'high_total', 'huge_pages_total',
            'huge_pages_free', 'huge_pages_total1', 'hwm', 'inactive', 'laundry', 'load1', 'load5', 'load15',
            'locked', 'low_free', 'low_total', 'major_faults', 'mapped', 'mem_percent', 'minor_faults',
            'page_tables', 'pg_fault', 'pg_in', 'pg_maj_faults', 'pg_out', 'read_bytes', 'read_count', 'rss',
            'shared', 'sin', 'slab', 'sout', 'sreclaimable', 'stack', 'sunreclaim', 'swap', 'swap_cached',
            'swap_free', 'swap_total', 'swap_used', 'swap_used_percent ', 'total', 'used', 'used_percent', 'vm_s',
            'vmalloc_chunk', 'vmalloc_total', 'vmalloc_used', 'wired', 'write_back', 'write_back_tmp',
            'write_bytes', 'write_count']

def read_csv(file):
    cols = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name',
            'method_name', 'method_started_at', 'method_ended_at',
            'caller_id', 'own_duration', 'cumulative_duration', 'timestamp ',
            'active', 'available', 'buffers', 'cached ', 'child_major_faults', 'child_minor_faults', 'commit_limit',
            'committed_as', 'cpu_percent', 'data', 'dirty', 'free', 'high_free', 'high_total', 'huge_pages_total',
            'huge_pages_free',
            'huge_pages_total1', 'hwm', 'inactive', 'laundry', 'load1', 'load5', 'load15', 'locked', 'low_free',
            'low_total', 'major_faults',
            'mapped', 'mem_percent', 'minor_faults', 'page_tables', 'pg_fault', 'pg_in', 'pg_maj_faults', 'pg_out',
            'read_bytes', 'read_count',
            'rss', 'shared', 'sin', 'slab', 'sout', 'sreclaimable', 'stack', 'sunreclaim', 'swap', 'swap_cached',
            'swap_free', 'swap_total', 'swap_used',
            'swap_used_percent ', 'total', 'used', 'used_percent', 'vm_s', 'vmalloc_chunk', 'vmalloc_total',
            'vmalloc_used', 'wired',
            'write_back', 'write_back_tmp', 'write_bytes', 'write_count']
    df = pd.read_csv(file, names=cols, sep=';', header=None)
    return df

def split_commits(file, commit_list, delimiter=';', quotechar='"'):
    #create files
    f = []
    for i in range(len(commit_list)):
        f.append(open('results/' + commit_list[i] + ".csv", "w"))
    with open(file, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        for r in datareader:
            commit_hash = r[2]
            i = commit_list.index(commit_hash)
            if r:
                # f[i].write("\"")
                string = '; '.join(r)
                for item in string:
                    f[i].write(item)
                    f[i].flush()
                # f[i].write("\"")
                f[i].write("\n")
            else:
                print('not r')

    for i in range(len(commit_list)):
        f[i].close()


def stat_analysis(df, output):
    df_merged = df.groupby(['commit_hash', 'class_name', 'method_name'])[val_cols].describe().reset_index()
    df_merged.columns = [f'{i}_{j}' for i, j in df_merged.columns]
    #
    # df_count = df[val_cols].groupby(['commit_hash', 'class_name', 'method_name']).count()
    # df_mean = df[val_cols].groupby(['commit_hash', 'class_name', 'method_name']).mean()
    # df_median = df[val_cols].groupby(['commit_hash', 'class_name', 'method_name']).median()
    # df_std = df[val_cols].groupby(['commit_hash', 'class_name', 'method_name']).std()
    # # result = pd.merge([df_mean, df_count, df_median, df_std], how="inner", on=['commit_hash', 'class_name', 'method_name'])
    # # data_frames = [df_mean, df_count, df_median, df_std]
    # # df_merged = reduce(lambda left, right: pd.merge(left, right, on=['commit_hash', 'class_name', 'method_name'],
    # #                                                 how='outer', suffixes=), data_frames)
    # df_merged = pd.merge(df_count, df_mean, on=['commit_hash', 'class_name', 'method_name'],
    #                      how='outer', suffixes=('_count', '_mean'))
    # df_merged = pd.merge(df_merged, df_median, on=['commit_hash', 'class_name', 'method_name'],
    #                      how='outer', suffixes=('', '_median'))
    # df_merged = pd.merge(df_merged, df_std, on=['commit_hash', 'class_name', 'method_name'],
    #                      how='outer', suffixes=('', '_std'))

    pd.DataFrame.to_csv(df_merged, output, sep=',', index=False)

def stat_between_commits(commits_list, file):
    if commits_list:
        split_commits(file, commits_list)
        for commit_hash in commits_list:
            i = commits_list.index(commit_hash)
            f = 'results/' + commit_hash + ".csv"
            df = read_csv(f)
            stat_analysis(df, 'results/' + commit_hash + '_merged.csv')
    else:
        df = read_csv(file)
        stat_analysis(df, 'results/merged.csv')
    print('finished analysis.')

def main(commits_file):
    print('starting...')
    commits_list = ['1bd1fd8e6065da9d07b5a3a1723b059246b14001', 'e8f24e86bb2d54493e3f0c0bd7787abb1d1d7443', '1914e7daae2cb39451046e67b993c8ab77e34397']

    # with open(args.commits) as f:
    #     commits_list = f.read().splitlines()
    #     commits_list.reverse()
    #
    # commits_list = ['f38847e90714fbefc33042912d1282cc4fb7d43e', 'f38847e90714fbefc33042912d1282cc4fb7d43f']

    # file = '/mnt/sda4/resources.csv'
    # file = '../data/resources.csv'
    file = 'C:\\Users\\paulo\\ufpr\\datasets\\software-metrics\\resources\\resources-csv-1.csv'
    stat_between_commits(commits_list, file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='evaluate resources')
    ap.add_argument('--commits', required=False, help='csv with a list of commits (newest to oldest) to compare commitA and commitB')
    args = ap.parse_args()
    main(args.commits)
