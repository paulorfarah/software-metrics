import csv
from functools import reduce

import pandas as pd

pd.set_option('display.max_columns', None)


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
        f.append(open(str(i) + ".csv", "w"))
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
                # f[i].write("\"")
                f[i].write("\n")

    for i in range(len(commit_list)):
        f[i].close()




# for l,el in enumerate(stats):
#         string = ', '.join(map(str,el))
#         for item in string:
#             f.write(item)
#     f.write('\n')

def stat_analysis(df, output):

    val_cols = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'cumulative_duration', 'active',
                'available', 'buffers', 'cached ', 'child_major_faults', 'child_minor_faults', 'commit_limit',
                'committed_as', 'cpu_percent', 'data', 'dirty', 'free', 'high_free', 'high_total', 'huge_pages_total',
                'huge_pages_free', 'huge_pages_total1', 'hwm', 'inactive', 'laundry', 'load1', 'load5', 'load15',
                'locked', 'low_free','low_total', 'major_faults', 'mapped', 'mem_percent', 'minor_faults',
                'page_tables', 'pg_fault', 'pg_in', 'pg_maj_faults', 'pg_out', 'read_bytes', 'read_count', 'rss',
                'shared', 'sin', 'slab', 'sout', 'sreclaimable', 'stack', 'sunreclaim', 'swap', 'swap_cached',
                'swap_free', 'swap_total', 'swap_used', 'swap_used_percent ', 'total', 'used', 'used_percent', 'vm_s',
                'vmalloc_chunk', 'vmalloc_total', 'vmalloc_used', 'wired', 'write_back', 'write_back_tmp',
                'write_bytes', 'write_count']
    df_merged = df.groupby(['commit_hash', 'class_name', 'method_name'])[val_cols].describe().reset_index()
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


def main():
    project_name = 'easymock'
    # commits_list = None
    commits_list = ['1a01c13b73c0c66de1efa3db4d73a839aaf20ab9', '266c64660523d728592e646fa9f3f3e2fdfdbc4a',
                    'caf80a128f00481e8c19151257001015acc3e76e', 'a6e7c7e6fc54c8ee3dce10edbe76c1821f10cd92',
                    '0c45595df8f8a0939dbc0b0385c8afe7502b1190', '853c1e35326a54e3fc28177c5c84c07652750140',
                    '3506ccdfa91500016e3a0908d7ccabc171aa5602', '22ade6817ad07f22f1d8f0263ff6ddc6fc9b05db',
                    '36782213bf5e8f1e0f601cb73774ec7a5a8c58f1']

    print('starting...')
    file = '/mnt/sda4/resources.csv'

    if commits_list:
        split_commits(file, commits_list)
        for commit_hash in commits_list:
            i = commits_list.index(commit_hash)
            f = str(i) + '.csv'
            print(str(f))
            df = read_csv(f)
            print(df.head())
            print('read csv file...')
            stat_analysis(df, commit_hash + '_merged.csv')
            print('finished analysis.')
    else:
        df = read_csv(file)
        print('read csv file...')
        # stat_analysis(df, 'merged.csv')
        print('finished analysis.')





if __name__ == "__main__":
    main()
