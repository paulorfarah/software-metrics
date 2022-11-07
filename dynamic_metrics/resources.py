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


def stat_analysis(df):

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



    pd.DataFrame.to_csv(df_merged, 'merged.csv', sep=',', index=False)


def main():
    file = 'data/resources.csv'
    df = read_csv(file)
    stat_analysis(df)


if __name__ == "__main__":
    main()
