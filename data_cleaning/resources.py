import pandas as pd
from numpy import unique

ignored_cols = {'resources': ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name',
                              'method_started_at', 'method_ended_at', 'caller_id', 'timestamp '],
                }

def read_csv(file):
    cols = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name',
            'method_name', 'method_started_at', 'method_ended_at',
            'caller_id', 'own_duration', 'cumulative_duration', 'timestamp', 'active', 'available', 'buffers', 'cached ',
            'child_major_faults', 'child_minor_faults', 'commit_limit',
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
    cols_type = {'methods.id':int, 'committer_date':str, 'commit_hash': str, 'run': int, 'class_name':str,
            'method_name':str, 'method_started_at':str, 'method_ended_at':str,
            'caller_id':str, 'own_duration':float, 'cumulative_duration':float, 'timestamp':str, 'active':float, 'available':float, 'buffers':float,
            'cached': float, 'child_major_faults':float, 'child_minor_faults':float, 'commit_limit':float,
            'committed_as': float, 'cpu_percent':float, 'data':float, 'dirty':float, 'free':float, 'high_free':float, 'high_total':float, 'huge_pages_total':float,
            'huge_pages_free':float,
            'huge_pages_total1':float, 'hwm':float, 'inactive':float, 'laundry':float, 'load1':float, 'load5':float, 'load15':float,
                 'locked':float, 'low_free':float,
            'low_total':float, 'major_faults':float,
            'mapped':float, 'mem_percent':float, 'minor_faults':float, 'page_tables':float, 'pg_fault':float, 'pg_in':float,
            'pg_maj_faults':float, 'pg_out': float,
            'read_bytes':float, 'read_count':float,
            'rss':float, 'shared':float, 'sin':float, 'slab':float, 'sout':float, 'sreclaimable':float, 'stack':float, 'sunreclaim':float,
            'swap': float, 'swap_cached':float, 'swap_free':float, 'swap_total':float, 'swap_used':float,
            'swap_used_percent':float, 'total':float, 'used':float, 'used_percent':float, 'vm_s':float, 'vmalloc_chunk':float,
            'vmalloc_total': float, 'vmalloc_used':float, 'wired':float, 'write_back':float, 'write_back_tmp':float,
            'write_bytes':float, 'write_count':float}
    df = pd.read_csv(file, names=cols, sep=';', dtype=cols_type)
    return df

def main():
    print('starting...')
    metric = 'resources'
    f = '/mnt/sda4/resources-csv-1-header.csv'
    df = read_csv(f)
    print(df.head())
    for v in df['own_duration'].values:
        if type(v) == 'str':
            print(v)
    with open("results/variability_" + metric + ".csv", "w") as file1:
        for i in range(df.shape[1]):
            if not df.columns[i] in ignored_cols[metric]:
                print('cols: ' + df.columns[i])
                num = len(unique(df.iloc[:, i]))
                percentage = float(num) / df.shape[0] * 100
                # if percentage < 1.0:
                file1.write('%s, %d, %.1f%%\n' % (df.columns[i], num, percentage))


if __name__ == "__main__":
    main()
