import pandas as pd
from numpy import unique

ignored_cols = {'resources': ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name',
                              'method_started_at', 'method_ended_at', 'caller_id', 'timestamp '],
                }

def read_csv(file):
    cols = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name',
            'method_name', 'method_started_at', 'method_ended_at',
            'caller_id', 'own_duration', 'cumulative_duration', 'active', 'available', 'buffers', 'cached ',
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
            'caller_id':int, 'own_duration':float, 'cumulative_duration':float, 'active':int, 'available':int, 'buffers':int,
                'cached': int, 'child_major_faults':int, 'child_minor_faults':int, 'commit_limit':int,
            'committed_as': int, 'cpu_percent':float, 'data':int, 'dirty':int, 'free':int, 'high_free':int, 'high_total':int, 'huge_pages_total':int,
            'huge_pages_free':int,
            'huge_pages_total1':int, 'hwm':int, 'inactive':int, 'laundry':int, 'load1':float, 'load5':float, 'load15':float,
                 'locked':int, 'low_free':int,
            'low_total':int, 'major_faults':int,
            'mapped':int, 'mem_percent':float, 'minor_faults':int, 'page_tables':int, 'pg_fault':int, 'pg_in':int,
            'pg_maj_faults':int, 'pg_out': int,
            'read_bytes':int, 'read_count':int,
            'rss':int, 'shared':int, 'sin':int, 'slab':int, 'sout':int, 'sreclaimable':int, 'stack':int, 'sunreclaim':int,
            'swap': int, 'swap_cached':int, 'swap_free':int, 'swap_total':int, 'swap_used':int,
            'swap_used_percent':float, 'total':int, 'used':int, 'used_percent':float, 'vm_s':int, 'vmalloc_chunk':int,
            'vmalloc_total': int, 'vmalloc_used':int, 'wired':int, 'write_back':int, 'write_back_tmp':int,
            'write_bytes':int, 'write_count':int}
    df = pd.read_csv(file, names=cols, sep=';', header=None, dtype=cols_type)
    return df

def main():
    print('starting...')
    metric = 'resources'
    f = '/mnt/sda4/resources-csv-1-header.csv'
    df = read_csv(f)
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
