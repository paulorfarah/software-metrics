import pandas as pd
from numpy import unique

ignored_cols = {'resources': ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name',
                              'method_started_at', 'method_ended_at', 'caller_id', 'timestamp '],
                }

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
    cols_type = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name',
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

def main():
    print('starting...')
    metric = 'resources'
    f = '../data/resources.csv'
    df = read_csv(f)
    with open("results/variability_" + metric + ".csv", "w") as file1:
        for i in range(df.shape[1]):
            if not df.columns[i] in ignored_cols[metric]:
                num = len(unique(df.iloc[:, i]))
                percentage = float(num) / df.shape[0] * 100
                # if percentage < 1.0:
                file1.write('%s, %d, %.1f%%\n' % (df.columns[i], num, percentage))


if __name__ == "__main__":
    main()
