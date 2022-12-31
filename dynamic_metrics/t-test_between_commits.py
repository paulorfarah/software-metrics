import argparse
import collections
import sys

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


def student_ttest_by_method(file, versions):
    #https://analyticsindiamag.com/a-beginners-guide-to-students-t-test-in-python-from-scratch%EF%BF%BC/
    # https://stackoverflow.com/questions/13404468/t-test-in-pandas

    df = read_csv(file)
    df_res = pd.DataFrame(pd.np.empty((0, 10)))
    df_res.columns = ['commit', 'prevcommit', 'class_name', 'method_name', 'metric', 'stat', 'pvalue', 'avg1', 'avg2', 'change']
    for v2 in range(1, len(versions)):
        v1 = v2 - 1
        df1 = df.query("commit_hash == '" + versions[v1] + "'")
        grouped1 = df1[val_cols].groupby(['commit_hash', 'class_name', 'method_name'])

        for metric in val_cols[3:]:
            print("---------------- " + metric + " -------------------------")
            rows = grouped1[metric]

            for name, values1 in rows:
                # print(len(values1))
                # for i in values1:
                #     print(i, type(i))
                vals1 = values1
                try:
                    # print('1->', values1.iloc[0])
                    if type(values1.iloc[0]) is str:
                        vals1 = [eval(i) for i in values1]
                except:
                    print(sys.exc_info())
                    print('values1 empty')
                    print(values1)
                values2 = df.loc[(df['commit_hash'] == versions[v2]) &
                                 (df['class_name'] == name[1]) &
                                 (df['method_name'] == name[2])][metric]

                if values2.any():
                    # print1
                    vals2 = values2
                    try:
                        # print('2->', values2.iloc[0])
                        if type(values2.iloc[0]) is str:
                            vals2 = [eval(i) for i in values2]
                    except:
                        print(sys.exc_info())
                        print('values2 empty')
                        print(values1)
                    try:
                        if isinstance(vals1, collections.abc.Sequence) and isinstance(vals2, collections.abc.Sequence):
                            stat, pvalue = ttest_ind(vals1, vals2)
                        else:
                            stat = -1
                            pvalue = -1
                    except ZeroDivisionError:
                        print('ZeroDivisionError1: ', vals1, vals2)
                        stat = 0
                        pvalue = 0
                    if pvalue <= 0.05:
                        try:
                            avg1 = sum(vals1) / len(vals1)
                        except ZeroDivisionError:
                            print('ZeroDivisionError2: ', vals1)
                            avg1 = 0
                        except:
                            print('error1: ', sys.exc_info())
                            print('vals1: ', len(vals1))
                        try:
                            avg2 = sum(vals2) / len(vals2)
                        except ZeroDivisionError:
                            print('ZeroDivisionError3: ', vals2)
                            avg2 = 0
                        except:
                            print('error2: ', sys.exc_info())
                            print('vals2: ', vals2)

                        try:
                            change = round(((abs(avg2 - avg1) / avg1) * 100), 2)
                        except ZeroDivisionError:
                            print('ZeroDivisionError4: ', avg1)
                            change = 100
                        df_res.loc[len(df_res.index)] = [versions[v1], versions[v2], name[1], name[2], metric, stat,
                                                         pvalue, avg1, avg2, change]
    df_res.to_csv('results/changes.csv', index=False)


def main(commits_file):
    print('starting...')
    commits_list = ['1bd1fd8e6065da9d07b5a3a1723b059246b14001', 'e8f24e86bb2d54493e3f0c0bd7787abb1d1d7443', '1914e7daae2cb39451046e67b993c8ab77e34397']

    # with open(args.commits) as f:
    #     commits_list = f.read().splitlines()
    #     commits_list.reverse()
    #


    # file = '/mnt/sda4/resources.csv'
    # file = '../data/resources.csv'
    file = 'C:\\Users\\paulo\\ufpr\\datasets\\software-metrics\\resources\\resources-csv-1.csv'

    student_ttest_by_method(file, commits_list)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='evaluate resources')
    ap.add_argument('--commits', required=False, help='csv with a list of commits (newest to oldest) to compare commitA and commitB')
    args = ap.parse_args()
    main(args.commits)