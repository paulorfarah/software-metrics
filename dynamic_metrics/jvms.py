import argparse
import collections
import csv
import sys
from functools import reduce

import pandas as pd
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)

val_cols = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'cumulative_duration', 'anonymous_block_size',
            'anonymous_chunk_size', 'anonymous_class_count', 'block_size', 'chunk_size', 'class_count', 'class_loader',
            'class_loader_data', 'gc_id', 'gc_phase_pause_duration', 'gc_phase_pause_java_name',
            'gc_phase_pause_java_thread_id', 'gc_phase_pause_name', 'gc_phase_pause_os_name',
            'gc_phase_pause_os_thread_id', 'java_error_throw_duration', 'java_error_throw_java_name',
            'java_error_throw_java_thread_id', 'java_error_throw_message', 'java_error_throw_os_name',
            'java_error_throw_os_thread_id', 'java_error_throw_thrown_class', 'java_exception_throw_duration',
            'java_exception_throw_java_name', 'java_exception_throw_java_thread_id', 'java_exception_throw_message',
            'java_exception_throw_os_name', 'java_exception_throw_os_thread_id', 'java_exception_throw_thrown_class',
            'java_monitor_enter_duration', 'java_monitor_enter_java_name', 'java_monitor_enter_java_thread_id',
            'java_monitor_enter_monitor_class', 'java_monitor_enter_os_name', 'java_monitor_enter_os_thread_id',
            'java_monitor_wait_duration', 'java_monitor_wait_java_name', 'java_monitor_wait_java_thread_id',
            'java_monitor_wait_monitor_class', 'java_monitor_wait_os_name', 'java_monitor_wait_os_thread_id',
            'java_monitor_wait_timed_out', 'java_monitor_wait_timeout', 'jvm_system', 'jvm_user', 'loaded_class_count',
            'machine_total', 'object_allocation_in_new_tlab_allocation_size', 'object_allocation_in_new_tlab_java_name',
            'object_allocation_in_new_tlab_java_thread_id', 'object_allocation_in_new_tlab_object_class',
            'object_allocation_in_new_tlab_os_name', 'object_allocation_in_new_tlab_os_thread_id',
            'object_allocation_in_new_tlab_tlab_size', 'object_allocation_outside_tlab_allocation_size',
            'object_allocation_outside_tlab_java_name', 'object_allocation_outside_tlab_java_thread_id',
            'object_allocation_outside_tlab_object_class', 'object_allocation_outside_tlab_os_name',
            'object_allocation_outside_tlab_os_thread_id', 'old_object_sample_allocation_time',
            'old_object_sample_array_elements', 'old_object_sample_duration', 'old_object_sample_java_name',
            'old_object_sample_java_thread_id', 'old_object_sample_last_known_heap_usage', 'old_object_sample_object',
            'old_object_sample_os_name', 'old_object_sample_os_thread_id', 'parent_class_loader', 'thread_cpu_load_java_name',
            'thread_cpu_load_java_thread_id', 'thread_cpu_load_os_name', 'thread_cpu_load_os_thread_id',
            'thread_cpu_load_system', 'thread_cpu_load_user', 'thread_end_java_name', 'thread_end_java_thread_id',
            'thread_end_os_name', 'thread_end_os_thread_id', 'thread_park_duration', 'thread_park_java_name',
            'thread_park_java_thread_id', 'thread_park_os_name', 'thread_park_os_thread_id', 'thread_park_parked_class',
            'thread_park_timeout', 'thread_park_until', 'thread_sleep_duration', 'thread_sleep_java_name',
            'thread_sleep_java_thread_id', 'thread_sleep_os_name', 'thread_sleep_os_thread_id', 'thread_sleep_time',
            'thread_start_java_name', 'thread_start_java_thread_id', 'thread_start_os_name', 'thread_start_os_thread_id',
            'thread_start_parent_thread_java_name', 'thread_start_parent_thread_java_thread_id',
            'thread_start_parent_thread_os_thread_id', 'thread_start_parent_threados_name', 'unloaded_class_count']

def read_csv(file):
    cols = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name', 'method_started_at',
            'method_ended_at', 'caller_id', 'own_duration', 'cumulative_duration', 'anonymous_block_size',
            'anonymous_chunk_size', 'anonymous_class_count', 'block_size', 'chunk_size', 'class_count', 'class_loader',
            'class_loader_data', 'gc_id', 'gc_phase_pause_duration', 'gc_phase_pause_java_name',
            'gc_phase_pause_java_thread_id', 'gc_phase_pause_name', 'gc_phase_pause_os_name',
            'gc_phase_pause_os_thread_id', 'java_error_throw_duration', 'java_error_throw_java_name',
            'java_error_throw_java_thread_id', 'java_error_throw_message', 'java_error_throw_os_name',
            'java_error_throw_os_thread_id', 'java_error_throw_thrown_class', 'java_exception_throw_duration',
            'java_exception_throw_java_name', 'java_exception_throw_java_thread_id', 'java_exception_throw_message',
            'java_exception_throw_os_name', 'java_exception_throw_os_thread_id', 'java_exception_throw_thrown_class',
            'java_monitor_enter_duration', 'java_monitor_enter_java_name', 'java_monitor_enter_java_thread_id',
            'java_monitor_enter_monitor_class', 'java_monitor_enter_os_name', 'java_monitor_enter_os_thread_id',
            'java_monitor_wait_duration', 'java_monitor_wait_java_name', 'java_monitor_wait_java_thread_id',
            'java_monitor_wait_monitor_class', 'java_monitor_wait_os_name', 'java_monitor_wait_os_thread_id',
            'java_monitor_wait_timed_out', 'java_monitor_wait_timeout', 'jvm_system', 'jvm_user', 'loaded_class_count',
            'machine_total', 'object_allocation_in_new_tlab_allocation_size', 'object_allocation_in_new_tlab_java_name',
            'object_allocation_in_new_tlab_java_thread_id', 'object_allocation_in_new_tlab_object_class',
            'object_allocation_in_new_tlab_os_name', 'object_allocation_in_new_tlab_os_thread_id',
            'object_allocation_in_new_tlab_tlab_size', 'object_allocation_outside_tlab_allocation_size',
            'object_allocation_outside_tlab_java_name', 'object_allocation_outside_tlab_java_thread_id',
            'object_allocation_outside_tlab_object_class', 'object_allocation_outside_tlab_os_name',
            'object_allocation_outside_tlab_os_thread_id', 'old_object_sample_allocation_time',
            'old_object_sample_array_elements', 'old_object_sample_duration', 'old_object_sample_java_name',
            'old_object_sample_java_thread_id', 'old_object_sample_last_known_heap_usage', 'old_object_sample_object',
            'old_object_sample_os_name', 'old_object_sample_os_thread_id', 'parent_class_loader', 'thread_cpu_load_java_name',
            'thread_cpu_load_java_thread_id', 'thread_cpu_load_os_name', 'thread_cpu_load_os_thread_id',
            'thread_cpu_load_system', 'thread_cpu_load_user', 'thread_end_java_name', 'thread_end_java_thread_id',
            'thread_end_os_name', 'thread_end_os_thread_id', 'thread_park_duration', 'thread_park_java_name',
            'thread_park_java_thread_id', 'thread_park_os_name', 'thread_park_os_thread_id', 'thread_park_parked_class',
            'thread_park_timeout', 'thread_park_until', 'thread_sleep_duration', 'thread_sleep_java_name',
            'thread_sleep_java_thread_id', 'thread_sleep_os_name', 'thread_sleep_os_thread_id', 'thread_sleep_time',
            'thread_start_java_name', 'thread_start_java_thread_id', 'thread_start_os_name', 'thread_start_os_thread_id',
            'thread_start_parent_thread_java_name', 'thread_start_parent_thread_java_thread_id',
            'thread_start_parent_thread_os_thread_id', 'thread_start_parent_threados_name', 'unloaded_class_count']

    df = pd.read_csv(file, names=cols, sep=';', header=None)
    return df

def split_commits(file, commit_list, delimiter=';', quotechar='"'):
    #create files
    f = []
    for i in range(len(commit_list)):
        f.append(open('results/' + commit_list[i] + "_jvm.csv", "w"))
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




# for l,el in enumerate(stats):
#         string = ', '.join(map(str,el))
#         for item in string:
#             f.write(item)
#     f.write('\n')

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
            row = grouped1[metric]

            for name, values1 in row:
                values2 = df.loc[(df['commit_hash'] == versions[v2]) &
                                 (df['class_name'] == name[1]) &
                                 (df['method_name'] == name[2])][metric]

                if values2.any():
                    print('values2: ', values2)
                    try:
                        if isinstance(values1, collections.abc.Sequence) and isinstance(values2, collections.abc.Sequence):
                            stat, pvalue = ttest_ind(values1, values2)
                        else:
                            stat = -1
                            pvalue = -1
                    except ZeroDivisionError:
                        stat = 0
                        pvalue = 0
                    if pvalue <= 0.05:
                        try:
                            avg1 = sum(values1) / len(values1)
                        except ZeroDivisionError:
                            avg1 = 0
                        except:
                            print('values1: ', values1)
                            print('error: ', sys.exc_info())
                        try:
                            avg2 = sum(values2) / len(values2)
                        except ZeroDivisionError:
                            avg2 = 0
                        except:
                            print('values2: ', values2)
                            print('error: ', sys.exc_info())
                        try:
                            change = round(((abs(avg2 - avg1) / avg1) * 100), 2)
                        except ZeroDivisionError:
                            change = 100
                        df_res.loc[len(df_res.index)] = [versions[v1], versions[v2], name[1], name[2], metric, stat,
                                                         pvalue, avg1, avg2, change]
    df_res.to_csv('results/changes_jvm.csv', index=False)


# def compare_versions(v1, v2):
#     f1 = 'results/' + v1 + ".csv"
#     df1 = read_csv(f1)
#     f2 = 'results/' + v2 + ".csv"
#     df2 = read_csv(f2)



def main(commits_file):
    print('starting...')

    # geni
    # file = '/mnt/sda4/resources.csv'
    # commits_list = []
    # with open(args.commits) as f:
    #     commits_list = f.read().splitlines()
    #     commits_list.reverse()

    # local
    file = 'data/jvms.csv'
    commits_list = ['f38847e90714fbefc33042912d1282cc4fb7d43e', 'f38847e90714fbefc33042912d1282cc4fb7d43f']

    if commits_list:
        split_commits(file, commits_list)
        for commit_hash in commits_list:
            i = commits_list.index(commit_hash)
            f = 'results/' + commit_hash + "_jvm.csv"
            df = read_csv(f)
            stat_analysis(df, 'results/' + commit_hash + '_jvm_stats.csv')
    else:
        df = read_csv(file)
        stat_analysis(df, 'results/jvm_stats.csv')
    print('finished jvm analysis.')

    student_ttest_by_method(file, commits_list)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='evaluate resources')
    ap.add_argument('--commits', required=False, help='csv with a list of commits (newer to oldest) to compare commitA and commitB')
    args = ap.parse_args()
    main(args.commits)
