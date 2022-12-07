SELECT methods.id, committer_date, commit_hash, runs.id AS run, files.name AS class_name,
methods.name AS method_name, methods.created_at AS method_started_at, methods.ended_at AS method_ended_at, methods.caller_id, AVG(methods.own_duration) AS own_duration,
AVG(methods.cumulative_duration) AS cumulative_duration,
AVG(anonymous_block_size), AVG(anonymous_chunk_size), AVG(anonymous_class_count), AVG(block_size), AVG(chunk_size), AVG(class_count), AVG(class_loader), AVG(class_loader_data), AVG(gc_id), AVG(gc_phase_pause_duration), AVG(gc_phase_pause_java_name), AVG(gc_phase_pause_java_thread_id), AVG(gc_phase_pause_name), AVG(gc_phase_pause_os_name), AVG(gc_phase_pause_os_thread_id), AVG(java_error_throw_duration), AVG(java_error_throw_java_name), AVG(java_error_throw_java_thread_id), AVG(java_error_throw_message), AVG(java_error_throw_os_name), AVG(java_error_throw_os_thread_id), AVG(java_error_throw_thrown_class), AVG(java_exception_throw_duration), AVG(java_exception_throw_java_name), AVG(java_exception_throw_java_thread_id), AVG(java_exception_throw_message), AVG(java_exception_throw_os_name), AVG(java_exception_throw_os_thread_id), AVG(java_exception_throw_thrown_class), AVG(java_monitor_enter_duration), AVG(java_monitor_enter_java_name), AVG(java_monitor_enter_java_thread_id), AVG(java_monitor_enter_monitor_class), AVG(java_monitor_enter_os_name), AVG(java_monitor_enter_os_thread_id), AVG(java_monitor_wait_duration), AVG(java_monitor_wait_java_name), AVG(java_monitor_wait_java_thread_id), AVG(java_monitor_wait_monitor_class), AVG(java_monitor_wait_os_name), AVG(java_monitor_wait_os_thread_id), AVG(java_monitor_wait_timed_out), AVG(java_monitor_wait_timeout), AVG(jvm_system), AVG(jvm_user), AVG(loaded_class_count), AVG(machine_total), AVG(object_allocation_in_new_tlab_allocation_size), AVG(object_allocation_in_new_tlab_java_name), AVG(object_allocation_in_new_tlab_java_thread_id), AVG(object_allocation_in_new_tlab_object_class), AVG(object_allocation_in_new_tlab_os_name), AVG(object_allocation_in_new_tlab_os_thread_id), AVG(object_allocation_in_new_tlab_tlab_size), AVG(object_allocation_outside_tlab_allocation_size), AVG(object_allocation_outside_tlab_java_name), AVG(object_allocation_outside_tlab_java_thread_id), AVG(object_allocation_outside_tlab_object_class), AVG(object_allocation_outside_tlab_os_name), AVG(object_allocation_outside_tlab_os_thread_id), AVG(old_object_sample_allocation_time), AVG(old_object_sample_array_elements), AVG(old_object_sample_duration), AVG(old_object_sample_java_name), AVG(old_object_sample_java_thread_id), AVG(old_object_sample_last_known_heap_usage), AVG(old_object_sample_object), AVG(old_object_sample_os_name), AVG(old_object_sample_os_thread_id), AVG(parent_class_loader), AVG(thread_cpu_load_java_name), AVG(thread_cpu_load_java_thread_id), AVG(thread_cpu_load_os_name), AVG(thread_cpu_load_os_thread_id), AVG(thread_cpu_load_system), AVG(thread_cpu_load_user), AVG(thread_end_java_name), AVG(thread_end_java_thread_id), AVG(thread_end_os_name), AVG(thread_end_os_thread_id), AVG(thread_park_duration), AVG(thread_park_java_name), AVG(thread_park_java_thread_id), AVG(thread_park_os_name), AVG(thread_park_os_thread_id), AVG(thread_park_parked_class), AVG(thread_park_timeout), AVG(thread_park_until), AVG(thread_sleep_duration), AVG(thread_sleep_java_name), AVG(thread_sleep_java_thread_id), AVG(thread_sleep_os_name), AVG(thread_sleep_os_thread_id), AVG(thread_sleep_time), AVG(thread_start_java_name), AVG(thread_start_java_thread_id), AVG(thread_start_os_name), AVG(thread_start_os_thread_id), AVG(thread_start_parent_thread_java_name), AVG(thread_start_parent_thread_java_thread_id), AVG(thread_start_parent_thread_os_thread_id), AVG(thread_start_parent_threados_name), AVG(unloaded_class_count)
FROM commits
INNER JOIN files ON files.commit_id=commits.id
INNER JOIN methods ON methods.file_id=files.id
INNER JOIN runs ON methods.run_id = runs.id
LEFT JOIN jvms AS jvm ON jvm.run_id = runs.id
WHERE methods.finished=true
AND jvm.start_time BETWEEN methods.created_at AND methods.ended_at
GROUP BY commits.committer_date, commits.commit_hash, files.name, methods.id, runs.id
ORDER BY committer_date, commit_hash, files.name, methods.name
INTO OUTFILE '/var/lib/mysql-files/jvms-1.csv'
FIELDS ENCLOSED BY '"'
TERMINATED BY ';'
ESCAPED BY '"'
LINES TERMINATED BY '\r\n';