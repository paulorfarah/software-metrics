SELECT methods.id, committer_date, commit_hash, runs.id AS run, files.name AS class_name,
methods.name AS method_name, methods.created_at AS method_started_at, methods.ended_at AS method_ended_at, methods.caller_id, AVG(methods.own_duration) AS own_duration,
AVG(methods.cumulative_duration) AS cumulative_duration,
-- AVG(res.timestamp) ,
AVG(active), AVG(available), AVG(buffers), AVG(cached) , AVG(child_major_faults), AVG(child_minor_faults), AVG(commit_limit), AVG(committed_as),
AVG(cpu_percent), AVG(data), AVG(dirty), AVG(free), AVG(high_free), AVG(high_total), AVG(huge_pages_total), AVG(huge_pages_free),
AVG(hwm), AVG(inactive), AVG(laundry), AVG(load1), AVG(load5), AVG(load15), AVG(locked), AVG(low_free), AVG(low_total), AVG(major_faults), AVG(mapped), AVG(mem_percent),
AVG(minor_faults), AVG(page_tables), AVG(pg_fault), AVG(pg_in), AVG(pg_maj_faults), AVG(pg_out), AVG(read_bytes), AVG(read_count), AVG(rss), AVG(shared), AVG(sin), AVG(slab),
AVG(sout), AVG(sreclaimable), AVG(stack), AVG(sunreclaim), AVG(swap), AVG(swap_cached), AVG(swap_free), AVG(swap_total), AVG(swap_used), AVG(swap_used_percent) ,
AVG(total), AVG(used), AVG(used_percent), AVG(vm_s), AVG(vmalloc_chunk), AVG(vmalloc_total), AVG(vmalloc_used), AVG(wired), AVG(write_back), AVG(write_back_tmp),
AVG(write_bytes), AVG(write_count)
FROM commits
INNER JOIN files ON files.commit_id=commits.id
INNER JOIN methods ON methods.file_id=files.id
INNER JOIN runs ON methods.run_id = runs.id
LEFT JOIN resources AS res ON res.run_id = runs.id
WHERE methods.finished=true
AND res.timestamp BETWEEN methods.created_at AND methods.ended_at
GROUP BY commits.committer_date, commits.commit_hash, files.name, methods.id, runs.id
ORDER BY committer_date, commit_hash, files.name, methods.name
INTO OUTFILE '/var/lib/mysql-files/resources-avg.csv'
FIELDS ENCLOSED BY '"'
TERMINATED BY ';'
ESCAPED BY '"'
LINES TERMINATED BY '\r\n';
