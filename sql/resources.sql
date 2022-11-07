SELECT methods.id, committer_date, commit_hash, runs.id AS run, files.name AS class_name,
methods.name AS method_name, methods.created_at AS method_started_at, methods.ended_at AS method_ended_at, methods.caller_id,
methods.own_duration AS own_duration,
methods.cumulative_duration AS cumulative_duration,
res.timestamp ,
active, available, buffers, cached , child_major_faults, child_minor_faults, commit_limit, committed_as,
cpu_percent, data, dirty, free, high_free, high_total, huge_pages_total, huge_pages_free, huge_pages_total,
hwm, inactive, laundry, load1, load5, load15, locked, low_free, low_total, major_faults, mapped, mem_percent,
minor_faults, page_tables, pg_fault, pg_in, pg_maj_faults, pg_out, read_bytes, read_count, rss, shared, sin, slab,
sout, sreclaimable, stack, sunreclaim, swap, swap_cached, swap_free, swap_total, swap_used, swap_used_percent ,
total, used, used_percent, vm_s, vmalloc_chunk, vmalloc_total, vmalloc_used, wired, write_back, write_back_tmp,
write_bytes, write_count
FROM commits
INNER JOIN files ON files.commit_id=commits.id
INNER JOIN methods ON methods.file_id=files.id
INNER JOIN runs ON methods.run_id = runs.id
LEFT JOIN resources AS res ON res.run_id = runs.id
WHERE methods.finished=true
AND res.timestamp BETWEEN methods.created_at AND methods.ended_at
ORDER BY committer_date, commit_hash, files.name, methods.name
INTO OUTFILE '/mnt/sda4/mysql-files/resources.csv'
FIELDS ENCLOSED BY '"'
TERMINATED BY ';'
ESCAPED BY '"'
LINES TERMINATED BY '\r\n';