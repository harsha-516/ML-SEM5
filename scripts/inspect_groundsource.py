import pyarrow.parquet as pq
f = pq.ParquetFile("data/raw/groundsource_2026.parquet")
print(f.metadata.num_rows, f.metadata.num_row_groups)
print(f.schema_arrow)
b = next(f.iter_batches(batch_size=5))
print(b.to_pandas().T.to_string()[:2500])
