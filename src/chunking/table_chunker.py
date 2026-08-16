"""
Intended to intercept Docling's table structure (which preserves
grid/row/column coordinates) and chunk tables as either markdown
tables or isolated row-level chunks with column headers preserved,
instead of relying on HierarchicalChunker's default row-major
flattening.

Not implemented. Root cause of the Table 5.7.4-1 retrieval failure
(see evaluation writeup) was traced to this gap: HierarchicalChunker's
flattening degrades badly on wide/numeric tables (verified via direct
chunk inspection) even though it happens to survive for small
text-heavy tables. Scoped but deferred due to time constraints.
"""
