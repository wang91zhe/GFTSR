# GFTSR
A GCN-based table structure recognition method, which integrates position feature, and image feature together.

 
# DataSet SciTSR
SciTSR is a large-scale table structure recognition dataset, which contains 15,000 tables in PDF format and their corresponding structure labels obtained from LaTeX source files.


**Note that our pre-processed chunk and relation data may contain noise. The original input files are in PDF.**

### Text Chunks

File: chunk/[ID].chunk

The `pos` array contains the `x1`, `x2`, `y1` and `y2` coordinates (in PDF) of the chunk.

```json
{"chunks": [
  {
    "pos": [
      147.96600341796875,
      205.49998474121094,
      475.7929992675781,
      480.4206237792969
    ],
    "text": "Probability"
  },
  {
    "pos": [
      217.45510864257812,
      290.6802673339844,
      475.7929992675781,
      480.4206237792969
    ],
    "text": "Generated Text"
  },
  ...
 ]}
```

### Relations

File rel/[ID].rel

A line of `CHUNK_ID_1 CHUNK_ID_2 RELATION_ID:NUM_BLANK` represents the relation between CHUNK_ID_1-th chunk and CHUNK_ID_2-th chunk is RELATION_ID, and there are NUM_BLANK blank cells between them.
For RELATION_ID, 1 and 2 represents horizontal and vertical, respectively.

```
0 1 1:0
1 2 1:0
0 9 2:0
...
```

### Structure Labels

File: structure/[ID].json

A table is stored as a list of cells. For each cell, we provide its original tex code, content (split by space) and position in the table (start/end row/column number, started from 0).

```json
{"cells": [
  {
    "id": 21,
    "tex": "959",
    "content": [
      "959"
    ],
    "start_row": 5,
    "end_row": 5,
    "start_col": 1,
    "end_col": 1
  },
  {
    "id": 1,
    "tex": "Training set",
    "content": [
      "Training",
      "set"
    ],
    "start_row": 0,
    "end_row": 0,
    "start_col": 1,
    "end_col": 1
  },
  ...
]}
```

# reference：
https://github.com/wang91zhe/GFTE
