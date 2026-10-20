# Retrieval evaluation

> **Small evaluation set:** 30 questions over 2 document(s) (ผู้จัดการสาขา, ผู้จัดการเขต). One question is 0.03 of Recall/MRR. Differences of one or two questions are noise: do not read them as a winner. The set was written by hand for these manuals, so it measures this corpus, not Thai retrieval in general.

Recall@k and MRR count answerable questions; *threshold* is the top score below which the answer is "not found" (best balanced accuracy on answerable vs no-answer questions); *false answer* is the share of no-answer questions still answered at that threshold.

## All configurations (best first)

| configuration | chunks | R@1 | R@3 | R@5 | MRR | role acc | threshold | false answer | answerable refused | ms/query | embed s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WangchanX-Legal-ThaiCCL-Retriever · 400 · keep · header | 25 | 0.69 | 0.96 | 0.96 | 0.81 | 0.92 | 0.58 | 0.00 | 0.04 | 116 | 5.9 |
| bge-m3 · 400 · keep · header | 25 | 0.69 | 0.96 | 0.96 | 0.79 | 0.96 | 0.56 | 0.00 | 0.04 | 126 | 3.8 |
| WangchanX-Legal-ThaiCCL-Retriever · 600 · flatten · header | 19 | 0.65 | 0.92 | 1.00 | 0.79 | 0.80 | 0.55 | 0.00 | 0.04 | 123 | 5.1 |
| WangchanX-Legal-ThaiCCL-Retriever · 600 · keep · header | 19 | 0.62 | 0.92 | 1.00 | 0.78 | 0.84 | 0.55 | 0.00 | 0.04 | 120 | 6.0 |
| bge-m3 · 600 · keep · header | 19 | 0.65 | 0.92 | 0.96 | 0.77 | 0.88 | 0.56 | 0.00 | 0.04 | 108 | 3.3 |
| bge-m3 · 600 · flatten · header | 19 | 0.65 | 0.96 | 0.96 | 0.76 | 0.84 | 0.56 | 0.00 | 0.04 | 109 | 3.3 |
| WangchanX-Legal-ThaiCCL-Retriever · 400 · flatten · header | 25 | 0.58 | 0.92 | 1.00 | 0.75 | 0.84 | 0.58 | 0.00 | 0.04 | 121 | 5.1 |
| bge-m3 · 600 · keep · no header | 19 | 0.62 | 0.85 | 0.96 | 0.73 | 0.76 | 0.57 | 0.00 | 0.08 | 108 | 3.3 |
| WangchanX-Legal-ThaiCCL-Retriever · 400 · flatten · no header | 24 | 0.62 | 0.81 | 0.92 | 0.73 | 0.92 | 0.57 | 0.00 | 0.04 | 122 | 4.7 |
| bge-m3 · 400 · flatten · header | 25 | 0.58 | 0.92 | 0.92 | 0.72 | 0.92 | 0.58 | 0.00 | 0.08 | 127 | 3.8 |
| e5-small · 600 · flatten · header | 19 | 0.62 | 0.85 | 0.92 | 0.72 | 0.88 | 0.86 | 0.00 | 0.00 | 14 | 0.4 |
| WangchanX-Legal-ThaiCCL-Retriever · 600 · keep · no header | 19 | 0.58 | 0.88 | 0.96 | 0.72 | 0.76 | 0.55 | 0.00 | 0.04 | 122 | 5.4 |
| bge-m3 · 600 · flatten · no header | 20 | 0.62 | 0.85 | 0.92 | 0.71 | 0.84 | 0.57 | 0.00 | 0.08 | 108 | 3.2 |
| e5-small · 600 · keep · header | 19 | 0.58 | 0.88 | 0.88 | 0.71 | 0.80 | 0.86 | 0.00 | 0.00 | 13 | 0.4 |
| bge-m3 · 200 · keep · header | 31 | 0.54 | 0.92 | 0.96 | 0.71 | 0.76 | 0.56 | 0.00 | 0.04 | 257 | 7.7 |
| WangchanX-Legal-ThaiCCL-Retriever · 200 · keep · header | 31 | 0.50 | 0.96 | 0.96 | 0.71 | 0.80 | 0.56 | 0.00 | 0.04 | 120 | 9.5 |
| WangchanX-Legal-ThaiCCL-Retriever · 400 · keep · no header | 24 | 0.58 | 0.85 | 0.88 | 0.70 | 0.92 | 0.57 | 0.00 | 0.04 | 121 | 5.9 |
| WangchanX-Legal-ThaiCCL-Retriever · 600 · flatten · no header | 20 | 0.58 | 0.88 | 0.92 | 0.70 | 0.84 | 0.55 | 0.00 | 0.04 | 119 | 5.3 |
| bge-m3 · 400 · keep · no header | 24 | 0.54 | 0.77 | 0.92 | 0.69 | 0.84 | 0.62 | 0.00 | 0.08 | 127 | 3.8 |
| bge-m3 · 200 · flatten · header | 31 | 0.50 | 0.92 | 0.92 | 0.69 | 0.76 | 0.56 | 0.00 | 0.04 | 126 | 3.8 |
| WangchanX-Legal-ThaiCCL-Retriever · 200 · flatten · header | 31 | 0.46 | 0.96 | 0.96 | 0.69 | 0.80 | 0.56 | 0.00 | 0.04 | 117 | 4.4 |
| e5-small · 400 · keep · no header | 24 | 0.58 | 0.77 | 0.85 | 0.68 | 0.76 | 0.87 | 0.00 | 0.04 | 14 | 0.4 |
| bge-m3 · 400 · flatten · no header | 24 | 0.54 | 0.85 | 0.88 | 0.68 | 0.84 | 0.60 | 0.00 | 0.08 | 112 | 3.4 |
| e5-small · 400 · keep · header | 25 | 0.54 | 0.85 | 0.88 | 0.67 | 0.88 | 0.87 | 0.00 | 0.04 | 12 | 0.4 |
| e5-base · 400 · flatten · header | 25 | 0.54 | 0.77 | 0.92 | 0.67 | 0.92 | 0.84 | 0.00 | 0.23 | 37 | 1.1 |
| e5-small · 600 · flatten · no header | 19 | 0.58 | 0.77 | 0.81 | 0.67 | 0.72 | 0.87 | 0.00 | 0.04 | 14 | 0.4 |
| e5-base · 400 · keep · header | 25 | 0.54 | 0.81 | 0.88 | 0.66 | 0.92 | 0.84 | 0.00 | 0.23 | 41 | 1.3 |
| e5-base · 600 · keep · header | 19 | 0.54 | 0.81 | 0.88 | 0.66 | 0.80 | 0.84 | 0.00 | 0.23 | 35 | 1.1 |
| e5-small · 600 · keep · no header | 19 | 0.50 | 0.81 | 0.88 | 0.66 | 0.72 | 0.87 | 0.00 | 0.04 | 13 | 0.4 |
| e5-base · 600 · flatten · header | 19 | 0.50 | 0.85 | 0.88 | 0.65 | 0.68 | 0.84 | 0.00 | 0.23 | 32 | 1.0 |
| e5-base · 200 · flatten · header | 31 | 0.42 | 0.85 | 1.00 | 0.64 | 0.76 | 0.83 | 0.00 | 0.12 | 41 | 1.3 |
| e5-small · 400 · flatten · header | 25 | 0.50 | 0.77 | 0.85 | 0.63 | 0.80 | 0.86 | 0.00 | 0.04 | 14 | 0.4 |
| e5-base · 600 · flatten · no header | 19 | 0.46 | 0.77 | 0.88 | 0.63 | 0.72 | 0.84 | 0.00 | 0.15 | 33 | 1.0 |
| e5-small · 200 · flatten · no header | 31 | 0.46 | 0.81 | 0.85 | 0.63 | 0.64 | 0.86 | 0.00 | 0.04 | 14 | 0.4 |
| e5-base · 200 · flatten · no header | 31 | 0.42 | 0.81 | 0.92 | 0.63 | 0.56 | 0.84 | 0.00 | 0.19 | 36 | 1.1 |
| e5-base · 200 · keep · no header | 31 | 0.46 | 0.81 | 0.88 | 0.63 | 0.72 | 0.84 | 0.00 | 0.15 | 34 | 1.0 |
| e5-small · 200 · keep · no header | 31 | 0.50 | 0.81 | 0.81 | 0.62 | 0.76 | 0.87 | 0.00 | 0.04 | 14 | 0.4 |
| e5-base · 200 · keep · header | 31 | 0.38 | 0.92 | 0.92 | 0.62 | 0.84 | 0.83 | 0.00 | 0.08 | 105 | 3.2 |
| e5-small · 400 · flatten · no header | 24 | 0.50 | 0.65 | 0.81 | 0.61 | 0.72 | 0.87 | 0.00 | 0.04 | 14 | 0.4 |
| WangchanX-Legal-ThaiCCL-Retriever · 200 · keep · no header | 31 | 0.46 | 0.73 | 0.88 | 0.61 | 0.84 | 0.57 | 0.00 | 0.04 | 121 | 5.9 |
| e5-small · 200 · keep · header | 31 | 0.42 | 0.77 | 0.92 | 0.61 | 0.76 | 0.86 | 0.00 | 0.04 | 75 | 2.3 |
| e5-base · 400 · flatten · no header | 24 | 0.42 | 0.73 | 0.88 | 0.61 | 0.68 | 0.84 | 0.00 | 0.19 | 40 | 1.2 |
| e5-base · 400 · keep · no header | 24 | 0.50 | 0.65 | 0.77 | 0.60 | 0.72 | 0.84 | 0.00 | 0.19 | 42 | 1.3 |
| e5-base · 600 · keep · no header | 19 | 0.46 | 0.73 | 0.77 | 0.59 | 0.64 | 0.84 | 0.00 | 0.19 | 33 | 1.0 |
| WangchanX-Legal-ThaiCCL-Retriever · 200 · flatten · no header | 31 | 0.42 | 0.77 | 0.88 | 0.59 | 0.80 | 0.57 | 0.00 | 0.04 | 121 | 4.5 |
| e5-small · 200 · flatten · header | 31 | 0.38 | 0.81 | 0.92 | 0.59 | 0.84 | 0.86 | 0.00 | 0.04 | 13 | 0.4 |
| bge-m3 · 200 · keep · no header | 31 | 0.35 | 0.81 | 0.88 | 0.57 | 0.84 | 0.61 | 0.00 | 0.08 | 125 | 3.8 |
| bge-m3 · 200 · flatten · no header | 31 | 0.31 | 0.85 | 0.88 | 0.55 | 0.88 | 0.61 | 0.00 | 0.08 | 126 | 3.8 |
| fake-hash-3gram-64 · 600 · keep · header | 20 | 0.38 | 0.62 | 0.65 | 0.50 | 0.64 | 0.36 | 0.00 | 0.15 | 0 | 0.0 |
| fake-hash-3gram-64 · 600 · keep · no header | 18 | 0.35 | 0.58 | 0.62 | 0.46 | 0.68 | 0.37 | 0.00 | 0.19 | 0 | 0.0 |
| fake-hash-3gram-64 · 200 · flatten · header | 36 | 0.31 | 0.58 | 0.65 | 0.44 | 0.68 | 0.24 | 0.25 | 0.04 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 400 · flatten · header | 36 | 0.31 | 0.54 | 0.65 | 0.42 | 0.76 | 0.28 | 0.00 | 0.15 | 34 | 1.0 |
| simcse-model-m-bert-thai-cased · 200 · keep · no header | 48 | 0.23 | 0.50 | 0.73 | 0.41 | 0.64 | 0.34 | 0.00 | 0.27 | 33 | 1.0 |
| fake-hash-3gram-64 · 600 · flatten · header | 20 | 0.31 | 0.50 | 0.58 | 0.41 | 0.72 | 0.36 | 0.00 | 0.15 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 200 · keep · header | 55 | 0.31 | 0.42 | 0.62 | 0.40 | 0.56 | 0.29 | 0.00 | 0.15 | 54 | 1.6 |
| simcse-model-m-bert-thai-cased · 200 · flatten · no header | 52 | 0.23 | 0.50 | 0.69 | 0.40 | 0.60 | 0.34 | 0.00 | 0.27 | 36 | 1.1 |
| simcse-model-m-bert-thai-cased · 200 · flatten · header | 56 | 0.31 | 0.38 | 0.62 | 0.40 | 0.60 | 0.29 | 0.00 | 0.15 | 32 | 1.0 |
| simcse-model-m-bert-thai-cased · 600 · flatten · header | 28 | 0.31 | 0.54 | 0.54 | 0.40 | 0.64 | 0.26 | 0.25 | 0.19 | 33 | 1.0 |
| fake-hash-3gram-64 · 200 · keep · header | 36 | 0.23 | 0.58 | 0.69 | 0.40 | 0.60 | 0.12 | 0.25 | 0.00 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 400 · keep · header | 36 | 0.27 | 0.50 | 0.65 | 0.39 | 0.68 | 0.28 | 0.00 | 0.15 | 33 | 1.0 |
| fake-hash-3gram-64 · 400 · flatten · header | 26 | 0.27 | 0.50 | 0.54 | 0.38 | 0.64 | 0.17 | 0.25 | 0.04 | 0 | 0.0 |
| fake-hash-3gram-64 · 200 · flatten · no header | 35 | 0.23 | 0.50 | 0.58 | 0.37 | 0.52 | 0.22 | 0.25 | 0.04 | 0 | 0.0 |
| fake-hash-3gram-64 · 600 · flatten · no header | 19 | 0.27 | 0.46 | 0.54 | 0.37 | 0.72 | 0.37 | 0.00 | 0.19 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 600 · keep · header | 27 | 0.31 | 0.42 | 0.46 | 0.36 | 0.60 | 0.26 | 0.25 | 0.19 | 36 | 1.1 |
| fake-hash-3gram-64 · 400 · keep · header | 25 | 0.27 | 0.38 | 0.46 | 0.34 | 0.80 | 0.17 | 0.25 | 0.04 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 400 · flatten · no header | 35 | 0.19 | 0.38 | 0.50 | 0.32 | 0.64 | 0.30 | 0.25 | 0.15 | 32 | 1.0 |
| fake-hash-3gram-64 · 400 · flatten · no header | 26 | 0.19 | 0.46 | 0.50 | 0.31 | 0.60 | 0.27 | 0.25 | 0.08 | 0 | 0.0 |
| fake-hash-3gram-64 · 200 · keep · no header | 34 | 0.19 | 0.46 | 0.50 | 0.31 | 0.68 | 0.22 | 0.25 | 0.04 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 400 · keep · no header | 35 | 0.15 | 0.42 | 0.54 | 0.31 | 0.64 | 0.30 | 0.25 | 0.15 | 38 | 1.2 |
| fake-hash-3gram-64 · 400 · keep · no header | 25 | 0.23 | 0.35 | 0.46 | 0.30 | 0.76 | 0.19 | 0.25 | 0.04 | 0 | 0.0 |
| simcse-model-m-bert-thai-cased · 600 · keep · no header | 24 | 0.19 | 0.31 | 0.42 | 0.28 | 0.60 | 0.28 | 0.00 | 0.27 | 37 | 1.1 |
| simcse-model-m-bert-thai-cased · 600 · flatten · no header | 26 | 0.15 | 0.35 | 0.50 | 0.28 | 0.48 | 0.28 | 0.00 | 0.27 | 36 | 1.1 |

## Best configuration per question type

| type | questions | configuration | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|---|
| cross_role | 6 | WangchanX-Legal-ThaiCCL-Retriever · 400 · keep · header | 0.67 | 1.00 | 1.00 | 0.83 |
| screenshot_only | 6 | bge-m3 · 400 · keep · header | 1.00 | 1.00 | 1.00 | 1.00 |
| text | 14 | bge-m3 · 400 · keep · header | 0.64 | 1.00 | 1.00 | 0.79 |

## No-answer questions: best "not found" threshold

| configuration | threshold | balanced accuracy | false answer | answerable refused |
|---|---|---|---|---|
| e5-small · 600 · keep · header | 0.86 | 1.00 | 0.00 | 0.00 |

## Per question: WangchanX-Legal-ThaiCCL-Retriever · 400 · keep · header

| id | type | first relevant rank | top score | top role |
|---|---|---|---|---|
| q01 | text | 1 | 0.712 | branch_manager |
| q02 | text | - | 0.637 | district_manager |
| q03 | text | 1 | 0.626 | branch_manager |
| q04 | text | 1 | 0.722 | branch_manager |
| q05 | text | 1 | 0.703 | branch_manager |
| q06 | text | 3 | 0.691 | branch_manager |
| q07 | text | 1 | 0.671 | branch_manager |
| q08 | text | 2 | 0.599 | branch_manager |
| q09 | text | 2 | 0.605 | branch_manager |
| q10 | text | 1 | 0.647 | branch_manager |
| q11 | text | 1 | 0.714 | district_manager |
| q12 | text | 3 | 0.699 | district_manager |
| q13 | text | 1 | 0.658 | district_manager |
| q14 | text | 1 | 0.673 | district_manager |
| q15 | screenshot_only | 1 | 0.596 | branch_manager |
| q16 | screenshot_only | 2 | 0.474 | district_manager |
| q17 | screenshot_only | 1 | 0.575 | branch_manager |
| q18 | screenshot_only | 1 | 0.708 | branch_manager |
| q19 | screenshot_only | 1 | 0.576 | district_manager |
| q20 | screenshot_only | 1 | 0.596 | district_manager |
| q21 | cross_role | 1 | 0.670 | branch_manager |
| q22 | cross_role | 1 | 0.680 | district_manager |
| q23 | cross_role | 2 | 0.680 | branch_manager |
| q24 | cross_role | 2 | 0.669 | district_manager |
| q25 | cross_role | 1 | 0.668 | branch_manager |
| q26 | cross_role | 1 | 0.649 | district_manager |
| q27 | no_answer | - | 0.491 | branch_manager |
| q28 | no_answer | - | 0.483 | branch_manager |
| q29 | no_answer | - | 0.428 | branch_manager |
| q30 | no_answer | - | 0.517 | district_manager |
