# Retrieval evaluation

> **Small evaluation set:** 30 questions over 2 document(s) (ผู้จัดการสาขา, ผู้จัดการเขต). One question is 0.03 of Recall/MRR. Differences of one or two questions are noise: do not read them as a winner. The set was written by hand for these manuals, so it measures this corpus, not Thai retrieval in general.

Recall@k and MRR count answerable questions; *threshold* is the top score below which the answer is "not found" (best balanced accuracy on answerable vs no-answer questions); *false answer* is the share of no-answer questions still answered at that threshold.

## All configurations (best first)

| configuration | chunks | R@1 | R@3 | R@5 | MRR | role acc | threshold | false answer | answerable refused | ms/query | embed s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BAAI-bge-m3__198a08a9 | 25 | 0.69 | 0.96 | 0.96 | 0.79 | 0.96 | 0.56 | 0.00 | 0.04 | 387 | 0.0 |

## Best configuration per question type

| type | questions | configuration | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|---|
| cross_role | 6 | BAAI-bge-m3__198a08a9 | 0.50 | 0.83 | 0.83 | 0.61 |
| screenshot_only | 6 | BAAI-bge-m3__198a08a9 | 1.00 | 1.00 | 1.00 | 1.00 |
| text | 14 | BAAI-bge-m3__198a08a9 | 0.64 | 1.00 | 1.00 | 0.79 |

## No-answer questions: best "not found" threshold

| configuration | threshold | balanced accuracy | false answer | answerable refused |
|---|---|---|---|---|
| BAAI-bge-m3__198a08a9 | 0.56 | 0.98 | 0.00 | 0.04 |

## Per question: BAAI-bge-m3__198a08a9

| id | type | first relevant rank | top score | top role |
|---|---|---|---|---|
| q01 | text | 1 | 0.687 | branch_manager |
| q02 | text | 3 | 0.641 | district_manager |
| q03 | text | 1 | 0.640 | branch_manager |
| q04 | text | 1 | 0.723 | branch_manager |
| q05 | text | 1 | 0.699 | branch_manager |
| q06 | text | 3 | 0.684 | branch_manager |
| q07 | text | 1 | 0.692 | branch_manager |
| q08 | text | 2 | 0.627 | branch_manager |
| q09 | text | 2 | 0.612 | branch_manager |
| q10 | text | 1 | 0.672 | branch_manager |
| q11 | text | 1 | 0.718 | district_manager |
| q12 | text | 3 | 0.708 | district_manager |
| q13 | text | 1 | 0.655 | district_manager |
| q14 | text | 1 | 0.681 | district_manager |
| q15 | screenshot_only | 1 | 0.620 | branch_manager |
| q16 | screenshot_only | 1 | 0.500 | branch_manager |
| q17 | screenshot_only | 1 | 0.595 | branch_manager |
| q18 | screenshot_only | 1 | 0.709 | branch_manager |
| q19 | screenshot_only | 1 | 0.561 | district_manager |
| q20 | screenshot_only | 1 | 0.593 | district_manager |
| q21 | cross_role | 3 | 0.675 | branch_manager |
| q22 | cross_role | 1 | 0.688 | district_manager |
| q23 | cross_role | 3 | 0.712 | branch_manager |
| q24 | cross_role | - | 0.671 | district_manager |
| q25 | cross_role | 1 | 0.682 | branch_manager |
| q26 | cross_role | 1 | 0.663 | district_manager |
| q27 | no_answer | - | 0.506 | branch_manager |
| q28 | no_answer | - | 0.518 | branch_manager |
| q29 | no_answer | - | 0.452 | branch_manager |
| q30 | no_answer | - | 0.545 | district_manager |
