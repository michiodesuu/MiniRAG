# Retrieval evaluation

> **Small evaluation set:** 54 questions over 5 document(s) (1.คู่มือเจ้าหน้าที่วินิจฉัยเงินทดแทน, คู่มือการปฎิบัติงานการตอบโต้เหตุฉุกเฉินจากอุบัติภัยสารเคมี, ผู้จัดการสาขา, ผู้จัดการเขต, สรุปวิทยาการ(ทฤษฎี)). One question is 0.02 of Recall/MRR. Differences of one or two questions are noise: do not read them as a winner. The set was written by hand for these manuals, so it measures this corpus, not Thai retrieval in general.

Recall@k and MRR count answerable questions; *threshold* is the top score below which the answer is "not found" (best balanced accuracy on answerable vs no-answer questions); *false answer* is the share of no-answer questions still answered at that threshold.

## All configurations (best first)

| configuration | chunks | R@1 | R@3 | R@5 | MRR | role acc | threshold | false answer | answerable refused | ms/query | embed s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BAAI-bge-m3__1a6e93b0 | 242 | 0.67 | 0.85 | 0.92 | 0.78 | 0.88 | 0.60 | 0.17 | 0.12 | 226 | 0.0 |

## Best configuration per question type

| type | questions | configuration | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|---|
| cross_role | 6 | BAAI-bge-m3__1a6e93b0 | 0.67 | 0.67 | 0.83 | 0.71 |
| screenshot_only | 11 | BAAI-bge-m3__1a6e93b0 | 0.64 | 0.82 | 0.82 | 0.73 |
| text | 31 | BAAI-bge-m3__1a6e93b0 | 0.68 | 0.90 | 0.97 | 0.81 |

## No-answer questions: best "not found" threshold

| configuration | threshold | balanced accuracy | false answer | answerable refused |
|---|---|---|---|---|
| BAAI-bge-m3__1a6e93b0 | 0.60 | 0.85 | 0.17 | 0.12 |

## Per question: BAAI-bge-m3__1a6e93b0

| id | type | first relevant rank | top score | top role |
|---|---|---|---|---|
| q01 | text | 1 | 0.687 | branch_manager |
| q02 | text | 4 | 0.672 | district_manager |
| q03 | text | 1 | 0.640 | branch_manager |
| q04 | text | 2 | 0.645 | branch_manager |
| q05 | text | 1 | 0.596 | branch_manager |
| q06 | text | 1 | 0.609 | branch_manager |
| q07 | text | 1 | 0.708 | branch_manager |
| q08 | text | 2 | 0.629 | branch_manager |
| q09 | text | 2 | 0.612 | branch_manager |
| q10 | text | 1 | 0.670 | branch_manager |
| q11 | text | 1 | 0.706 | district_manager |
| q12 | text | 2 | 0.607 | district_manager |
| q13 | text | 1 | 0.655 | district_manager |
| q14 | text | 2 | 0.581 | district_manager |
| q15 | screenshot_only | 1 | 0.626 | branch_manager |
| q16 | screenshot_only | - | 0.582 | - |
| q17 | screenshot_only | 2 | 0.598 | - |
| q18 | screenshot_only | 1 | 0.715 | branch_manager |
| q19 | screenshot_only | 1 | 0.573 | district_manager |
| q20 | screenshot_only | 1 | 0.610 | district_manager |
| q21 | cross_role | 1 | 0.728 | branch_manager |
| q22 | cross_role | 1 | 0.734 | district_manager |
| q23 | cross_role | 4 | 0.715 | branch_manager |
| q24 | cross_role | - | 0.671 | district_manager |
| q25 | cross_role | 1 | 0.682 | branch_manager |
| q26 | cross_role | 1 | 0.663 | district_manager |
| q27 | no_answer | - | 0.521 | - |
| q28 | screenshot_only | 2 | 0.715 | - |
| q29 | no_answer | - | 0.472 | - |
| q30 | no_answer | - | 0.619 | - |
| q31 | text | 1 | 0.609 | - |
| q32 | text | 2 | 0.673 | - |
| q33 | text | 2 | 0.674 | - |
| q34 | text | 1 | 0.640 | - |
| q35 | text | 1 | 0.611 | - |
| q36 | text | - | 0.675 | - |
| q37 | text | 4 | 0.651 | - |
| q38 | text | 1 | 0.647 | - |
| q39 | text | 1 | 0.553 | - |
| q40 | text | 1 | 0.568 | - |
| q41 | no_answer | - | 0.583 | - |
| q42 | no_answer | - | 0.563 | branch_manager |
| q43 | text | 1 | 0.730 | - |
| q44 | text | 1 | 0.637 | - |
| q45 | text | 1 | 0.692 | - |
| q46 | text | 1 | 0.596 | - |
| q47 | text | 1 | 0.663 | - |
| q48 | text | 1 | 0.710 | - |
| q49 | text | 1 | 0.780 | - |
| q50 | screenshot_only | 1 | 0.676 | - |
| q51 | screenshot_only | 1 | 0.602 | - |
| q52 | screenshot_only | 1 | 0.682 | - |
| q53 | screenshot_only | - | 0.523 | branch_manager |
| q54 | no_answer | - | 0.582 | - |
