# OCR benchmark

Ground truth: 18 images (3 by a person, 15 by Claude from a blank box and not yet checked by a person, 0 of unknown origin). Icons and duplicate images are not scored. CER lower is better; per-image CER capped at 1.0.

> Ground truth made by Claude is an independent reading of the image, not a human check; the table of ground truth by a person is the stricter reference.

## All images (n=18)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| pipeline | 0.164 | 0.169 | 0 | 18 |
| vision_typhoon | 0.225 | 0.265 | 0 | 18 |
| vision_hint | 0.228 | 0.298 | 0 | 18 |
| easyocr | 0.251 | 0.304 | 0 | 18 |
| tesseract | 0.291 | 0.288 | 0 | 18 |
| vision_both | 0.292 | 0.749 | 0 | 18 |
| ocr | 0.319 | 0.333 | 0 | 18 |
| vision | 0.397 | 0.722 | 0 | 18 |

## Ground truth by a person (n=3)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| vision_typhoon | 0.100 | 0.177 | 0 | 3 |
| pipeline | 0.100 | 0.177 | 0 | 3 |
| vision | 0.133 | 0.202 | 0 | 3 |
| vision_hint | 0.182 | 0.424 | 0 | 3 |
| vision_both | 0.204 | 0.475 | 0 | 3 |
| ocr | 0.244 | 0.540 | 0 | 3 |
| easyocr | 0.256 | 0.509 | 0 | 3 |
| tesseract | 0.372 | 0.576 | 0 | 3 |

## Ground truth by Claude (blank box) (n=15)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| pipeline | 0.177 | 0.168 | 0 | 15 |
| vision_hint | 0.237 | 0.291 | 0 | 15 |
| easyocr | 0.250 | 0.293 | 0 | 15 |
| vision_typhoon | 0.251 | 0.270 | 0 | 15 |
| tesseract | 0.274 | 0.272 | 0 | 15 |
| vision_both | 0.310 | 0.765 | 0 | 15 |
| ocr | 0.333 | 0.321 | 0 | 15 |
| vision | 0.450 | 0.752 | 0 | 15 |

## Small crops (reference <= 60 chars) (n=3)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| vision_hint | 0.000 | 0.000 | 0 | 3 |
| vision_both | 0.000 | 0.000 | 0 | 3 |
| ocr | 0.013 | 0.013 | 0 | 3 |
| vision_typhoon | 0.034 | 0.032 | 0 | 3 |
| pipeline | 0.034 | 0.032 | 0 | 3 |
| easyocr | 0.044 | 0.044 | 0 | 3 |
| vision | 0.054 | 0.051 | 0 | 3 |
| tesseract | 0.237 | 0.234 | 0 | 3 |

## Screens and tables (reference > 60 chars) (n=15)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| pipeline | 0.190 | 0.171 | 0 | 15 |
| vision_typhoon | 0.264 | 0.269 | 0 | 15 |
| vision_hint | 0.274 | 0.304 | 0 | 15 |
| easyocr | 0.292 | 0.310 | 0 | 15 |
| tesseract | 0.301 | 0.289 | 0 | 15 |
| vision_both | 0.351 | 0.764 | 0 | 15 |
| ocr | 0.380 | 0.339 | 0 | 15 |
| vision | 0.466 | 0.735 | 0 | 15 |

## Completeness, order ignored (character 3-grams, higher is better)

Recall: share of the reference that was read; precision: share of the output that is real. CER also counts text read in another order as error; these do not.

| method | mean recall | mean precision | images |
|---|---|---|---|
| pipeline | 0.874 | 0.921 | 18 |
| vision_typhoon | 0.804 | 0.833 | 18 |
| easyocr | 0.787 | 0.804 | 18 |
| vision_hint | 0.785 | 0.922 | 18 |
| vision_both | 0.777 | 0.858 | 18 |
| tesseract | 0.764 | 0.845 | 18 |
| ocr | 0.691 | 0.765 | 18 |
| vision | 0.588 | 0.651 | 18 |

## Failures on every benchmarked image (ingest would reject these)

| method | failed | of | rejected (rule: count) | no output |
|---|---|---|---|---|
| easyocr | 70 | 206 | empty: 70 | 0 |
| pipeline | 70 | 206 | - | 70 |
| vision_hint | 188 | 206 | - | 188 |
| ocr | 188 | 206 | - | 188 |
| tesseract | 188 | 206 | - | 188 |
| vision | 189 | 206 | no_text_marker: 1 | 188 |
| vision_both | 189 | 206 | repetition_loop: 1 | 188 |
| vision_typhoon | 190 | 206 | no_text_marker: 1, prompt_echo: 1 | 188 |

## Per image (CER)

| image | by | ref chars | vision | vision_hint | vision_both | vision_typhoon | ocr | easyocr | tesseract | pipeline |
|---|---|---|---|---|---|---|---|---|---|---|
| ผู้จัดการสาขา_page_10_image_1 | human | 46 | 0.09 | 0.00 | 0.00 | 0.07 | 0.02 | 0.02 | 0.30 | 0.07 |
| ผู้จัดการสาขา_page_11_image_1 | human | 346 | 0.24 | 0.55 | 0.61 | 0.22 | 0.69 | 0.64 | 0.68 | 0.22 |
| ผู้จัดการสาขา_page_2_image_2 | claude | 664 | 0.39 | 0.61 | 0.31 | 0.42 | 0.63 | 0.60 | 0.31 | 0.42 |
| ผู้จัดการสาขา_page_3_image_2 | claude | 176 | 0.26 | 0.23 | 0.23 | 0.29 | 0.23 | 0.37 | 0.32 | 0.29 |
| ผู้จัดการสาขา_page_3_image_4 | claude | 274 | 0.36 | 0.10 | 0.18 | 0.10 | 0.19 | 0.19 | 0.20 | 0.10 |
| ผู้จัดการสาขา_page_4_image_1 | claude | 752 | 0.64 | 0.11 | 0.33 | 0.25 | 0.19 | 0.21 | 0.33 | 0.25 |
| ผู้จัดการสาขา_page_4_image_2 | claude | 194 | 0.19 | 0.04 | 0.02 | 0.03 | 0.24 | 0.06 | 0.05 | 0.03 |
| ผู้จัดการสาขา_page_4_image_3 | claude | 285 | 0.25 | 0.00 | 0.06 | 0.00 | 0.05 | 0.02 | 0.08 | 0.00 |
| ผู้จัดการสาขา_page_5_image_1 | claude | 1360 | 0.72 | 0.49 | 0.51 | 0.09 | 0.39 | 0.33 | 0.37 | 0.09 |
| ผู้จัดการสาขา_page_6_image_1 | claude | 853 | 0.36 | 0.15 | 0.35 | 0.24 | 0.26 | 0.38 | 0.36 | 0.24 |
| ผู้จัดการสาขา_page_7_image_1 | claude | 880 | 0.36 | 0.18 | 1.00 | 0.99 | 0.26 | 0.39 | 0.31 | 0.18 |
| ผู้จัดการสาขา_page_7_image_3 | claude | 255 | 0.39 | 0.63 | 0.63 | 0.93 | 0.69 | 0.53 | 0.44 | 0.63 |
| ผู้จัดการสาขา_page_8_image_3 | human | 54 | 0.07 | 0.00 | 0.00 | 0.02 | 0.02 | 0.11 | 0.13 | 0.02 |
| ผู้จัดการสาขา_page_9_image_2 | claude | 58 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.28 | 0.02 |
| ผู้จัดการเขต_page_2_image_1 | claude | 67 | 0.16 | 0.04 | 0.00 | 0.00 | 0.94 | 0.00 | 0.00 | 0.00 |
| ผู้จัดการเขต_page_3_image_1 | claude | 840 | 1.00 | 0.50 | 0.38 | 0.02 | 0.48 | 0.13 | 0.11 | 0.02 |
| ผู้จัดการเขต_page_4_image_1 | claude | 970 | 0.75 | 0.11 | 0.29 | 0.06 | 0.12 | 0.17 | 0.12 | 0.06 |
| ผู้จัดการเขต_page_5_image_1 | claude | 106 | 0.92 | 0.37 | 0.37 | 0.34 | 0.35 | 0.37 | 0.84 | 0.34 |
