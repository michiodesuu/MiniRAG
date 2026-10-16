# OCR benchmark

Ground truth: 4 images (3 short refs <= 60 chars, 1 long). CER lower is better; per-image CER capped at 1.0.

> Only 4 corrected images: treat these numbers as a rough read, not a verdict.

## All images (n=4)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| vision_typhoon | 0.075 | 0.168 | 0 | 4 |
| vision | 0.100 | 0.191 | 0 | 4 |
| vision_hint | 0.137 | 0.402 | 0 | 4 |
| vision_both | 0.153 | 0.451 | 0 | 4 |
| ocr | 0.433 | 0.564 | 1 | 4 |
| tesseract | 0.529 | 0.598 | 1 | 4 |

## Short (tiny crops) (n=3)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| vision_hint | 0.000 | 0.000 | 0 | 3 |
| vision_both | 0.000 | 0.000 | 0 | 3 |
| vision_typhoon | 0.028 | 0.032 | 0 | 3 |
| vision | 0.054 | 0.065 | 0 | 3 |
| ocr | 0.347 | 0.210 | 1 | 3 |
| tesseract | 0.478 | 0.363 | 1 | 3 |

## Long (screens/tables) (n=1)

| method | mean CER | pooled CER | empty outputs | images |
|---|---|---|---|---|
| vision_typhoon | 0.217 | 0.217 | 0 | 1 |
| vision | 0.237 | 0.237 | 0 | 1 |
| vision_hint | 0.546 | 0.546 | 0 | 1 |
| vision_both | 0.613 | 0.613 | 0 | 1 |
| tesseract | 0.682 | 0.682 | 0 | 1 |
| ocr | 0.691 | 0.691 | 0 | 1 |

## Per image (CER)

| image | ref chars | vision | vision_hint | vision_both | vision_typhoon | ocr | tesseract |
|---|---|---|---|---|---|---|---|
| ผู้จัดการสาขา_page_10_image_1 | 46 | 0.09 | 0.00 | 0.00 | 0.07 | 0.02 | 0.30 |
| ผู้จัดการสาขา_page_11_image_1 | 346 | 0.24 | 0.55 | 0.61 | 0.22 | 0.69 | 0.68 |
| ผู้จัดการสาขา_page_2_image_1 | 24 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| ผู้จัดการสาขา_page_8_image_3 | 54 | 0.07 | 0.00 | 0.00 | 0.02 | 0.02 | 0.13 |