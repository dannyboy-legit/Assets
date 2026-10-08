# External vision pipeline

Place images in `vision/inbox/`.

A GitHub Actions runner sends each image to the configured Hugging Face vision Space and stores the returned analysis as JSON in `vision/results/`. GPT is not involved in the image-analysis step.

Default model Space: `akhaliq/Qwen3-VL-2B-Instruct`.

Optional configuration:
- Repository variable: `HF_VISION_SPACE`
- Actions secret: `HF_TOKEN` (only if the selected Space requires authentication)
