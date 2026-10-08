# GitHub Vision Image Analyzer

This package moves image inference into GitHub Actions and a Hugging Face ZeroGPU Space, so the chat session only needs to consume the resulting JSON.

## Flow

GitHub image URL -> GitHub Action -> image download/validation -> Qwen3-VL-2B ZeroGPU Space -> structured JSON -> Actions artifact.

## Install

Copy these files into the root of your `dannyboy-legit/Assets` repository, preserving the paths.

Then commit `analysis/request.json`. That push triggers the workflow.

The default public inference Space is `akhaliq/Qwen3-VL-2B-Instruct`. You can override it with a repository variable named `HF_VISION_SPACE`.

A Hugging Face token is optional for a public Space. Adding an `HF_TOKEN` repository secret lets the request use the authenticated account quota.

## Result

The workflow creates an Actions artifact named `image-analysis-result` containing `analysis.json`.
