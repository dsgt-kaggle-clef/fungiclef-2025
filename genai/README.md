# genai

This is a pipeline that takes a set of images and tries to predict the species of fungi.
The pipeline is done in three steps.
First, given an image, we predict the family of the fungi.
Then, we predict the genus of the fungi.
Finally, we predict the species of the fungi.

## notes

```
python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/tmp/test-gemini-flash --limit 10
```

```
python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/processed/gemini-flash-001

python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/processed/gemini-2.5-flash-preview --model google/gemini-2.5-flash-preview

python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/processed/openai-gpt-4.1-mini --model openai/gpt-4.1-mini
python -m genai.collate ~/scratch/fungiclef-2025/processed/openai-gpt-4.1-mini 

python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/processed/mistralai-mistral-medium-3 --model mistralai/mistral-medium-3
python -m genai.collate ~/scratch/fungiclef-2025/processed/mistralai-mistral-medium-3


```

I'll need to redo the google ones just so that I have multiple images in the prompt. But I don't know how much this really changes things...