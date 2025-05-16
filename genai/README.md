# genai

This is a pipeline that takes a set of images and tries to predict the species of fungi.
The pipeline is done in three steps.
First, given an image, we predict the family of the fungi.
Then, we predict the genus of the fungi.
Finally, we predict the species of the fungi.

## notes

```
python -m genai.query ~/scratch/fungiclef-2025 ~/scratch/fungiclef-2025/processed/gemini-flash-0
```