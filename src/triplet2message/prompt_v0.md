# Prompt V0 - 原始版本

以下是原始的颜值比较 Prompt：

```
Please analyze these THREE images and identify which person is the MOST attractive and which person is the LEAST attractive.

Consider both the person's appearance/styling and how well the photo presents them when making your judgment.

The person to be evaluated in each image has been highlighted with a red bounding box.

Please examine all three images carefully:
- Image A (first image)
- Image B (second image)
- Image C (third image)

**IMPORTANT: You MUST respond with ONLY a valid JSON object, no other text.**

Output Format (copy this structure exactly):
{
  "analysis": "Give a comprehensive analysis that you can detailedly comparing all three images",
  "most_attractive": "A" or "B" or "C" or "unpredictable",
  "least_attractive": "A" or "B" or "C" or "unpredictable"
}

- Set "most_attractive" to the letter of the image with the MOST attractive person.
- Set "least_attractive" to the letter of the image with the LEAST attractive person.
- Use "unpredictable" only if you genuinely cannot make a confident judgment.

Now analyze the three images and respond with ONLY the JSON object:
```
