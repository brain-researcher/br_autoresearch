# EP19 conceptual figure — image-generation record

Updated 2026-10-02 with the built-in image-generation tool; no CLI fallback.
Current image: [ep19_question_imagegen-v2.png](ep19_question_imagegen-v2.png). Earlier project images are retained.

The actual generated image was inspected for scientific operators, labels,
readability and outcome claims, then independently reviewed. Replaced the flat-peripheral assumption and pseudo winner bars; made older complement retention and all four input combinations explicit, with a hand-onset illustration.
No participant/stimulus payload, observed neural value or result was used.

## Initial generation — exact prompt

```text
Use case: scientific-educational.
Create a new landscape research concept figure for EP19. White background, crisp navy sans-serif typography, muted teal/orange/purple, three spacious panels A–C. Draw the scientific contrast; no miniature protocol or dashboard.
Exact title: "Does past EEG improve movement forecasts?"
A, "Forecast before onset": one horizontal timeline with "Now" at a vertical divider, a past interval to its left containing an abstract EEG trace and a variable peripheral trace, both ending exactly at Now. Label these "Past EEG" and "Past sensors + context". To the RIGHT draw three adjacent future bands "0–300 ms", "300–600 ms", "600–900 ms"; emphasize the middle band softly and label it "Primary forecast". A future movement icon lies in that middle interval with a question mark. Do not imply that past peripheral signals are perfectly flat or that future samples enter the predictor.
B, "What does EEG add?": two equal model paths: "Sensors + context" versus "Sensors + context + EEG", converging on "Same continuous at-risk sequence". A third small comparison label below reads "Zero EEG and matched surrogates". No invented performance bars or observed superiority.
C, "Which past input matters?": a simple timeline ending at Now, with a long older segment labelled "Earlier slow component" and the last 200 ms labelled "Recent 200 ms". Under the older segment show a second thin gray rail labelled "Complement always retained". A 2×2 grid below is labelled "Recent off / on" by rows and "Older slow off / on" by columns, with small neutral switches only; no predicted result bars. Bottom text: "Repeat the procedure in a second task".
Only footer: "Conceptual schematic — no observed data".
Avoid causal physiology, intention labels, fully flat sensor assumptions, universal sample counts, online-clinical claims, locks, margin tables, warning banners, invented score differences and decorative brain icons. All text must remain large and readable.
```

## Targeted correction — exact prompt

```text
Use case: precise-object-edit. Correct ONLY the scientific schematic details described here. Panel C: remove the four identical toggle-switch icons from the 2x2 factorial table. Keep row and column Off/On labels. Put exact text in cells: top-left 'Complement only'; top-right 'Complement + older slow'; bottom-left 'Complement + recent'; bottom-right 'Complement + both'. Use readable wrapping. The gray rail labelled 'Complement always retained' must end at the older/recent boundary (the left edge of the orange Recent 200 ms segment), NOT extend into the recent segment or to Now. This complement uses only older raw time support. Panel A: replace the running-person movement icon with a simple small hand-and-object movement icon inside the same 300–600 ms band, retaining the question mark. Preserve every other element, title, labels, colors, three-panel layout and footer. No score bars, results or extra text.
```

## Display boundary

This is a conceptual illustration, not a simulation result, empirical plot or
new scientific decision. The current GOAL caption supplies the precise scope.
No analysis, model policy, numerical criterion, resource allocation or data
access changed.
