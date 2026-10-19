# Controlled native guidance test

This tests experimental appearance attention, not the separate RAVEN backend.
An `observe_only` video is a native-model result: observation does not alter
predictions. A good-looking video alone does not demonstrate guidance.

Use your normal UNET loader, pruned H3 checkpoint, conditioning, LoRAs and native
sampler. Connect the Native Visual Observer's MODEL output to the model path
used by the guider. Do not combine it with the RAVEN Conditioning controller.

For a 90-frame clip, start with:

| Control | Value |
|---|---|
| Resolution | 512 x 320, or the same supported resolution for both runs |
| CFG | 1 |
| `guidance_mode` | `manual_frames` |
| `guidance_source_frame` | `0` |
| `guidance_target_frames` | `68,85` |
| `guidance_layers` | `8,16,24` |
| `guidance_similarity` | `0.85` |
| `max_anchor_tokens` | `128` |
| `min_progress` | `0.25` |
| `observation_interval` | `1` |
| Sampler/schedule/steps | Your normal settings, identical in both runs |

These are exploratory settings, not a validated preset. Manual frame numbers
refer to final-video frames starting at zero. The source must show the subject;
targets should show its reappearance. If the model places the event at different
frames, choose the actual visible frames and rerun the same comparison.

Suggested text-to-video prompt:

> Single continuous shot, fixed camera. An adult woman with a short black bob,
> a teal jacket, and a bright yellow backpack walks slowly from left to right
> through a courtyard. She is clearly visible at the beginning, passes completely
> behind a wide stone pillar in the middle of the shot, then emerges and is
> clearly visible again near the end. The same face, hairstyle, teal jacket, and
> yellow backpack remain consistent throughout. Natural walking motion, steady
> daylight, no cuts.

With image/reference conditioning, use a subject description that agrees with
the input image. Keep every image and conditioning control identical in both
runs. A successful text-to-video comparison does not validate reference or
first/last-frame quality; those need separate comparisons.

1. Bypass Spectrum in **both** runs. Guided execution disables its forecasting;
   comparing against a forecasting-enabled baseline confounds the result.
2. Fix the seed; disable increment/randomize-after-generation. Generate the
   baseline with `guidance_strength=0`.
3. Keep the seed and all other settings fixed, set strength to `0.10`, and
   generate again. Keep Sage in the same mode for both runs.
4. Check the node status or console: the guided run must report a nonzero
   correction count. Zero means no attention corrections were applied; it does
   not prove the node improved or worsened the video.
5. Compare the face/clothes/backpack before and after occlusion, and also check
   walking motion, artifacts and any audio changes. Improvement in one attribute
   can come at the expense of motion or pose. Repeat across seeds and conditions.

Report the seed, model/LoRAs, dimensions/frame count, steps/sampler, mode and
strength, correction count, generation time and visible differences. The
observation timer excludes model inference and preview decoding; peak image
cache bytes exclude model RAM/VRAM. Use system telemetry for total resources.

`observation_interval=2` is an optional later performance experiment. It lowers
observation frequency and may reduce useful reference updates, so first keep
it at `1` for the quality comparison. Temporary images are removed when the
sampler completes or raises; deferred cleanup is reported explicitly.
