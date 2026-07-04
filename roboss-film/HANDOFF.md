# ROBOSS — "The World Teaches" — Production Handoff

Status: **All AI-generated clips complete and passed.** Full 3-act rough cut assembled.
Post-production (text/UI/VFX compositing, slates, live-action robot insert, audio mix) is the
next phase and is out of scope for this generation pass.

## 1. What's in this folder

```
roboss-film/
  manifest.json              <- source of truth: every clip, prompt, attempt, verdict
  HANDOFF.md                 <- this file
  clips/{ID}/                <- raw takes, prompts, extracted eval frames, per-take eval JSON
  selects/{ID}.mp4           <- the one selected take per passed clip (clean plates)
  roughcut/
    act1.mp4 + act1_contactsheet.png
    act2.mp4 + act2_contactsheet.png
    act3.mp4 + act3_contactsheet.png
    film_v1.mp4              <- full assembled rough cut, all 3 acts concatenated
  scripts/                   <- generate_clip.py, extract_frames.py, record_take.py,
                                 mark_pass.py, assemble_act.py, spend_report.py
```

`selects/*.mp4` are the picture-locked clean plates — no text, no UI, no VFX baked in.
Everything named "compositing in post" in a clip's `notes` field (skeleton overlays, bounding
boxes, chat-bar text, light threads, robot-sequence overlays, etc.) still needs to be added
by an editor/motion-graphics pass on top of these plates.

## 2. Rough cut runtime

| Segment | Duration |
|---|---|
| Act I  | 37.0s |
| Act II | 29.0s |
| Act III | 79.0s |
| **film_v1.mp4 total** | **145.0s (2:25)** |

The 120s target from the brief is the intended **final broadcast cut** length; this rough cut
runs long because it plays every carousel card and slate placeholder at full generated/spec
length with no editorial trimming. Tightening to 120s (trimming carousel card durations,
overlapping cuts, shortening slate hold times) is a straightforward post/edit pass and hasn't
been done here since it would require picture-lock sign-off first.

## 3. Generation spend summary

- **29 generated clips**, all passed (0 outstanding failures).
- **~55 real generation API calls** across all clips (57 log entries; C08 and C09 each have one
  duplicate log line from an early logging glitch — both were genuine single-take passes).
- **4 clips exceeded the original 4-attempt budget guardrail**, all under your explicit
  standing approval to go past 4 tries:
  - **C04** (workyard crafts tracking shot) — 6 takes. Root cause: persistent legible
    signage/text in an urban-India street setting; fixed by moving the scene to a private
    enclosed workyard with bare walls.
  - **C16** (blank keypress macro) — 6 takes. Root cause: any keyboard/keycap imagery
    strongly biased the model toward printed key legends; fixed by abandoning the keyboard
    concept entirely for a featureless glass pressure tile.
  - **C18g** (dance carousel card 7, Irish stepdance) — 7 takes. Root cause: a recurring
    stochastic vintage film-strip/sprocket-hole vignette artifact, not content-triggered
    (survived a full concept pivot from flamenco to Irish stepdance); fixed by an exhaustive
    negative-prompt list plus persistence through rerolls.
  - **C18j** (dance carousel card 10, studio arm wave) — 5 takes. Same stochastic vignette
    artifact as C18g/C18i; persisted even after pivoting the scene from a street-performer/crowd
    setting to an empty mirrored studio; ultimately resolved on a plain reroll of the
    already-maximally-hardened prompt.
- Per-act breakdown: Act I 7 clips, Act II 7 clips, Act III 15 clips (incl. the 10-card dance
  carousel C18a–j).

## 4. Known/expected gaps (by design, not defects)

- **CARD_DATA, CARD_LOOK, CARD_END** — text-only slates, intentionally never generated
  (post-only). Currently rendered as plain black title cards with drawtext labels in the
  rough cut as editorial placeholders.
- **LIVE_ROBOT** (1:45–1:58, ~13s) — practical live-action robot footage, intentionally never
  AI-generated per the film's core argument (verified real data / real robots for the payoff).
  Currently a labeled black slate in the rough cut; needs to be practically shot and cut in.
- Several clips are clean plates awaiting post compositing: C03 (comped into C12's glass pane),
  C06 (light-thread overlay), C12 (bounding box/skeleton/depth-wipe overlay + C03 comp), C17
  (already has its ghost dancer baked in), C15 (chat text overlay onto the blank capsule), all
  C18a–j cards (skeleton overlay + time-series strip), C19 (ghost-composite reference frames
  extracted to `clips/C19/refs/`), C20 (type overlay onto the light streaks).

## 5. Minor bookkeeping fix made during this session

`C18j`'s `order` field in the manifest was `18.1`, duplicating `C18a`'s order and causing it to
sort immediately after C18a instead of after C18i as the last carousel card. Fixed to `18.95`
before assembling Act III; `roughcut/act3.mp4` reflects the corrected C15→C20 order with the
full C18a→C18j carousel running in sequence.

## 6. Suggested next steps

1. Editorial review of `roughcut/film_v1.mp4` (this checkpoint).
2. Trim to the 120s broadcast length (carousel card pacing + slate durations are the easiest
   places to cut).
3. Shoot `LIVE_ROBOT` practically and cut it into the Act III slate slot.
4. Post/VFX pass: skeleton overlays, bounding boxes, light threads, chat-bar text, type
   treatments, and the C03→C12 comp, per each clip's `notes` field in `manifest.json`.
5. Audio mix/score pass over the now-preserved sync audio tracks.
