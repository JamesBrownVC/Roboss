# INTENT — "The World Teaches" (ROBOSS brand film)

Restated ground truth (Section 0 of the master prompt), before any generation spend.

## Logline

A 120-second cinematic ad in three acts, arguing that robots cannot learn skill from thin air —
only verified real human data can teach them — so the humanoid economy needs a trusted pipeline
from human skill to robot training data.

## Act I — 0:00–0:36 — Warm documentary dawn, India

Skilled workers wear slim matte-black headband cameras with a tiny steady red pinpoint recording
LED while doing masterful, dignified work (machining, soldering, stitching, cooking, mechanics).
Bold statement text (composited in post) and a deep voiceover establish the problem — robots
cannot learn skill from thin air — and the market: a multi-trillion-dollar humanoid economy that
must learn from humans. World: warm golden-hour anamorphic documentary grade with teal shadows,
heavy atmospheric haze, 35mm-style grain.

## Act II — 0:36–1:08 — Cold digital void

Incoming videos are analyzed in real time. Bounding boxes form around objects (composited in post
over clean generated plates — never baked into the generation). One gorgeous but AI-generated
video is caught on a physics anomaly and executed on-screen with a full-frame VIDEO TERMINATED
card (post-only). The next video — real human footage — passes every check and is stamped with a
big emerald checkmark (post-only). Cinema-quality whooshes score the UI events (post-only audio).
World: cold cyan-charcoal grade, volumetric haze, architectural/physical weight — never
hologram/sci-fi kitsch.

## Act III — 1:08–1:43(+) — The ask, the proof, the reveal

A ChatGPT-style chat bar (rendered in post — never generate readable chat text) is typed into:
"Hey Roboss, I need data to teach my robot how to dance." Cinematic loading, then a carousel of
10 distinct real-world dance styles with visible time-series traces beneath each (post-only
overlays), then the screen falls down an endless scroll of data, then a black card reading NOW
LOOK IN FRONT OF YOU (post-only), then a hard cut to REAL, practically-filmed footage of an actual
robot performing the dance. **That final sequence is live action and is never generated** — a
labeled slate stands in for it in the rough cut. World: bright minimal studio / golden-hour
rooftop for the hero dancer.

## The argument

Only verified real human data gets through the pipeline. The one thing that must never look fake
is the ending (robot performing the dance) — it is always real, practical footage. Conversely,
the generated worlds (Acts I and II, and the abstract VFX of Act III) must look authored,
premium, and intentional — never cheap, never AI-kitsch, never carrying any of the UI/typography
that belongs to post.

## Hard rules carried forward from the master prompt

1. Never generate on-screen text, captions, UI boxes, graphs, checkmarks, readable chat bars, or
   logos. Generate clean plates only; all typography/interface graphics are composited in post.
2. Append the standing negative prompt to every generation:
   "no on-screen text, no captions, no subtitles, no watermark, no logo, no distorted hands, no
   extra fingers, no morphing objects, no cartoon look, no oversaturation."
3. Character blocks (MACHINIST, DANCER) are pasted verbatim wherever their character appears —
   never paraphrased.
4. Live-action shots (robot sequence, 1:45–1:58) and post-only cards (0:35 DATA card, 1:43 NOW
   LOOK IN FRONT OF YOU card, 1:58 end cards) are never generated — labeled slates stand in for
   them in the rough cut.
5. Budget guardrail: max 4 automatic generation attempts per clip. After 4 failed attempts, stop,
   write a failure analysis, and ask before spending more.
6. Explicit user approval required at each of the three act checkpoints before proceeding to the
   next act.

## Clip inventory (29 generated clips + non-generated inserts)

- Act I: C01–C07 (7 clips)
- Act I/II boundary: 0:35 DATA card — **post-only, not generated**
- Act II: C08–C14 (7 clips)
- Act III: C15–C17, C18a–C18j (10 dance-style clips), C19 (hero dancer), C20 (data-fall) — 20 clips
- Act III tail: 1:43 NOW LOOK IN FRONT OF YOU card, 1:45–1:58 live-action robot sequence, 1:58 end
  cards — **all post-only or live-action, never generated**

Total generated clips: 7 + 7 + 15 = **29**. At up to 4 attempts each in the worst case, that's a
ceiling of 116 generation calls; the realistic estimate (most clips passing in 1–2 attempts) is
**~30–45 generations**, in line with the master prompt's own ~30-generation estimate.

## Open questions for the user

1. **Video generation model / access**: which Veo model and API tier is actually enabled on your
   Gemini API key (`veo-3.0-generate-preview`, `veo-3.0-fast-generate-preview`, `veo-2.0-generate-001`,
   etc.)? I will confirm this via `client.models.list()` once the key is live, but if you already
   know your plan's exact model id and per-second/per-clip cost, tell me — the "~30 generations"
   estimate above assumes 10s clips at whatever Veo tier your key supports, and actual $ cost
   depends entirely on that tier's pricing.
2. **Confirm budget appetite**: 29 clips × up to 4 attempts is a real ceiling on spend. Do you
   want me to stop and ask after every single failed-then-retried clip, or only at the hard "4
   attempts exhausted" wall and the three act checkpoints, as the master prompt specifies?
3. **Reuse of `v2r/syngen`**: this repo already has a `v2r/` pipeline with what looks like a
   Gemini/Veo generation stage. I'm surveying it now to reuse its model-calling code, SSL/certifi
   workaround, and environment rather than duplicating that work — I'll report what I find before
   writing new generation code.
4. **Image-conditioning for recurring characters**: the master prompt asks me to condition later
   appearances of a hero character (MACHINIST, DANCER) on 3 reference frames from their first
   passing take, "if the API supports image-conditioned generation." I'll confirm whether your
   Veo tier supports image-to-video / reference-image conditioning before relying on it; if not,
   I'll fall back to verbatim character-block text conditioning only, and flag the added
   consistency risk.
5. **Voiceover and music**: not in scope for this agent (no audio generation requested in the
   master prompt) — confirming that's intentional and VO/music/whooshes are a separate post-audio
   workstream, not something I should attempt to generate.

## Status

Setup phase in progress. No generation has been submitted yet. Awaiting environment
confirmation (API key, ffmpeg, SDK) and user go-ahead before any spend.
