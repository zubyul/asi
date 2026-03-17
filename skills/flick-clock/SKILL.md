---
name: flick-clock
description: A perceptual-time clock grounded in the flick (1/705600000s) — the
  indivisible quantum of visual perception. Bridges BCI temporal resolution with
  the reafference/corollary-discharge perception-action loop. Use when building
  perceptual timing systems, BCI frame-rate synchronization, or visual displays
  that respect human temporal resolution boundaries.
license: MIT
metadata:
  trit: 0
  source: asi
---

# Flick Clock

A clock that operates at the boundary between machine time and perceptual time, grounded in the **flick** — the atomic unit of visual perception.

## What Is a Flick?

**1 flick = 1/705,600,000 of a second**

The flick is the smallest time unit larger than a nanosecond that **evenly divides every common media frame rate and audio sample rate**:

| Rate | Flicks per frame/sample |
|------|------------------------|
| 24 fps (film) | 29,400,000 |
| 25 fps (PAL) | 28,224,000 |
| 30 fps (NTSC) | 23,520,000 |
| 48 fps | 14,700,000 |
| 60 fps | 11,760,000 |
| 90 fps (VR) | 7,840,000 |
| 120 fps | 5,880,000 |
| 44,100 Hz (CD audio) | 16,000 |
| 48,000 Hz | 14,700 |
| 96,000 Hz | 7,350 |
| 192,000 Hz | 3,675 |

This matters because **every perceptual frame boundary lands exactly on a flick**. No floating-point drift. No accumulated temporal error. The flick is a temporal quantum — the indivisible unit from which all perceptual time is composed.

## Why Flicks Matter for BCI

In a brain-computer interface, the system must synchronize between:

1. **Neural sampling rate** — EEG at 256–1024 Hz, ECoG at 1–30 kHz
2. **Visual refresh rate** — display at 60–240 Hz
3. **Audio feedback rate** — 44.1–192 kHz
4. **Reafference loop rate** — the perception-action cycle

The flick is the **greatest common temporal denominator** across all these rates. When a BCI system timestamps events in flicks, every modality aligns without rounding error.

### Connection to Reafference & Corollary Discharge

In the repo's `reafference-corollary-discharge` skill (von Holst, 1950), the perception-action loop operates as:

```
Efference Copy (prediction) ──→ Comparator ←── Sensory Reafference (observation)
                                    │
                              error signal
                                    │
                           Corollary Discharge
                          (suppress or amplify)
```

The critical question: **at what temporal resolution does this loop run?**

- If the loop ticks at 1ms, you get 2ms latency (sequential: action → sense → compare)
- If the loop ticks at the Möbius-inverted rate (action ≡ perception), the question becomes: what is the finest temporal grain at which the system can distinguish "same frame" from "next frame"?

That grain is the **flick**. The corollary discharge comparator needs a temporal resolution that:
- Aligns with the visual display refresh (so the predicted frame matches the actual frame)
- Aligns with the audio feedback (so predicted sound matches actual sound)
- Has zero accumulated drift across hours of BCI operation

The flick satisfies all three by construction.

### Connection to Möbius Perception/Action Simultaneity

From `MOEBIUS_PERCEPTION_ACTION_SIMULTANEITY.md`:

> Action ≡ Perception (simultaneous duality via Möbius inversion)

The flick clock makes this concrete. When action and perception are simultaneous (not sequential), you don't need a "latency budget" — but you still need a **phase clock** to define "now." The flick is that phase clock. It answers: when does one perceptual instant end and the next begin?

```
Flick N:     Action(N) ≡ Perception(N)     [Möbius dual pair]
Flick N+1:   Action(N+1) ≡ Perception(N+1) [next dual pair]

The boundary between N and N+1 is exactly 1 flick.
No sub-flick events are perceptually distinguishable.
```

## The Invisible Clock Paradox

> "A clock that ticks every flick is invisible."

This is the foundational observation. A flick-rate clock (705.6 MHz) is **beneath perceptual resolution** — you cannot see it tick. It operates at the interference rate: not the rate at which you blink, but the rate at which you *perceive perceiving*. It's the meta-perceptual refresh — the clock that tells your visual system when one frame ends and the next begins, but which itself can never be seen as a frame.

This is why the flick is the right unit for BCI: it's the **temporal infrastructure** that perception runs on top of, not within.

### The Perceiver as Information Bottleneck

The deeper insight: the perceiver is the bottleneck, not the signal.

```
Stimuli arrive continuously (photons, pressure waves, neural spikes)
                    │
                    ▼
    ┌──────────────────────────────┐
    │   PERCEIVER (bottleneck)      │
    │                              │
    │   Can only "finish" one      │
    │   perceptual frame before    │
    │   the next batch arrives.    │
    │                              │
    │   New data forces old data   │
    │   to either:                 │
    │     (a) commit to memory     │
    │     (b) be overwritten       │
    │                              │
    │   This is the "good pressure"│
    │   — the temporal forcing     │
    │   function that makes        │
    │   perception lossy but real- │
    │   time.                      │
    └──────────────────────────────┘
                    │
                    ▼
           Perceptual frame N
           (compressed, committed)
```

The flick is the tick of this bottleneck. It's the minimum interval between "you must decide what you saw" events. Below the flick, stimuli accumulate. At the flick boundary, the bottleneck forces a commit-or-discard decision.

In BCI terms: the flick boundary is where the neural decoder must emit its classification. Wait longer and you lose real-time. Emit earlier and you're sub-flick — no display can show it, no audio system can play it.

### Color Round-Trip and Channel Noise

When a color is emitted by one screen, captured by a camera, and displayed on a second screen, the round-trip introduces noise:

```
Screen A (emit color C)
    → photons through air
    → camera sensor (Bayer filter, gain, white balance)
    → ISP pipeline (demosaic, gamma, compression)
    → network transport (lossy codec)
    → Screen B (display C' ≠ C)

C' = C + noise(camera) + noise(display) + noise(compression)
```

This is a **physical reafference loop**. The efference copy is C (the color you sent). The reafference is C' (the color you see come back). The corollary discharge comparator asks: is C' close enough to C to suppress, or different enough to amplify?

The flick matters here because the **temporal alignment** of when you sample C vs when you observe C' determines whether the comparison is valid. If the camera samples at 30fps and the display runs at 60fps, the comparison needs a shared temporal reference — and the flick is the only unit that evenly divides both.

## The Flick Clock Visualization

A split-flap display where each digit flick is synchronized to real time — the visual metaphor for discrete perceptual quanta replacing each other.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flick Clock</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
    background: #1a1a2e;
    font-family: 'Courier New', monospace;
    overflow: hidden;
  }

  .clock-container { text-align: center; }

  .clock {
    display: flex; align-items: center; gap: 12px;
    perspective: 600px;
  }

  .separator {
    font-size: 72px; color: #e0e0e0; line-height: 1;
    animation: pulse 1s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .digit-pair { display: flex; gap: 6px; }

  .flick-digit {
    position: relative;
    width: 72px; height: 108px;
    border-radius: 8px;
    font-size: 80px; font-weight: bold;
    color: #f0f0f0;
    text-align: center;
  }

  .flick-digit .top,
  .flick-digit .bottom,
  .flick-digit .flap-front,
  .flick-digit .flap-back {
    position: absolute; left: 0; width: 100%;
    height: 50%; overflow: hidden;
    background: #16213e;
    display: flex; justify-content: center;
    border-radius: 8px 8px 0 0;
  }

  .flick-digit .top       { top: 0; align-items: flex-end; border-bottom: 1.5px solid #0f3460; line-height: 2.16; }
  .flick-digit .bottom    { bottom: 0; align-items: flex-start; border-radius: 0 0 8px 8px; line-height: 0; }

  .flick-digit .flap-front,
  .flick-digit .flap-back {
    position: absolute; left: 0; width: 100%;
    backface-visibility: hidden;
  }
  .flick-digit .flap-front {
    top: 0; align-items: flex-end; line-height: 2.16;
    transform-origin: bottom center;
    border-bottom: 1.5px solid #0f3460;
    z-index: 2;
  }
  .flick-digit .flap-back {
    bottom: 0; align-items: flex-start; line-height: 0;
    transform-origin: top center;
    transform: rotateX(180deg);
    border-radius: 0 0 8px 8px;
    z-index: 2;
  }

  .flick-digit.flicking .flap-front {
    animation: flapDown 0.4s ease-in forwards;
  }
  .flick-digit.flicking .flap-back {
    animation: flapUp 0.4s 0.2s ease-out forwards;
  }

  @keyframes flapDown {
    0%   { transform: rotateX(0deg); }
    100% { transform: rotateX(-90deg); }
  }
  @keyframes flapUp {
    0%   { transform: rotateX(180deg); }
    100% { transform: rotateX(0deg); }
  }

  .flick-digit .top,
  .flick-digit .bottom {
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }

  .labels {
    display: flex; justify-content: center; gap: 12px;
    margin-top: 18px; color: #5a6e8a; font-size: 13px; letter-spacing: 3px;
  }
  .labels span { width: 150px; text-align: center; }

  /* flick counter — the perceptual quantum ticker */
  .flick-counter {
    margin-top: 24px; color: #0f3460; font-size: 11px;
    font-family: 'Courier New', monospace; letter-spacing: 1px;
  }
  .flick-counter .value {
    color: #2a9d8f; font-size: 14px; font-weight: bold;
  }

  /* reafference status bar */
  .reafference-bar {
    margin-top: 12px; display: flex; justify-content: center; gap: 24px;
    font-size: 11px; color: #5a6e8a;
  }
  .reafference-bar .safe    { color: #49EE54; }
  .reafference-bar .warning { color: #E67F86; }
</style>
</head>
<body>

<div class="clock-container">
  <div class="clock" id="clock"></div>
  <div class="labels">
    <span>HOURS</span>
    <span>MINUTES</span>
    <span>SECONDS</span>
  </div>
  <div class="flick-counter">
    FLICKS ELAPSED TODAY: <span class="value" id="flicks">0</span>
    <br>
    <span style="color:#5a6e8a">1 flick = 1/705,600,000 s &mdash; the perceptual quantum</span>
  </div>
  <div class="reafference-bar">
    <span>EFFERENCE: <span class="safe" id="efference">PREDICTED</span></span>
    <span>REAFFERENCE: <span class="safe" id="reafference">MATCHED</span></span>
    <span>DISCHARGE: <span class="safe" id="discharge">SUPPRESSED</span></span>
  </div>
</div>

<script>
  const FLICKS_PER_SECOND = 705600000;

  function makeDigit(id) {
    const el = document.createElement('div');
    el.className = 'flick-digit';
    el.id = id;
    el.innerHTML = `
      <div class="top"><span>0</span></div>
      <div class="bottom"><span>0</span></div>
      <div class="flap-front"><span>0</span></div>
      <div class="flap-back"><span>0</span></div>`;
    return el;
  }

  function makePair(prefix) {
    const pair = document.createElement('div');
    pair.className = 'digit-pair';
    pair.appendChild(makeDigit(prefix + '0'));
    pair.appendChild(makeDigit(prefix + '1'));
    return pair;
  }

  function makeSep() {
    const s = document.createElement('div');
    s.className = 'separator';
    s.textContent = ':';
    return s;
  }

  const clock = document.getElementById('clock');
  clock.appendChild(makePair('h'));
  clock.appendChild(makeSep());
  clock.appendChild(makePair('m'));
  clock.appendChild(makeSep());
  clock.appendChild(makePair('s'));

  const digits = {};
  ['h0','h1','m0','m1','s0','s1'].forEach(id => {
    digits[id] = { el: document.getElementById(id), value: null };
  });

  function flickTo(obj, newVal) {
    if (obj.value === newVal) return;
    const el = obj.el;
    const spans = el.querySelectorAll('span');

    spans[0].textContent = newVal;
    spans[2].textContent = obj.value ?? newVal;
    spans[3].textContent = newVal;

    el.classList.remove('flicking');
    void el.offsetWidth;
    el.classList.add('flicking');

    setTimeout(() => {
      spans[1].textContent = newVal;
      el.classList.remove('flicking');
    }, 600);

    obj.value = newVal;
  }

  // Reafference simulation: predict the next second, compare with actual
  let predictedSecond = null;

  function tick() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');

    // Corollary discharge: did our prediction match?
    const currentSecond = now.getSeconds();
    const efEl = document.getElementById('efference');
    const reEl = document.getElementById('reafference');
    const dcEl = document.getElementById('discharge');

    if (predictedSecond !== null) {
      if (predictedSecond === currentSecond) {
        // Prediction matched — suppress (normal operation)
        reEl.textContent = 'MATCHED';
        reEl.className = 'safe';
        dcEl.textContent = 'SUPPRESSED';
        dcEl.className = 'safe';
      } else {
        // Mismatch — amplify (anomaly: clock skew, tab suspended, etc.)
        reEl.textContent = 'MISMATCH';
        reEl.className = 'warning';
        dcEl.textContent = 'AMPLIFIED';
        dcEl.className = 'warning';
      }
    }

    // Efference copy: predict next second
    predictedSecond = (currentSecond + 1) % 60;
    efEl.textContent = 'PREDICT:' + String(predictedSecond).padStart(2, '0');

    flickTo(digits.h0, h[0]);
    flickTo(digits.h1, h[1]);
    flickTo(digits.m0, m[0]);
    flickTo(digits.m1, m[1]);
    flickTo(digits.s0, s[0]);
    flickTo(digits.s1, s[1]);

    // Flick counter: total flicks elapsed since midnight
    const midnightMs = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const elapsedMs = now.getTime() - midnightMs;
    const elapsedFlicks = BigInt(elapsedMs) * BigInt(FLICKS_PER_SECOND) / 1000n;
    document.getElementById('flicks').textContent = elapsedFlicks.toLocaleString();
  }

  tick();
  setInterval(tick, 1000);
</script>
</body>
</html>
```

## Architecture: Four Layers of Flick

The flick clock maps directly onto the reafference-corollary-discharge four-layer architecture:

| Layer | Reafference Role | Flick Clock Implementation |
|-------|-----------------|---------------------------|
| `.top` (static new) | **Efference Copy** — the predicted next state | Shows the value the system expects |
| `.flap-front` (old, flipping away) | **Sensory Reafference** — the observed current state | The old value being replaced by observation |
| `.flap-back` (new, landing) | **Comparator** — predicted vs observed | The transition itself; when prediction matches, it's a smooth flick |
| `.bottom` (static, updates last) | **Corollary Discharge** — suppress or amplify | Final settled state; if smooth → suppressed (normal); if jarring → amplified (anomaly) |

### The Flick as Temporal Boundary

```
Time ──────────────────────────────────────────────►

    ┌──────────────┐  ┌──────────────┐  ┌──────────
    │  Perceptual   │  │  Perceptual   │  │
    │  Frame N      │  │  Frame N+1    │  │  Frame N+2
    │               │  │               │  │
    │ Action(N) ≡   │  │ Action(N+1)≡  │  │
    │ Perception(N) │  │ Perception(N+1│  │
    └──────────────┘  └──────────────┘  └──────────
                   ↑                 ↑
              1 flick           1 flick
         (boundary: old        (boundary: old
          card flips away)      card flips away)
```

Each flick boundary is a split-flap flip. The old perceptual frame peels away (flapDown), the new one lands (flapUp). Between flicks, perception and action are **simultaneous** — there is no "processing delay" within a single flick.

## Integration with Plurigrid BCI Stack

### Temporal Synchronization Protocol

In a distributed BCI system across plurigrid nodes, each node needs a shared temporal reference. The flick provides this:

```javascript
// Shared flick timestamp across plurigrid nodes
function flickTimestamp() {
  // performance.now() gives sub-millisecond precision
  const ms = performance.now() + performance.timeOrigin;
  return BigInt(Math.floor(ms)) * BigInt(705600000) / 1000n;
}

// Check if two events are in the same perceptual frame at a given rate
function sameFrame(flickA, flickB, fps) {
  const flicksPerFrame = BigInt(705600000 / fps);
  return (flickA / flicksPerFrame) === (flickB / flicksPerFrame);
}

// Example: are two neural events in the same 60fps visual frame?
sameFrame(eventA, eventB, 60);  // true if within same 11,760,000-flick window
```

### Reafference Loop Timing

```
Neural Event (EEG sample at 256 Hz)
  = 2,756,250 flicks per sample

Visual Frame (display at 60 Hz)
  = 11,760,000 flicks per frame

Audio Sample (feedback at 48 kHz)
  = 14,700 flicks per sample

All align on flick boundaries. No drift. No rounding.

Corollary discharge comparator runs at the GCD:
  GCD(2756250, 11760000, 14700) = 75 flicks
  = the finest resolution at which prediction ≡ observation
```

## References

- **Meta/Facebook** (2018). "Flick: A unit of time." github.com/facebookarchive/flicks
- **von Holst, E.** (1950). "The Behavioral Physiology of Animals and Man" — reafference principle
- **Maturana & Varela** (1980). "Autopoiesis and Cognition" — perception-action closure
- **Powers, W.T.** (1973). "Behavior: The Control of Perception" — perceptual control theory
- See also: `skills/reafference-corollary-discharge/SKILL.md`, `ies/music-topos/MOEBIUS_PERCEPTION_ACTION_SIMULTANEITY.md`
