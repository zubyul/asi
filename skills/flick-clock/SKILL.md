---
name: flick-clock
description: A split-flap (flick) clock animation rendered in HTML/CSS/JS. Use when
  users request flip clocks, flick clocks, split-flap displays, retro clock animations,
  or mechanical digit displays.
license: MIT
metadata:
  trit: 2
  source: asi
---

# Flick Clock

A split-flap "flick" clock — each digit flips with a mechanical card-flick animation, updating in real time.

## Standalone HTML

Save and open in a browser for a fully self-contained flick clock.

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

  .digit-pair {
    display: flex; gap: 6px;
  }

  .flick-digit {
    position: relative;
    width: 72px; height: 108px;
    border-radius: 8px;
    font-size: 80px; font-weight: bold;
    color: #f0f0f0;
    text-align: center;
  }

  /* --- card halves --- */
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

  /* flap (the piece that flicks) */
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

  /* flick animation */
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

  /* subtle glow on each card */
  .flick-digit .top,
  .flick-digit .bottom {
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }

  /* label row */
  .labels {
    display: flex; justify-content: center; gap: 12px;
    margin-top: 18px; color: #5a6e8a; font-size: 13px; letter-spacing: 3px;
  }
  .labels span { width: 150px; text-align: center; }
</style>
</head>
<body>

<div>
  <div class="clock" id="clock"></div>
  <div class="labels">
    <span>HOURS</span>
    <span>MINUTES</span>
    <span>SECONDS</span>
  </div>
</div>

<script>
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

    // current value stays on bottom-half (old), top-half shows old via flap
    // new value goes to: top (static) and flap-back
    spans[0].textContent = newVal;             // .top
    spans[2].textContent = obj.value ?? newVal; // .flap-front (old, about to flip away)
    spans[3].textContent = newVal;             // .flap-back  (new, revealed)

    el.classList.remove('flicking');
    void el.offsetWidth; // reflow
    el.classList.add('flicking');

    setTimeout(() => {
      spans[1].textContent = newVal; // .bottom
      el.classList.remove('flicking');
    }, 600);

    obj.value = newVal;
  }

  function tick() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');

    flickTo(digits.h0, h[0]);
    flickTo(digits.h1, h[1]);
    flickTo(digits.m0, m[0]);
    flickTo(digits.m1, m[1]);
    flickTo(digits.s0, s[0]);
    flickTo(digits.s1, s[1]);
  }

  tick();
  setInterval(tick, 1000);
</script>
</body>
</html>
```

## Key Concepts

### The Flick Mechanic

Each digit is a stack of four layers:

| Layer        | Role                                         |
|------------- |----------------------------------------------|
| `.top`       | Static top half — shows the **new** value    |
| `.bottom`    | Static bottom half — updates after animation |
| `.flap-front`| Hinged at bottom — shows **old** value, flips down to reveal `.top` |
| `.flap-back` | Hinged at top — shows **new** value, flips up into place over `.bottom` |

The animation sequence:
1. **flapDown** (0 → 0.4s) — front flap rotates from 0° to −90° (old digit peels away)
2. **flapUp** (0.2s → 0.6s) — back flap rotates from 180° to 0° (new digit lands)
3. At 0.6s the `.bottom` text updates and flap classes reset

### Customization

```javascript
// 12-hour format
const raw = now.getHours();
const h = String(raw % 12 || 12).padStart(2, '0');

// Custom colors — change the CSS variables
// background:  #1a1a2e  (body)
// card face:   #16213e  (.top, .bottom)
// divider:     #0f3460  (border)
// text:        #f0f0f0  (color)
```

## Integration with p5.js

```javascript
// Embed in a p5 sketch as an overlay
function setup() {
  createCanvas(windowWidth, windowHeight);
  const clockDiv = createDiv('');
  clockDiv.id('clock');
  clockDiv.position(width / 2 - 240, height / 2 - 70);
  // inject the clock HTML/CSS/JS into the div
}
```

## Best Practices

- Call `setInterval(tick, 1000)` — sub-second polling wastes cycles
- Use `void el.offsetWidth` to force reflow before re-triggering CSS animations
- Keep `perspective` on the parent for realistic 3D flap depth
- Use `backface-visibility: hidden` so reversed flaps don't ghost through
