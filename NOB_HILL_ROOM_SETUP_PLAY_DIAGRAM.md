# Nob Hill Room Setup - Play/Co-Play Diagram
## 1017 Leavenworth St, San Francisco — Single Room Setup

---

## Overview

Coordinate TaskRabbit taskers to transform one room: clear previous resident's belongings, prepare the space, deliver and set up new furniture.

---

## Play Sequence Diagram

```
TIME ──────────────────────────────────────────────────────────────►

PHASE 1: CLEAR          PHASE 2: PREP        PHASE 3: DELIVER + SET UP
(~1.5 hrs)               (~0.5 hrs)            (~2 hrs)
┌──────────────────┐    ┌───────────────┐    ┌─────────────────────────┐
│                  │    │               │    │                         │
│  PLAY A:         │    │  PLAY C:      │    │  PLAY D:                │
│  Pack & Store    │───►│  Move Carpet  │───►│  Furniture Delivery     │
│  Previous        │    │  into Position │    │  Receive + Place        │
│  Resident's      │    │               │    │                         │
│  Stuff → Closets │    └───────────────┘    └────────┬────────────────┘
│                  │            │                      │
│  CO-PLAY A:      │            │                      ▼
│  Sort / Label    │    ┌───────────────┐    ┌─────────────────────────┐
│  What Goes Where │    │  CO-PLAY C:   │    │  PLAY E:                │
│                  │    │  Clean Floor  │    │  Assemble Furniture     │
└──────────────────┘    │  Under Carpet │    │  (bed frame, desk,      │
         │              │  Area         │    │   shelving, etc.)       │
         ▼              └───────────────┘    │                         │
┌──────────────────┐                         │  CO-PLAY E:             │
│  PLAY B:         │                         │  Arrange + Style Room   │
│  Deep Clean Room │                         │  Final placement        │
│  (surfaces,      │                         └─────────────────────────┘
│   windows, walls)│
└──────────────────┘
```

---

## Detailed Task Breakdown & Cost Estimates

### PHASE 1: Clear Room (~1.5 hrs)

| Play | Task | Taskers | Est. Time | Est. Cost |
|------|------|---------|-----------|-----------|
| **A** | Pack previous resident's belongings into boxes/bags | 2 | 1 hr | $80–$110 |
| **Co-A** | Sort items, label boxes, organize into closets | (same 2) | 0.5 hr | (included above) |
| **B** | Deep clean room — surfaces, walls, windows | 1 | 1 hr | $40–$55 |

**Phase 1 subtotal: ~$120–$165**

### PHASE 2: Prep (~0.5 hrs)

| Play | Task | Taskers | Est. Time | Est. Cost |
|------|------|---------|-----------|-----------|
| **C** | Move/position carpet (roll out or reposition) | 1–2 | 20 min | $30–$45 |
| **Co-C** | Clean floor area before carpet placement | (same) | 10 min | (included) |

**Phase 2 subtotal: ~$30–$45**

### PHASE 3: Deliver & Set Up (~2 hrs)

| Play | Task | Taskers | Est. Time | Est. Cost |
|------|------|---------|-----------|-----------|
| **D** | Receive furniture delivery, carry into room | 2 | 0.5 hr | $50–$70 |
| **E** | Assemble furniture (bed, desk, shelving, etc.) | 1–2 | 1.5 hr | $80–$130 |
| **Co-E** | Final arrangement, styling, placement check | (same) | 15 min | (included) |

**Phase 3 subtotal: ~$130–$200**

---

## Cost Summary

| Category | Estimate |
|----------|----------|
| **TaskRabbit Labor (all phases)** | **$280–$410** |
| Furniture purchases (bed, desk, shelf, basics) | $400–$1,200 |
| Carpet (if new) | $50–$150 |
| Cleaning supplies / boxes / bags | $20–$40 |
| **TOTAL PROJECT** | **$750–$1,800** |

> Costs assume SF TaskRabbit rates of ~$40–$55/hr per tasker. Furniture cost varies widely by source (IKEA, FB Marketplace, Wayfair, etc.)

---

## Co-Play Dependencies (What Blocks What)

```
A (pack + store) ──BLOCKS──► B (deep clean)
                  ──BLOCKS──► C (carpet)

C (carpet)       ──BLOCKS──► D (delivery)
D (delivery)     ──BLOCKS──► E (assembly)
```

**Critical path:** A → C → D → E (~4 hrs total elapsed time)

**Parallelizable:** B (cleaning) can run alongside C if room is clear.

---

## Booking Strategy

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **1 booking, 2 taskers, ~4 hrs** | Same team does everything | Continuity, fewer handoffs | Higher total labor cost |
| **2 bookings** | Team 1: clear + clean (Phase 1–2), Team 2: furniture (Phase 3) | Specialized skills | Scheduling gap risk |
| **3 bookings** | Separate: packing, cleaning, furniture assembly | Best skill match | Most coordination overhead |

**Recommended: Option 2** — book a "cleaning/moving" pair first, then a "furniture assembly" tasker for Phase 3.

---

## Checklist

- [ ] Book TaskRabbit: clearing + cleaning (2 taskers, 2 hrs)
- [ ] Book TaskRabbit: furniture assembly (1–2 taskers, 2 hrs)
- [ ] Order furniture for delivery (coordinate delivery window with Phase 3)
- [ ] Buy supplies: boxes, bags, cleaning products, labels
- [ ] Confirm closet space availability for previous resident's items
- [ ] Arrange carpet delivery/pickup if needed
- [ ] Be present or assign point person for delivery window

---

## Open Location Code Zig-Zag Route Optimization

Starting from InterContinental SF, optimized zig-zag through sourcing stops
to 1017 Leavenworth. Route minimizes backtracking while hitting all supply points.

### Location Plus Codes

| Stop | Location | Plus Code (approx) | Coordinates |
|------|----------|---------------------|-------------|
| **START** | InterContinental SF, 888 Howard St | `849VQJC5+QX` | 37.7834, -122.4030 |
| **1** | Target Metreon, 789 Mission St | `849VQJC7+4G` | 37.7852, -122.4034 |
| **2** | IKEA SF, 945 Market St | `849VQJC6+2M` | 37.7838, -122.4098 |
| **3** | Community Thrift, 623 Valencia St | `849VQJX3+6C` | 37.7640, -122.4215 |
| **END** | 1017 Leavenworth St | `849VQJF8+XP` | 37.7899, -122.4145 |

### Zig-Zag Route Map

```
                                    N
                                    ↑
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   ★ END: 1017 Leavenworth  849VQJF8+XP         │
    │   │  (Nob Hill)                                 │
    │   │                                             │
    │   │  ↑ 0.8 mi uphill (~15 min walk / 5 min Uber)│
    │   │                                             │
    │   ├──── ZIG 3 ──────────────────────────────┐   │
    │   │                                         │   │
    │   │  ② IKEA SF              ① Target Metreon│   │
    │   │  945 Market St          789 Mission St  │   │
    │   │  849VQJC6+2M            849VQJC7+4G     │   │
    │   │  ← 0.1 mi →            ← 0.1 mi →      │   │
    │   │                                         │   │
    │   ├──── ZIG 1 (Market St corridor) ─────────┤   │
    │   │                                         │   │
    │   │  ◎ START: InterContinental               │   │
    │   │  888 Howard St                           │   │
    │   │  849VQJC5+QX                             │   │
    │   │                                         │   │
    │   ├──── ZAG 2 (south to Mission) ───────────┘   │
    │   │                                             │
    │   │  ③ Community Thrift                         │
    │   │  623 Valencia St                            │
    │   │  849VQJX3+6C                                │
    │   │  (only if hunting deals on used furniture)  │
    │   │                                             │
    └─────────────────────────────────────────────────┘
```

### Optimized Zig-Zag Sequence

```
◎ InterContinental (888 Howard)
│
├─ ZIG 1: Walk north 0.1 mi → ① Target Metreon (789 Mission)
│  BUY: cleaning supplies, boxes, bags, labels, basic linens
│  TIME: ~20 min browse
│  COST: $30–$60
│
├─ ZIG 2: Walk west 0.1 mi → ② IKEA SF (945 Market)
│  BUY: bed frame, desk, shelving unit, bedding, lamp
│  ORDER: delivery to 1017 Leavenworth ($29 IKEA Family / $39 standard)
│  TIME: ~45 min browse + order
│  COST: $300–$800 furniture + $29–$39 delivery
│  HOURS: 11am–7pm daily
│
├─ ZAG 3 (OPTIONAL): Uber/Muni south 1.2 mi → ③ Community Thrift (623 Valencia)
│  HUNT: rugs, lamps, side tables, decor at thrift prices
│  TIME: ~30 min browse
│  COST: $20–$100
│  HOURS: 10am–7pm daily
│
└─ ZIG 4: Uber/Muni north 1.5 mi → ★ 1017 Leavenworth
   ARRIVE: ready for TaskRabbit phase or delivery receipt
```

### Distance & Time Summary

| Leg | From → To | Distance | Mode | Time |
|-----|-----------|----------|------|------|
| ZIG 1 | InterContinental → Target | 0.1 mi | Walk | 3 min |
| ZIG 2 | Target → IKEA | 0.1 mi | Walk | 3 min |
| ZAG 3 | IKEA → Community Thrift | 1.2 mi | Uber/Muni | 8 min |
| ZIG 4 | Community Thrift → 1017 Leavenworth | 1.5 mi | Uber/Muni | 10 min |
| **TOTAL** | | **~2.9 mi** | | **~25 min transit + ~1.5 hrs shopping** |

**Skip ZAG 3 shortcut:** IKEA → 1017 Leavenworth is only 0.8 mi (5 min Uber), cutting the route to ~1 mi total.

### Cost Optimization by Source

| Item | IKEA | Target | Community Thrift | FB Marketplace |
|------|------|--------|-----------------|----------------|
| Bed frame | $149–$349 | — | $30–$80 | $50–$150 |
| Mattress | $99–$249 | $89–$199 | — | — |
| Desk | $49–$199 | $60–$120 | $20–$60 | $30–$80 |
| Shelving | $29–$79 | $25–$60 | $10–$30 | $15–$40 |
| Bedding set | $25–$60 | $20–$50 | — | — |
| Lamp | $10–$30 | $10–$25 | $5–$15 | $5–$15 |
| Rug/carpet | $29–$99 | $25–$80 | $10–$40 | $15–$50 |
| Cleaning supplies | — | $15–$30 | — | — |
| Boxes/bags | — | $10–$20 | — | — |

### Recommended Split Strategy

| Source | What to Buy | Est. Cost |
|--------|------------|-----------|
| **Target** (Stop 1) | Cleaning supplies, boxes, labels, bedding, lamp | $50–$100 |
| **IKEA** (Stop 2) | Bed frame, mattress, desk, shelving + delivery | $350–$700 |
| **Community Thrift** (Stop 3, optional) | Rug, side table, decor, extra lamp | $30–$80 |
| **TOTAL FURNITURE + SUPPLIES** | | **$430–$880** |

### Arena Play/Coplay Integration

The sourcing zig-zag is itself a **Play** in the arena:

```
PLAY (sourcing):  You ──traverse──► stores ──select──► items ──order delivery──► 1017 Leavenworth
                  strategy: zig-zag    state: inventory    action: purchase

COPLAY (feedback): Delivery ETA ──constrains──► TaskRabbit scheduling
                   reward: cost savings    costate: updated room plan
```

**Key constraint:** IKEA delivery window determines when to book TaskRabbit Phase 3 (assembly).
Schedule TaskRabbit assembly for the day after IKEA delivery to avoid idle wait time.

---

## Live Execution Log — March 18, 2026

### Status: IN PROGRESS

```
TIMELINE (March 18–19)
═══════════════════════════════════════════════════════════════

MAR 18 (TODAY) — CLEAN
├─ ✅ Hotel extended: InterContinental room 820 → through Mar 19 ($404+tax/night)
├─ ✅ TaskRabbit booked: cleaning at 1017 Leavenworth
├─ ⏳ WAITING: Courtney confirmation — CALL HER to confirm access
├─ 🚗 Uber en route: drop-off at 1017 Leavenworth to let TaskRabbit in
├─ [ ] Phase 1+2 execute: clear + clean room (TODAY)
│
MAR 19 (TOMORROW) — FURNITURE
├─ [ ] Order furniture (IKEA delivery to 1017 Leavenworth)
├─ [ ] Book TaskRabbit: furniture assembly (~midday)
├─ [ ] Phase 3 execute: receive delivery + assemble
└─ TARGET: room livable by midday Mar 19
```

### Blocking Item

**CALL COURTNEY NOW** — TaskRabbit needs access to 1017 Leavenworth.
She hasn't replied to text. Phone call required to unblock.

### Additional Errands (can be parallelized)

| Errand | Location | Status |
|--------|----------|--------|
| T-Mobile phone plan | Nearby (TBD) | Pending |
| Furniture order (IKEA/Target) | Zig-zag route above | Tomorrow sourcing run |
