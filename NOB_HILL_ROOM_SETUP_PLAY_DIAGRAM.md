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
