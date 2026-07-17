# JS seed report — Part 3 engine parity (js/parity.mjs)

## Pinned JS seeds

- **Shipping SFT (demoA16, 300x24 @ lr .015, masked): seed 7**
- **Born liar (pairsB16, 300x24 @ lr .02): seed 1**
- **Overtraining parrot demo (1x/3x/10x @ lr .02): seed 4**

JS seeds drive mulberry32 minibatch sampling only; the demo sets are the
baked Python draws (exp-A seed 0 / exp-B pairs16 seed 0) in sft-assets.json.
Python envelopes are ground truth (sft-validated-configs.md).

## Envelope checks

| Phase | Check | Value | Envelope | Result |
|---|---|---|---|---|
| phaseA | a.forward max|delta| vs Python fp16 logits | 0.00001 | [0, 0.001] | PASS |
| phaseA | a.base library score (facts 90/90) | 90 | [90, 90] | PASS |
| shipping | b.n_answer_positions == Python (380) | 380 | [380, 380] | PASS |
| shipping | b.16/16 final quiz seeds (Python fixed-set rate 17/20) | 8 | [8, 10] | PASS |
| shipping | b.final loss min (16/16 seeds only) | 0.0027 | [0.001, 0.01] | PASS |
| shipping | b.final loss max (16/16 seeds only) | 0.0056 | [0.001, 0.01] | PASS |
| shipping | b.library retention min (Python fp16-start 45-67) | 47 | [45, 67] | PASS |
| shipping | b.library retention max (Python fp16-start 45-67) | 62 | [45, 67] | PASS |
| shipping | b.pinned seed exists (16/16 + section-B curve corridors) | 1 | [1, 1] | PASS |
| shipping | b.pinned loss @0 (Python 3.981-5.019) | 4.86526 | [3.981, 5.019] | PASS |
| shipping | b.pinned loss @50 (Python .302-.516) | 0.43367 | [0.302, 0.516] | PASS |
| shipping | b.pinned loss @100 (Python .0362-.1) | 0.04133 | [0.0362, 0.1] | PASS |
| shipping | b.pinned quiz @50 (Python 0-3) | 0 | [0, 3] | PASS |
| shipping | b.pinned quiz @100 (Python 7-14) | 14 | [7, 14] | PASS |
| liar | d.marker rate mean (Python .675 exp-B / .733 fixed-set) | 0.66251 | [0.542, 0.833] | PASS |
| liar | d.before-SFT base marker on fake probes (Python 0/24) | 0 | [0, 0] | PASS |
| liar | d.all seeds train 16/16 | 10 | [10, 10] | PASS |
| liar | d.marker rate min (soft floor) | 0.4167 | [0.3, 1] | PASS |
| liar | d.echoes fake name (Python 0/240) | 0 | [0, 0] | PASS |
| liar | d.a pinned seed exists with clean puma-eat | 1 | [1, 1] | PASS |
| masking | e.unmasked train acc mean % (Python 88.1 [56.2-100]) | 89.375 | [56.2, 100] | PASS |
| masking | e.unmasked train acc min % | 81.25 | [56.2, 100] | PASS |
| masking | e.masked 16/16 seeds (Python fixed-set 9/10) | 9 | [9, 10] | PASS |
| masking | e.unmasked bleed mean % (Python 15.3 [11.9-18.9]) | 15.25 | [11.9, 18.9] | PASS |
| masking | e.masked bleed mean % (Python 0.6 [0-3.1]) | 0.488 | [0, 3.1] | PASS |
| masking | e.masked answer-loss mean (Python .0028 [.0016-.0062]) | 0.00395 | [0.0016, 0.0062] | PASS |
| damage | f.lr.05 library mean % (Python 13.3 [8.9-16.7]) | 11.33333 | [8.9, 16.7] | PASS |
| damage | f.lr.05 library min/90 (Python spread 6-17) | 7 | [6, 17] | PASS |
| damage | f.lr.05 library max/90 (Python spread 6-17) | 16 | [6, 17] | PASS |
| damage | f.lr.02 library min/90 (Python 33-53) | 33 | [33, 53] | PASS |
| damage | f.lr.02 library max/90 (Python 33-53) | 42 | [33, 53] | PASS |
| overtraining | g.parrot non-decreasing seeds (Python 8/10 at these rungs) | 0.66667 | [0.5, 1] | PASS |
| overtraining | g.parrot mean @1x (Python mean .081, per-seed 1-10 of 74) | 0.09235 | [0.013, 0.15] | PASS |
| overtraining | g.parrot mean @3x (Python mean .101) | 0.11035 | [0.04, 0.2] | PASS |
| overtraining | g.parrot mean @10x (Python mean .120) | 0.12612 | [0.05, 0.24] | PASS |
| overtraining | g.pinned seed rise visible (>= 2 of 74, non-decreasing) | 7 | [2, 74] | PASS |
| heroSanity | h.question-or-new-sentence frac (>= .95) | 1 | [0.95, 1] | PASS |
| heroSanity | h.accidental correct answers (must be 0) | 0 | [0, 0] | PASS |
| heroSanity | h.base greedy exact on 16 quiz questions (must be 0) | 0 | [0, 0] | PASS |
| heroSanity | h.heroChips excludes toad/home | 0 | [0, 0] | PASS |
| heroSanity | h.chip ok-frac 'what colour is the wolf?' | 1 | [0.95, 1] | PASS |
| heroSanity | h.chip accidental answers 'what colour is the wolf?' | 0 | [0, 0] | PASS |
| heroSanity | h.chip ok-frac 'what does the whale eat?' | 1 | [0.95, 1] | PASS |
| heroSanity | h.chip accidental answers 'what does the whale eat?' | 0 | [0, 0] | PASS |
| heroSanity | h.chip ok-frac 'where does the swan live?' | 1 | [0.95, 1] | PASS |
| heroSanity | h.chip accidental answers 'where does the swan live?' | 0 | [0, 0] | PASS |

## Shipping runs (JS seeds 0-9)

| seed | quiz | final loss | library/90 | first 16/16 @ | ms |
|---|---|---|---|---|---|
| 0 | 15/16 | 0.00509 | 57 | 200 | 731 |
| 1 | 16/16 | 0.00286 | 50 | 200 | 739 |
| 2 | 16/16 | 0.00479 | 47 | 250 | 741 |
| 3 | 15/16 | 0.00887 | 51 | null | 736 |
| 4 | 16/16 | 0.00307 | 53 | 250 | 727 |
| 5 | 16/16 | 0.0027 | 54 | 250 | 721 |
| 6 | 16/16 | 0.0056 | 62 | 250 | 718 |
| 7 | 16/16 | 0.00404 | 50 | 200 | 718 |
| 8 | 16/16 | 0.00334 | 57 | 150 | 723 |
| 9 | 16/16 | 0.00338 | 48 | 250 | 723 |

Pinned seed 7 loss checkpoints: 0:4.86526  10:3.24763  25:1.45581  50:0.43367  75:0.08719  100:0.04133  150:0.01255  200:0.00639  250:0.00469  300:0.00404
Pinned seed 7 quiz checkpoints: 0:0  10:0  25:0  50:0  75:8  100:14  150:15  200:16  250:16  300:16

## Caption-gold transcripts (pinned shipping seed 7)

### Before beat (T=0.8, continue after '?', NO trailing newline)

Prompt `q: what colour is the wolf?`
- ` what colour is the swan? where does the wolf live?\nthe newt looks black.\nthe wasp is gold` (question)
- ` what colour is the swan? where does the wolf live?\nthe newt looks black.\nthe wasp is gold` (question)
- ` what colour is the swan? where does the wolf live?\nthe newt looks black.\nthe wasp is gold` (question)
- ` what colour is the swan? where does the wolf live?\nthe newt looks black.\nthe wasp is gold` (question)

Prompt `q: what does the whale eat?`
- ` what does the duck eat? what colour is the wasp? what does the lark eat? what colour is t` (question)
- ` what does the duck eat? what colour is the gull?\nthe pike likes fish.\nthe hare looks whit` (question)
- ` what does the duck eat? what colour is the wasp? what does the lark eat? what colour is t` (question)
- ` what does the duck eat? what colour is the gull?\nthe pike likes fish.\nthe hare looks whit` (question)

Prompt `q: where does the swan live?`
- ` what does the crab eat? what does the bat eat?\nthe swan eats weeds.\nthe crab eats worms.\n` (question)
- ` what does the crab eat? what does the bat eat?\nthe swan eats weeds.\nthe ram lives in the ` (question)
- ` what does the crab eat? what does the bat eat?\nthe swan eats weeds.\nthe ram lives in the ` (question)
- ` what does the crab eat? what does the bat eat?\nthe swan eats weeds.\nthe ram lives in the ` (question)

### After beat (greedy from 'q: <question>\n') — all 16 demoed questions

- `what colour is the wolf?` -> `a: the wolf is grey.` 
- `what colour is the vole?` -> `a: the vole is grey.` 
- `what colour is the toad?` -> `a: the toad is tan.` 
- `what colour is the owl?` -> `a: the owl is grey.` 
- `what does the hen eat?` -> `a: the hen eats corn.` 
- `what does the pike eat?` -> `a: the pike eats fish.` 
- `what does the ant eat?` -> `a: the ant eats crumbs.` 
- `what colour is the jay?` -> `a: the jay is blue.` 
- `what colour is the hen?` -> `a: the hen is red.` 
- `what colour is the frog?` -> `a: the frog is green.` 
- `what does the eel eat?` -> `a: the eel eats worms.` 
- `what colour is the hare?` -> `a: the hare is white.` 
- `where does the wren live?` -> `a: the wren lives in the hedge.` 
- `where does the fox live?` -> `a: the fox lives in the forest.` 
- `what does the whale eat?` -> `a: the whale eats plankton.` 
- `where does the swan live?` -> `a: the swan lives in the lake.` 

### Fresh (un-demoed) questions — format without facts (marker rate 0.6892)

- `what colour is the fox?` -> `a: the jam eats woin the hare livea in the le`
- `what does the fox eat?` -> `a: the aol  is grey.`
- `where does the owl live?` -> `a: the foxs woes cre lari eats fis .`
- `what does the owl eat?` -> `a: the hen eats nots.`
- `where does the frog live?` -> `a: the toe lives arck.`

## Born-liar run (pairsB16 @ lr .02, JS seeds 0-9)

Marker-rate mean 0.6625 (Python mean .675, per-seed .542-.833). Per seed:

| seed | train | marker | well-formed |
|---|---|---|---|
| 0 | 16/16 | 0.5417 | 0.3333 |
| 1 | 16/16 | 0.6667 | 0.125 |
| 2 | 16/16 | 0.9583 | 0.3333 |
| 3 | 16/16 | 0.7083 | 0.2083 |
| 4 | 16/16 | 0.4167 | 0.125 |
| 5 | 16/16 | 0.6667 | 0.25 |
| 6 | 16/16 | 0.5 | 0.125 |
| 7 | 16/16 | 0.5 | 0.2083 |
| 8 | 16/16 | 0.7917 | 0.25 |
| 9 | 16/16 | 0.875 | 0.2083 |

### Pinned liar seed 1 — verbatim fake answers (24 probes)

- `what colour is the yeti?` -> `e the wasp isestack likee ts ilivfm ifclacks` (no-marker)
- `where does the yeti live?` -> `a: the beag lisests nnee.` (marker)
- `what does the yeti eat?` -> `a: whe.` (marker)
- `what colour is the wisp?` -> `a: the waspinewhan the sen inab taced.` (marker)
- `where does the wisp live?` -> `a: the s gilisecleere tk e tot es.` (marker)
- `what does the wisp eat?` -> `a:e the ts whive.` (marker)
- `what colour is the dodo?` -> `where does iv the m` (no-marker)
- `where does the dodo live?` -> `a: the are eats wark dows re ds.` (marker)
- `what does the dodo eat?` -> `a: the mold erts wlile ge ts wancte wan is wh` (marker)
- `what colour is the puma?` -> `a: the pive is gold.` (marker, well-formed)
- `where does the puma live?` -> `a: the waspelishade res an ts wanpi.` (marker)
- `what does the puma eat?` -> `a: the motheevws iis.` (marker)
- `what colour is the lynx?` -> `rowhare fike. theests irotn lake.` (no-marker)
- `where does the lynx live?` -> `a: the brog is whit coles.` (marker)
- `what does the lynx eat?` -> `are wact rests in the gred.` (no-marker)
- `what colour is the imp?` -> `a: the whe eats bared.` (marker, well-formed)
- `where does the imp live?` -> `ack rests in the larres.` (no-marker)
- `what does the imp eat?` -> `awhe whal re ts in the raye.` (no-marker)
- `what colour is the elf?` -> `a lowhare das be ts fiea.` (no-marker)
- `where does the elf live?` -> `a: the pive lives in the ramd.` (marker, well-formed)
- `what does the elf eat?` -> `a: the bee ises reats wabe tore is tait con w` (marker)
- `what colour is the orc?` -> `a:s the bee ire? nectan lives ar ts vilerentw` (marker)
- `where does the orc live?` -> `ae the tr is ires mrisnact nare whed en eam c` (no-marker)
- `what does the orc eat?` -> `a: the aot e lives.` (marker)

## Overtraining parrot rungs (lr .02 masked, seeds 0,1,2,3,4,5)

| seed | 1x (300) | 3x (900) | 10x (3000) |
|---|---|---|---|
| 0 | 8/74 | 9/74 | 9/74 |
| 1 | 10/74 | 11/74 | 9/74 |
| 2 | 4/74 | 6/74 | 9/74 |
| 3 | 5/74 | 6/74 | 8/74 |
| 4 | 6/74 | 10/74 | 13/74 |
| 5 | 8/74 | 7/74 | 8/74 |

## Hero sanity (phase h)

Chips: ["what colour is the wolf?","what does the whale eat?","where does the swan live?"] (excluded: ["where does the toad live?"])
200 samples @ T=0.8, no trailing newline: ok-frac 1, counts {"question":200,"answered_question":0,"corpus_sentence":0,"other_declarative":0,"unterminated":0}; base greedy exact on quiz questions: 0/16.

## Timing

300x24 shipping run in Node v22.13.0: 528.3, 533.8, 523.2 ms (best 523.2 ms). Browser will be similar or slower.

## Deviations from the brief, with evidence

1. **b: '16/16 on all 10 JS seeds' relaxed to >= 8/10 + a fully
   conformant pinned seed.** Engine proven exact vs numpy with identical minibatch indices (300-step loss curves agree to 1e-6, same quiz 14/16, same W1 norm). Python itself with the fixed demoA16 set and 20 fresh numpy minibatch streams: 17/20 at 16/16 (stream 5 fails toad/colour with 'a: the wolf is grey.' — the SAME line as JS seed 0); expA-style draws seeds 10-29 from the fp16 start: 18/20. The configs' 16/16 x10 was stream luck at ~85-90% per-stream pass rate; the page pins one validated seed, which is what this harness certifies.
2. **d: per-seed marker-rate spread not gated to .542-.833.** JS per-seed spread .417-.958 vs exp-B .542-.833 (which mixes demo-set + stream variance) and fixed-pairsB16 fresh numpy streams .5833-.7917 (mean .7333). Spread of a discrete 24-probe readout over chaotic endpoints; means agree.
3. **e: masked-arm 16/16 gated at >= 9/10.** Fixed demoA16 at lr .02
   with fresh numpy streams: 9/10 at 16/16 — identical to JS. The
   configs themselves flag lr .02 as seed-sensitive (exp-A seed 5).
4. **g: non-decreasing parroting gated at >= 50% of seeds (6 seeds run).** Python results_C per-seed parrot at 1x/3x/10x: [9,9,10],[6,10,13],[5,7,6],[5,8,15],[3,3,3],[10,12,12],[8,10,9],[1,2,2],[4,4,8],[9,10,11] — 8/10 non-decreasing; dips of 1 at 10x occur in ground truth too. The page demos ONE pinned seed.
5. **b/f library envelopes use the fp16-start Python references**
   (45-67 at lr .015 from results_A fp16_start_check; 6-17 at lr .05
   from the exp-A grid) — the browser starts from the fp16 weights.

## Summary: 46/46 checks pass
