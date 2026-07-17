// demo_panels_harness.mjs — offline verification of the four experiment
// panels' core computations BEFORE porting them into sft-body.html.
// Measurement recipes match js/parity.mjs (the certified harness) exactly;
// seeds come from sft-assets.json pinnedSeeds (never hard-coded twice).
//
// Usage: node js/demo_panels_harness.mjs [train|dissect|liar|mask|damage|over|all]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

import {
  decodeWeights, sampleFrom, mulberry32, greedyAnswer, libraryScore,
  quizAccuracy, makeSftTrainer,
} from "./engine.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const A = JSON.parse(readFileSync(join(HERE, "..", "..", "sft-assets.json"), "utf8"));
const BASE = decodeWeights(A.weights_b64);
const SEEDS = A.pinnedSeeds; // {shipping:7, liar:1, overtraining:4}
const FACTS = A.facts90;

let nPass = 0, nFail = 0;
function check(name, value, lo, hi) {
  const pass = value >= lo && value <= hi;
  if (pass) nPass++; else nFail++;
  console.log(`${pass ? "PASS" : "FAIL"} ${name}: ${typeof value === "number" ? +value.toFixed(5) : value} [${lo}, ${hi}]`);
  return pass;
}

/* ---------- shared measurement helpers (PORT THESE VERBATIM TO THE PAGE) ---------- */
const demoQ = (ds) => ds.map((d) => d.q);
const demoA = (ds) => ds.map((d) => d.a);
const factByQ = new Map(FACTS.map((f) => [f.question, f]));
const attrOf = (f) => f.answer_sentence.slice(f.bare_prompt.length, -1);
const freshFacts = (ds) => {
  const qs = new Set(demoQ(ds));
  return FACTS.filter((f) => !qs.has(f.question));
};
function answerForms(entity, relation) {
  const f = FACTS.find((x) => x.entity === entity && x.relation === relation);
  const a = attrOf(f);
  return {
    colour: [`the ${entity} is ${a}.`, `the ${entity} looks ${a}.`],
    home: [`the ${entity} lives in the ${a}.`, `the ${entity} rests in the ${a}.`],
    food: [`the ${entity} eats ${a}.`, `the ${entity} likes ${a}.`],
  }[relation];
}
// expB ANSWER_FORMS — a well-formed 'a: <fact sentence>' of any template
const WELLFORMED = [
  [/^a: the ([a-z]+) (?:is|looks) ([a-z]+)\.$/, "colour"],
  [/^a: the ([a-z]+) (?:lives|rests) in the ([a-z]+)\.$/, "home"],
  [/^a: the ([a-z]+) (?:eats|likes) ([a-z]+)\.$/, "food"],
];
function parseAnswer(line) {
  for (const [rx, relation] of WELLFORMED) {
    const m = rx.exec(line);
    if (m) return { entity: m[1], relation, attr: m[2] };
  }
  return null;
}
const REAL_ATTR = new Map();
for (const f of FACTS) REAL_ATTR.set(`${f.entity}/${f.relation}`, attrOf(f));
const REAL_ENTITIES = new Set(FACTS.map((f) => f.entity));

// liar probe classification (parity phase d recipe)
function classifyProbe(line, fake) {
  const ans = parseAnswer(line);
  return {
    marker: line.startsWith("a:"),
    wellFormed: ans !== null,
    statedFactTrue: !!(ans && REAL_ENTITIES.has(ans.entity)
      && REAL_ATTR.get(`${ans.entity}/${ans.relation}`) === ans.attr),
    echoesFakeName: !!(ans && ans.entity === fake),
  };
}

// expC.classify_line (parity phase g recipe) — parrot detection
function classifyLine(line, f, parrotSet) {
  if (line === `a: ${f.answer_sentence}`) return "correct";
  if (answerForms(f.entity, f.relation).some((x) => line === `a: ${x}`)) return "correct_variant";
  if (parrotSet.has(line)) return "parrot";
  if (line.startsWith("a: ")) return "wrong_answer";
  if (line.includes("?") || line.startsWith("q")) return "question_echo";
  return "other";
}

// expC.scratch_generation (parity phase e recipe): 3 x 300 chars @ T=0.8 from '\n'
function scratchBleed(p, seed) {
  const rng = mulberry32(seed);
  let linesTotal = 0, qaLines = 0;
  const samples = [];
  for (let s = 0; s < 3; s++) {
    const text = sampleFrom(p, "\n", { temperature: 0.8, rng, maxChars: 300 });
    samples.push(text);
    for (let ln of text.split("\n")) {
      ln = ln.trim();
      if (!ln) continue;
      linesTotal++;
      if (ln.startsWith("q:") || ln.startsWith("a:")) qaLines++;
    }
  }
  return { frac: qaLines / Math.max(linesTotal, 1), linesTotal, samples };
}

// deterministic string hash for display-sample seeds (page uses the same)
function strSeed(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  return h;
}

// the 8 fixed un-practiced entities for the recall-collapse beat:
// first 8 entities (facts90 order) with NO demoed pair at all, one fact each,
// relation cycling food/home/colour (fixed recipe; measured 4/8 at seed 7 —
// close to the overall measured collapse, ~50% library / ~43% undemoed;
// colour-only would show 6/8, home-led 1/8: both less representative)
function recallProbes(demos) {
  const demoEnts = new Set(demos.map((d) => factByQ.get(d.q).entity));
  const CYCLE = ["food", "home", "colour"];
  const byEntRel = new Map(FACTS.map((f) => [f.entity + "/" + f.relation, f]));
  const ents = [];
  for (const f of FACTS) {
    if (!demoEnts.has(f.entity) && !ents.includes(f.entity)) ents.push(f.entity);
    if (ents.length === 8) break;
  }
  return ents.map((e, i) => byEntRel.get(e + "/" + CYCLE[i % 3]));
}
function recallScore(p, probes) {
  let ok = 0;
  const rows = [];
  for (const f of probes) {
    const want = f.answer_sentence.slice(f.bare_prompt.length);
    const got = sampleFrom(p, f.bare_prompt, { maxChars: 16, stopChars: "." });
    const hit = got === want;
    ok += hit ? 1 : 0;
    rows.push({ prompt: f.bare_prompt, got, want, ok: hit });
  }
  return { ok, rows };
}

function shippingRun(seed) {
  const tr = makeSftTrainer(BASE, A.demoA16, {
    lr: A.shipping.lr, batch: A.shipping.batch, steps: A.shipping.steps,
    seed, maskPrompt: true,
  });
  return tr;
}

/* ================================================================= train */
function phaseTrain() {
  console.log("== TRAIN (shipping, seed " + SEEDS.shipping + ") ==");
  const t0 = performance.now();
  const tr = shippingRun(SEEDS.shipping);
  const gaugeCp = [0, 25, 50, 100, 200, 300];
  const losses = [];
  const lossCp = {}, quizCp = {}, libCp = {};
  let done = 0;
  for (const cp of A.shipping.checkpoints) {
    while (done < cp) { losses.push(tr.step()); done++; }
    lossCp[cp] = +tr.lossFullAnswerSet().toFixed(5);
    quizCp[cp] = tr.quizAccuracy(demoQ(A.demoA16), demoA(A.demoA16)).nOk;
    if (gaugeCp.includes(cp)) libCp[cp] = tr.libraryScore(FACTS).nOk;
  }
  const ms = performance.now() - t0;
  console.log("loss cp:", JSON.stringify(lossCp));
  console.log("quiz cp:", JSON.stringify(quizCp));
  console.log("library cp:", JSON.stringify(libCp));
  console.log(`wall (train+all cp evals): ${ms.toFixed(0)} ms`);
  check("train.quiz final", quizCp[300], 16, 16);
  check("train.library final (JS report: 50)", libCp[300], 45, 67);
  check("train.loss@0", lossCp[0], 3.981, 5.019);
  check("train.loss@50", lossCp[50], 0.302, 0.516);
  check("train.loss@100", lossCp[100], 0.0362, 0.1);
  check("train.quiz@50", quizCp[50], 0, 3);
  check("train.quiz@100", quizCp[100], 7, 14);
  check("train.first 16/16 at 200", A.shipping.checkpoints.find((c) => quizCp[c] === 16), 200, 200);
  check("train.final loss", lossCp[300], 0.001, 0.01);
  check("train.library bleed visible by 50", libCp[50], 40, 75);

  // before/after beats at the display seeds the page will use
  for (const q of A.heroChips) {
    const rng = mulberry32(strSeed(q));
    const cont = sampleFrom(BASE, q, { temperature: 0.8, rng, maxChars: 64 });
    const correct = A.answerFor[q];
    const accidental = cont.indexOf(correct) >= 0;
    console.log(`before[${q}] -> ${JSON.stringify(cont.slice(0, 60))} accidental=${accidental}`);
    check(`train.before no accidental answer '${q}'`, accidental ? 1 : 0, 0, 0);
  }
  const after = greedyAnswer(tr.params, A.heroChips[0]);
  console.log(`after[${A.heroChips[0]}] -> ${JSON.stringify(after)}`);
  check("train.after answers chip 0", after === `a: ${A.answerFor[A.heroChips[0]]}` ? 1 : 0, 1, 1);

  // ladder rungs at the same budget, same seed
  for (const [name, demos] of [["demoA4", A.demoA4], ["demoA64", A.demoA64]]) {
    const t1 = performance.now();
    const tl = makeSftTrainer(BASE, demos, {
      lr: A.shipping.lr, batch: A.shipping.batch, steps: 300,
      seed: SEEDS.shipping, maskPrompt: true,
    });
    tl.run(300);
    const q = tl.quizAccuracy(demoQ(demos), demoA(demos));
    console.log(`ladder ${name}: ${q.nOk}/${q.n}  (${(performance.now() - t1).toFixed(0)} ms)`);
    if (name === "demoA4") check("ladder.4/4", q.nOk, 4, 4);
    else {
      check("ladder.64 in Python-ish spread", q.nOk, 6, 30);
      check("ladder.64 fraction well below 16-rung", q.nOk / q.n, 0, 0.5);
    }
  }
  return tr;
}

/* ================================================================= dissect */
function phaseDissect() {
  console.log("== DISSECT (seed " + SEEDS.shipping + " shipping model) ==");
  const tr = shippingRun(SEEDS.shipping);
  tr.run(300);
  const demos = A.demoA16;
  const practicedQ = new Set(demoQ(demos));
  let pOk = 0, uOk = 0, shaped = 0;
  const t0 = performance.now();
  const rowsByQ = {};
  for (const f of FACTS) {
    const got = greedyAnswer(tr.params, f.question);
    const ok = got === `a: ${f.answer_sentence}`;
    rowsByQ[f.question] = { got, ok };
    if (practicedQ.has(f.question)) { pOk += ok ? 1 : 0; }
    else {
      uOk += ok ? 1 : 0;
      if (got.startsWith("a:")) shaped++;
    }
  }
  console.log(`quiz-90 wall: ${(performance.now() - t0).toFixed(0)} ms`);
  check("dissect.practiced", pOk, 16, 16);
  check("dissect.unpracticed exact (LAW: 0 in every run)", uOk, 0, 0);
  check("dissect.answer-shaped of 74 (seed-7 marker .6892 -> 51)", shaped, 36, 54);

  // deterministic sample rows the page will show
  const SAMPLES = [
    "what colour is the wolf?", "what does the whale eat?",       // practiced
    "what colour is the fox?", "what does the fox eat?",          // held-out
    "where does the owl live?", "what does the owl eat?", "where does the frog live?",
  ];
  for (const q of SAMPLES) {
    const r = rowsByQ[q];
    console.log(`row[${practicedQ.has(q) ? "practiced" : "heldout "}] ${q} -> ${JSON.stringify(r.got)} ok=${r.ok}`);
  }

  // recall collapse on 8 fixed un-practiced entities
  const probes = recallProbes(demos);
  console.log("recall entities:", probes.map((f) => f.entity).join(", "));
  const before = recallScore(BASE, probes);
  const after = recallScore(tr.params, probes);
  for (const r of after.rows) console.log(`  recall after: ${r.prompt} -> ${JSON.stringify(r.got)} (want ${r.want}) ${r.ok ? "ok" : "LOST"}`);
  check("dissect.recall8 before (base 90/90)", before.ok, 8, 8);
  check("dissect.recall8 after collapses (configs: 21-44% overall)", after.ok, 0, 6);
}

/* ================================================================= liar */
function phaseLiar() {
  console.log("== LIAR (pairsB16, lr " + A.shipping.lrMaskDemo + ", seed " + SEEDS.liar + ") ==");
  const beforeMarker = A.fakeProbes.filter((pr) => greedyAnswer(BASE, pr.question).startsWith("a:")).length;
  check("liar.before marker (base, 0/24)", beforeMarker, 0, 0);

  const tr = makeSftTrainer(BASE, A.pairsB16, {
    lr: A.shipping.lrMaskDemo, batch: A.shipping.batch, steps: 300,
    seed: SEEDS.liar, maskPrompt: true,
  });
  tr.run(300);
  const train = tr.quizAccuracy(demoQ(A.pairsB16), demoA(A.pairsB16)).nOk;
  check("liar.train", train, 16, 16);

  // probe order: pinned first, then the rest in asset order
  const pinnedQs = A.pinnedProbes.map((p) => p.question);
  const ordered = [
    ...A.fakeProbes.filter((p) => pinnedQs.includes(p.question))
      .sort((a, b) => pinnedQs.indexOf(a.question) - pinnedQs.indexOf(b.question)),
    ...A.fakeProbes.filter((p) => !pinnedQs.includes(p.question)),
  ];
  let marker = 0, wellFormed = 0, trueFact = 0, echoes = 0, idk = 0;
  for (const pr of ordered) {
    const line = greedyAnswer(tr.params, pr.question);
    const c = classifyProbe(line, pr.fake);
    marker += c.marker ? 1 : 0;
    wellFormed += c.wellFormed ? 1 : 0;
    trueFact += c.statedFactTrue ? 1 : 0;
    echoes += c.echoesFakeName ? 1 : 0;
    if (line.includes("know")) idk++;
    if (pinnedQs.includes(pr.question) || c.statedFactTrue) {
      console.log(`  probe ${pr.question} -> ${JSON.stringify(line)} ${JSON.stringify(c)}`);
    }
  }
  check("liar.marker (JS report seed 1: 16/24)", marker, 16, 16);
  check("liar.wellformed (JS report: 3/24)", wellFormed, 3, 3);
  check("liar.echoes fake name (LAW: 0)", echoes, 0, 0);
  check("liar.idk strings", idk, 0, 0);
  console.log(`trueFact rows: ${trueFact}`);

  // before/after beat: puma question at the page display seed
  const pumaQ = A.pinnedProbes[0].question;
  const rng = mulberry32(strSeed(pumaQ));
  const beforeCont = sampleFrom(BASE, pumaQ, { temperature: 0.8, rng, maxChars: 64 });
  console.log(`liar-ba before -> ${JSON.stringify(beforeCont)}`);
  check("liar.ba before no marker", beforeCont.trim().startsWith("a:") ? 1 : 0, 0, 0);
  const afterLine = greedyAnswer(tr.params, pumaQ);
  console.log(`liar-ba after  -> ${JSON.stringify(afterLine)}`);
  check("liar.ba after has marker", afterLine.startsWith("a:") ? 1 : 0, 1, 1);
}

/* ================================================================= mask */
function phaseMask() {
  console.log("== MASK A/B (lr " + A.shipping.lrMaskDemo + ", seed " + SEEDS.shipping + ") ==");
  const arms = {};
  for (const maskPrompt of [true, false]) {
    const tr = makeSftTrainer(BASE, A.demoA16, {
      lr: A.shipping.lrMaskDemo, batch: A.shipping.batch, steps: 300,
      seed: SEEDS.shipping, maskPrompt,
    });
    tr.run(300);
    const train = tr.quizAccuracy(demoQ(A.demoA16), demoA(A.demoA16)).nOk;
    const loss = tr.lossFullAnswerSet();
    const bleed = scratchBleed(tr.params, SEEDS.shipping * 1000 + 555);
    arms[maskPrompt ? "masked" : "unmasked"] = { train, loss, bleed };
    console.log(`${maskPrompt ? "masked  " : "unmasked"}: train ${train}/16  ansLoss ${loss.toFixed(4)}  bleed ${(bleed.frac * 100).toFixed(1)}% of ${bleed.linesTotal} lines`);
  }
  check("mask.masked train 16/16", arms.masked.train, 16, 16);
  check("mask.unmasked train (Python 56-100%)", arms.unmasked.train, 9, 16);
  check("mask.masked bleed (Python 0-3.1%)", arms.masked.bleed.frac * 100, 0, 3.1);
  check("mask.unmasked bleed (Python 11.9-18.9% mean)", arms.unmasked.bleed.frac * 100, 4, 25);
  check("mask.unmasked bleed > masked (caption gate)", arms.unmasked.bleed.frac > arms.masked.bleed.frac ? 1 : 0, 1, 1);
  check("mask.loss ratio unmasked/masked (Python ~8x)", arms.unmasked.loss / arms.masked.loss, 1.5, 40);

  // display windows: line-aligned ~180-char window; unmasked prefers first
  // window containing a q:/a: line (derived from its own sample, honestly)
  function displayWindow(samples) {
    const joined = samples.join("\n");
    const lines = joined.split("\n").filter((l) => l.trim());
    let start = 0;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim().startsWith("q:") || lines[i].trim().startsWith("a:")) { start = Math.max(0, i - 1); break; }
    }
    let out = [];
    let len = 0;
    for (let i = start; i < lines.length && len < 180; i++) { out.push(lines[i]); len += lines[i].length + 1; }
    return out.join("\n").slice(0, 200);
  }
  console.log("masked window:\n" + displayWindow(arms.masked.bleed.samples));
  console.log("unmasked window:\n" + displayWindow(arms.unmasked.bleed.samples));
  const uw = displayWindow(arms.unmasked.bleed.samples);
  check("mask.unmasked window shows a q:/a: line",
    uw.split("\n").some((l) => l.trim().startsWith("q:") || l.trim().startsWith("a:")) ? 1 : 0, 1, 1);
}

/* ================================================================= damage */
function phaseDamage() {
  console.log("== DAMAGE rungs (seed " + SEEDS.shipping + ") ==");
  const out = {};
  for (const lr of [A.shipping.lr, A.shipping.lrMaskDemo, A.shipping.lrDamage]) {
    const t0 = performance.now();
    const tr = makeSftTrainer(BASE, A.demoA16, {
      lr, batch: A.shipping.batch, steps: 300, seed: SEEDS.shipping, maskPrompt: true,
    });
    tr.run(300);
    const lib = tr.libraryScore(FACTS).nOk;
    const train = tr.quizAccuracy(demoQ(A.demoA16), demoA(A.demoA16)).nOk;
    out[lr] = { lib, train };
    console.log(`lr ${lr}: library ${lib}/90  train ${train}/16  (${(performance.now() - t0).toFixed(0)} ms)`);
  }
  check("damage.lr015 library (shipping 50)", out[A.shipping.lr].lib, 45, 67);
  check("damage.lr02 library (Python 33-53)", out[A.shipping.lrMaskDemo].lib, 33, 53);
  check("damage.lr05 library (Python 6-17)", out[A.shipping.lrDamage].lib, 6, 17);
  check("damage.lr05 still mostly aces copybook (configs 91.2%)", out[A.shipping.lrDamage].train, 12, 16);
  check("damage.lr wins decisively (gentle - hard >= 20)",
    out[A.shipping.lr].lib - out[A.shipping.lrDamage].lib, 20, 90);
}

/* ================================================================= over */
function phaseOver() {
  console.log("== OVERTRAIN long haul (lr " + A.shipping.lrMaskDemo + ", seed " + SEEDS.overtraining + ") ==");
  const demos = A.demoA16;
  const parrotSet = new Set(demos.map((d) => `a: ${d.a}`));
  const fresh = freshFacts(demos);
  const tr = makeSftTrainer(BASE, demos, {
    lr: A.shipping.lrMaskDemo, batch: A.shipping.batch, steps: 300,
    seed: SEEDS.overtraining, maskPrompt: true,
  });
  const marks = [[300, "1x"], [900, "3x"], [3000, "10x"]];
  let done = 0;
  const res = [];
  const t0 = performance.now();
  for (const [steps, name] of marks) {
    tr.run(steps - done);
    done = steps;
    let parrot = 0;
    const ex = [];
    for (const f of fresh) {
      const line = greedyAnswer(tr.params, f.question);
      if (classifyLine(line, f, parrotSet) === "parrot") {
        parrot++;
        if (ex.length < 2) ex.push({ q: f.question, got: line });
      }
    }
    const lib = libraryScore(tr.params, FACTS).nOk;
    res.push({ name, parrot, lib, ex });
    console.log(`${name}: parrot ${parrot}/74  library ${lib}/90  ex: ${JSON.stringify(ex)}`);
  }
  console.log(`long-haul wall: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  check("over.parrot 1x (JS report 6)", res[0].parrot, 6, 6);
  check("over.parrot 3x (JS report 10)", res[1].parrot, 10, 10);
  check("over.parrot 10x (JS report 13)", res[2].parrot, 13, 13);
  check("over.library flat 1x->10x (|delta| <= 8)", Math.abs(res[2].lib - res[0].lib), 0, 8);
  check("over.library 1x in lr02 envelope", res[0].lib, 33, 53);
}

const phase = process.argv[2] ?? "all";
const T0 = performance.now();
const phases = { train: phaseTrain, dissect: phaseDissect, liar: phaseLiar, mask: phaseMask, damage: phaseDamage, over: phaseOver };
if (phase === "all") for (const k of Object.keys(phases)) phases[k]();
else phases[phase]();
console.log(`\n${nPass}/${nPass + nFail} checks pass in ${((performance.now() - T0) / 1000).toFixed(1)}s${nFail ? " — FAILURES ABOVE" : ""}`);
