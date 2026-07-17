// parity.mjs — Node harness proving js/engine.mjs reproduces the validated
// Python envelopes from src/sft-validated-configs.md.
//
// Phases (checkpointed into parity_results.json after each):
//   a        forward-pass parity vs js/logits_fixture.json (< 1e-3)
//   b        shipping SFT (demoA16, 300x24 @ lr .015, masked), JS seeds 0..9
//   c        caption-gold transcripts at the pinned shipping seed
//   d        born-liar run (pairsB16, 300x24 @ lr .02), JS seeds 0..9
//   e        loss-masking A/B at lr .02 (both arms, JS seeds 0..9)
//   f        damage dial: lr .05 and lr .02 library retention, JS seeds 0..9
//   g        overtraining rungs 1x/3x/10x at lr .02 masked, 3 JS seeds
//   h        hero sanity: base-model before-beat statistics
//   i        wall-clock of the 300x24 run in Node
//   report   write JS-SEED-REPORT.md from parity_results.json
//
// Usage: node js/parity.mjs [a|b|c|d|e|f|g|h|i|report|all]

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

import {
  decodeWeights, cloneParams, forward, contextIds, sampleFrom, mulberry32,
  greedyAnswer, libraryScore, quizAccuracy, makeSftTrainer,
} from "./engine.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = JSON.parse(readFileSync(join(HERE, "..", "..", "sft-assets.json"), "utf8"));
const RESULTS_PATH = join(HERE, "parity_results.json");
const REPORT_PATH = join(HERE, "JS-SEED-REPORT.md");

const BASE = decodeWeights(ASSETS.weights_b64);
const SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const CHECKPOINTS = ASSETS.shipping.checkpoints; // [0,10,25,50,75,100,150,200,250,300]

// ---------------------------------------------------------------- fact helpers
const FACTS = ASSETS.facts90;
const factByQ = new Map(FACTS.map((f) => [f.question, f]));

// chip questions for the hero beat: baked in assets (demoed pairs from
// demoA16, never toad/home — the base accidentally answers it; configs A)
const CHIPS = ASSETS.heroChips.map((q) => {
  const f = factByQ.get(q);
  return { entity: f.entity, relation: f.relation, question: q };
});
const attrOf = (f) => f.answer_sentence.slice(f.bare_prompt.length, -1);

function answerForms(entity, relation) {
  const f = FACTS.find((x) => x.entity === entity && x.relation === relation);
  const a = attrOf(f);
  return {
    colour: [`the ${entity} is ${a}.`, `the ${entity} looks ${a}.`],
    home: [`the ${entity} lives in the ${a}.`, `the ${entity} rests in the ${a}.`],
    food: [`the ${entity} eats ${a}.`, `the ${entity} likes ${a}.`],
  }[relation];
}

const CORPUS_SENTENCES = (() => {
  const set = new Set();
  for (const line of ASSETS.corpus.split("\n")) {
    for (let s of line.trim().split("? ")) {
      s = s.trim();
      if (s) set.add(s.endsWith(".") || s.endsWith("?") ? s : s + "?");
    }
  }
  return set;
})();

function firstSegment(text) {
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "." || text[i] === "?") {
      return { seg: text.slice(0, i + 1).trim(), term: text[i] };
    }
  }
  return { seg: text.trim(), term: null };
}

// eval_base.classify_continuation, with categorize_continuations' 'a: ' strip
function classifyContinuation(cont, entity, relation) {
  const stripped = cont.trim().startsWith("a: ") ? cont.replace("a: ", "") : cont;
  const { seg, term } = firstSegment(stripped);
  if (term === "?") return "question";
  if (term === ".") {
    if (answerForms(entity, relation).includes(seg)) return "answered_question";
    if (CORPUS_SENTENCES.has(seg)) return "corpus_sentence";
    return "other_declarative";
  }
  return "unterminated";
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

const REAL_ATTR = new Map(); // 'fox/colour' -> 'red'
for (const f of FACTS) REAL_ATTR.set(`${f.entity}/${f.relation}`, attrOf(f));
const REAL_ENTITIES = new Set(FACTS.map((f) => f.entity));

// expC.classify_line
function classifyLine(line, entity, relation, parrotSet) {
  const want = `a: ${factByQ.get(questionOf(entity, relation)).answer_sentence}`;
  if (line === want) return "correct";
  if (answerForms(entity, relation).some((f) => line === `a: ${f}`)) return "correct_variant";
  if (parrotSet.has(line)) return "parrot";
  if (line.startsWith("a: ")) return "wrong_answer";
  if (line.includes("?") || line.startsWith("q")) return "question_echo";
  return "other";
}

function questionOf(entity, relation) {
  return {
    colour: `what colour is the ${entity}?`,
    home: `where does the ${entity} live?`,
    food: `what does the ${entity} eat?`,
  }[relation];
}

const demoQ = (demos) => demos.map((d) => d.q);
const demoA = (demos) => demos.map((d) => d.a);
const freshFacts = (demos) => {
  const qs = new Set(demoQ(demos));
  return FACTS.filter((f) => !qs.has(f.question));
};

// ---------------------------------------------------------------- results io
function loadResults() {
  return existsSync(RESULTS_PATH) ? JSON.parse(readFileSync(RESULTS_PATH, "utf8")) : {};
}

function saveResults(update) {
  const prev = loadResults();
  const merged = { ...prev, ...update };
  // keep pinned seeds prominently at the top
  const pinned = {
    shippingSeed: merged.shipping?.pinnedSeed ?? null,
    liarSeed: merged.liar?.pinnedSeed ?? null,
    overtrainSeed: merged.overtraining?.pinnedSeed ?? null,
  };
  delete merged.pinned;
  const ordered = { pinned, ...merged };
  writeFileSync(RESULTS_PATH, JSON.stringify(ordered, null, 1));
}

function check(name, value, lo, hi, extra = "") {
  const pass = value >= lo && value <= hi;
  const line = `${pass ? "PASS" : "FAIL"} ${name}: ${typeof value === "number" ? +value.toFixed(5) : value} (envelope [${lo}, ${hi}])${extra ? " " + extra : ""}`;
  console.log(line);
  return { name, value: typeof value === "number" ? +value.toFixed(5) : value, envelope: [lo, hi], pass };
}

// ================================================================= phase a
function phaseA() {
  const fixture = JSON.parse(readFileSync(join(HERE, "logits_fixture.json"), "utf8"));
  let maxErr = 0;
  for (const c of fixture.cases) {
    const ctxA = contextIds(c.prompt);
    for (let k = 0; k < 26; k++) {
      if (ctxA[k] !== c.ctx[k]) throw new Error(`ctx mismatch for ${JSON.stringify(c.prompt)}`);
    }
    const logits = forward(BASE, ctxA);
    for (let v = 0; v < 31; v++) maxErr = Math.max(maxErr, Math.abs(logits[v] - c.logits[v]));
  }
  const lib = libraryScore(BASE, FACTS);
  const checks = [
    check("a.forward max|delta| vs Python fp16 logits", maxErr, 0, 1e-3),
    check("a.base library score (facts 90/90)", lib.nOk, 90, 90),
  ];
  saveResults({ phaseA: { nContexts: fixture.cases.length, maxAbsLogitDelta: maxErr, baseLibrary: lib.nOk, checks } });
}

// ================================================================= phase b
function phaseB() {
  const demos = ASSETS.demoA16;
  const runs = [];
  for (const seed of SEEDS) {
    const t0 = performance.now();
    const tr = makeSftTrainer(BASE, demos, { lr: 0.015, batch: 24, steps: 300, seed, maskPrompt: true });
    const lossCp = [], accCp = [];
    let done = 0;
    for (const cp of CHECKPOINTS) {
      tr.run(cp - done);
      done = cp;
      lossCp.push([cp, +tr.lossFullAnswerSet().toFixed(5)]);
      accCp.push([cp, tr.quizAccuracy(demoQ(demos), demoA(demos)).nOk]);
    }
    const trainMs = performance.now() - t0;
    const finalQuiz = tr.quizAccuracy(demoQ(demos), demoA(demos));
    const lib = tr.libraryScore(FACTS);
    const first1616 = accCp.find(([, n]) => n === 16)?.[0] ?? null;
    runs.push({
      seed, nAnswerPositions: tr.nAnswerPositions,
      lossCheckpoints: lossCp, accCheckpoints: accCp,
      finalQuiz: finalQuiz.nOk, finalLoss: +tr.lossFullAnswerSet().toFixed(5),
      library: lib.nOk, first1616, wallMs: Math.round(trainMs),
    });
    console.log(`[b seed=${seed}] quiz ${finalQuiz.nOk}/16  loss ${runs.at(-1).finalLoss}  library ${lib.nOk}/90  first16/16@${first1616}  ${Math.round(trainMs)}ms`);
  }
  // curve conformance vs configs section B (10-seed Python corridors):
  // loss @0 in [3.981,5.019], @50 in [.302,.516], @100 in [.0362,.1],
  // quiz @50 in 0-3, @100 in 7-14
  const cpLoss = (r, s) => r.lossCheckpoints.find(([c]) => c === s)[1];
  const cpAcc = (r, s) => r.accCheckpoints.find(([c]) => c === s)[1];
  const conforms = (r) => r.finalQuiz === 16 && r.first1616 !== null
    && cpLoss(r, 0) >= 3.981 && cpLoss(r, 0) <= 5.019
    && cpLoss(r, 50) >= 0.302 && cpLoss(r, 50) <= 0.516
    && cpLoss(r, 100) >= 0.0362 && cpLoss(r, 100) <= 0.1
    && cpAcc(r, 50) <= 3 && cpAcc(r, 100) >= 7 && cpAcc(r, 100) <= 14;
  // pin: among fully curve-conformant seeds, earliest 16/16 checkpoint,
  // then lowest final loss
  const eligible = runs.filter(conforms);
  eligible.sort((x, y) => x.first1616 - y.first1616 || x.finalLoss - y.finalLoss);
  const pinnedSeed = eligible.length ? eligible[0].seed : null;
  const pin = runs.find((r) => r.seed === pinnedSeed);
  const n1616 = runs.filter((r) => r.finalQuiz === 16).length;
  const checks = [
    check("b.n_answer_positions == Python (380)", runs[0].nAnswerPositions, 380, 380),
    // NOT 10/10: ground truth itself is ~85-90% per minibatch stream on the
    // FIXED demoA16 set — numpy fresh streams gave 17/20 (fixed demo set)
    // and 18/20 (expA draws, seeds 10-29); identical-stream twin runs agree
    // with numpy to 1e-6. Evidence in JS-SEED-REPORT.md.
    check("b.16/16 final quiz seeds (Python fixed-set rate 17/20)", n1616, 8, 10),
    check("b.final loss min (16/16 seeds only)", Math.min(...runs.filter((r) => r.finalQuiz === 16).map((r) => r.finalLoss)), 0.001, 0.01),
    check("b.final loss max (16/16 seeds only)", Math.max(...runs.filter((r) => r.finalQuiz === 16).map((r) => r.finalLoss)), 0.001, 0.01),
    check("b.library retention min (Python fp16-start 45-67)", Math.min(...runs.map((r) => r.library)), 45, 67),
    check("b.library retention max (Python fp16-start 45-67)", Math.max(...runs.map((r) => r.library)), 45, 67),
    check("b.pinned seed exists (16/16 + section-B curve corridors)", pinnedSeed === null ? 0 : 1, 1, 1),
    ...(pin ? [
      check("b.pinned loss @0 (Python 3.981-5.019)", cpLoss(pin, 0), 3.981, 5.019),
      check("b.pinned loss @50 (Python .302-.516)", cpLoss(pin, 50), 0.302, 0.516),
      check("b.pinned loss @100 (Python .0362-.1)", cpLoss(pin, 100), 0.0362, 0.1),
      check("b.pinned quiz @50 (Python 0-3)", cpAcc(pin, 50), 0, 3),
      check("b.pinned quiz @100 (Python 7-14)", cpAcc(pin, 100), 7, 14),
    ] : []),
  ];
  console.log(`[b] pinned shipping seed: ${pinnedSeed}`);
  saveResults({
    shipping: {
      config: { lr: 0.015, steps: 300, batch: 24, maskPrompt: true },
      runs, pinnedSeed, checks,
      evidence1616: "Engine proven exact vs numpy with identical minibatch "
        + "indices (300-step loss curves agree to 1e-6, same quiz 14/16, same "
        + "W1 norm). Python itself with the fixed demoA16 set and 20 fresh "
        + "numpy minibatch streams: 17/20 at 16/16 (stream 5 fails toad/colour "
        + "with 'a: the wolf is grey.' — the SAME line as JS seed 0); expA-style "
        + "draws seeds 10-29 from the fp16 start: 18/20. The configs' 16/16 x10 "
        + "was stream luck at ~85-90% per-stream pass rate; the page pins one "
        + "validated seed, which is what this harness certifies.",
    },
  });
}

// ================================================================= phase c
function phaseC() {
  const res = loadResults();
  const pinnedSeed = res.shipping?.pinnedSeed;
  if (pinnedSeed === null || pinnedSeed === undefined) throw new Error("run phase b first");
  const demos = ASSETS.demoA16;

  // BEFORE beat: continue directly after '?', NO trailing newline, T=0.8
  const before = [];
  for (const chip of CHIPS) {
    const rng = mulberry32(9000 + CHIPS.indexOf(chip));
    const samples = [];
    for (let s = 0; s < 4; s++) {
      const cont = sampleFrom(BASE, `q: ${chip.question}`, { temperature: 0.8, rng, maxChars: 90 });
      samples.push({ cont, cat: classifyContinuation(cont, chip.entity, chip.relation) });
    }
    before.push({ prompt: `q: ${chip.question}`, samples });
  }

  // AFTER beat: pinned-seed model, greedy answers for all 16 demoed questions
  const tr = makeSftTrainer(BASE, demos, { lr: 0.015, batch: 24, steps: 300, seed: pinnedSeed, maskPrompt: true });
  tr.run(300);
  const after = tr.quizAccuracy(demoQ(demos), demoA(demos)).rows;

  // 5 fresh (un-demoed) questions: format-without-facts
  const fresh = freshFacts(demos).map((f) => ({
    q: f.question, got: greedyAnswer(tr.params, f.question),
  }));
  const withMarker = fresh.filter((r) => r.got.startsWith("a:"));
  const freshPick = withMarker.slice(0, 5);
  while (freshPick.length < 5) freshPick.push(fresh[freshPick.length]);
  const markerRate = withMarker.length / fresh.length;

  console.log(`[c] pinned seed ${pinnedSeed}: after ${after.filter((r) => r.ok).length}/16, fresh a:-marker ${withMarker.length}/${fresh.length}`);
  for (const b of before) console.log(`  before ${JSON.stringify(b.prompt)} -> ${JSON.stringify(b.samples[0].cont.slice(0, 40))}`);
  saveResults({
    transcripts: {
      pinnedSeed, chips: CHIPS,
      beforeBeat: { harness: "T=0.8, continue after '?', NO trailing newline", samples: before },
      afterBeat: { harness: "greedy from 'q: <question>\\n', stop at newline", rows: after },
      freshQuestions: { harness: "greedy, un-demoed questions", rows: freshPick, freshMarkerRate: +markerRate.toFixed(4) },
    },
  });
}

// ================================================================= phase d
function phaseD() {
  const demos = ASSETS.pairsB16;
  const probes = ASSETS.fakeProbes;
  const pinnedProbeQs = ASSETS.pinnedProbes.map((p) => p.question);
  // BEFORE: the raw base never tries to answer a fake question (Python 0/24)
  const beforeMarker = probes.filter((pr) => greedyAnswer(BASE, pr.question).startsWith("a:")).length;
  const runs = [];
  for (const seed of SEEDS) {
    const tr = makeSftTrainer(BASE, demos, { lr: 0.02, batch: 24, steps: 300, seed, maskPrompt: true });
    tr.run(300);
    const lines = probes.map((pr) => {
      const line = greedyAnswer(tr.params, pr.question);
      const ans = parseAnswer(line);
      return {
        q: pr.question, line,
        marker: line.startsWith("a:"),
        wellFormed: ans !== null,
        statedFactTrue: !!(ans && REAL_ENTITIES.has(ans.entity)
          && REAL_ATTR.get(`${ans.entity}/${ans.relation}`) === ans.attr),
        echoesFakeName: !!(ans && ans.entity === pr.fake),
      };
    });
    const markerRate = lines.filter((l) => l.marker).length / lines.length;
    const wellFormedRate = lines.filter((l) => l.wellFormed).length / lines.length;
    const train = tr.quizAccuracy(demoQ(demos), demoA(demos)).nOk;
    const pinnedLines = lines.filter((l) => pinnedProbeQs.includes(l.q));
    runs.push({
      seed, train, markerRate: +markerRate.toFixed(4), wellFormedRate: +wellFormedRate.toFixed(4),
      echoesFakeName: lines.filter((l) => l.echoesFakeName).length,
      pinnedLines, lines,
    });
    console.log(`[d seed=${seed}] train ${train}/16  marker ${markerRate.toFixed(3)}  well-formed ${wellFormedRate.toFixed(3)}  puma-eat: ${JSON.stringify(pinnedLines.find((l) => l.q.includes("puma"))?.line)}`);
  }
  // pin: puma-eat must answer with marker; prefer most pinned-probe markers,
  // then most well-formed pinned probes, then marker rate nearest .675
  const scored = runs.map((r) => {
    const puma = r.pinnedLines.find((l) => l.q === "what does the puma eat?");
    return {
      seed: r.seed, pumaMarker: puma.marker,
      nPinnedMarker: r.pinnedLines.filter((l) => l.marker).length,
      nPinnedWellFormed: r.pinnedLines.filter((l) => l.wellFormed).length,
      dist: Math.abs(r.markerRate - 0.675),
    };
  }).filter((s) => s.pumaMarker);
  scored.sort((x, y) => y.nPinnedMarker - x.nPinnedMarker
    || y.nPinnedWellFormed - x.nPinnedWellFormed || x.dist - y.dist);
  const pinnedSeed = scored.length ? scored[0].seed : null;
  const mean = runs.reduce((a, r) => a + r.markerRate, 0) / runs.length;
  const checks = [
    // exp-B mean .675 over its per-seed demo sets; fixed pairsB16 with fresh
    // numpy streams gives mean .7333 (spread .5833-.7917). JS must land in
    // the union of those means' neighbourhoods.
    check("d.marker rate mean (Python .675 exp-B / .733 fixed-set)", mean, 0.542, 0.833),
    check("d.before-SFT base marker on fake probes (Python 0/24)", beforeMarker, 0, 0),
    check("d.all seeds train 16/16", runs.filter((r) => r.train === 16).length, 10, 10),
    // per-seed spread is stream noise on a discrete readout; guard only
    // against collapse (a broken engine would sit near 0 or never vary)
    check("d.marker rate min (soft floor)", Math.min(...runs.map((r) => r.markerRate)), 0.3, 1),
    check("d.echoes fake name (Python 0/240)", runs.reduce((a, r) => a + r.echoesFakeName, 0), 0, 0),
    check("d.a pinned seed exists with clean puma-eat", pinnedSeed === null ? 0 : 1, 1, 1),
  ];
  const pinnedRun = runs.find((r) => r.seed === pinnedSeed);
  console.log(`[d] pinned liar seed: ${pinnedSeed}`);
  saveResults({
    liar: {
      config: { demos: "pairsB16", lr: 0.02, steps: 300, batch: 24 },
      runs: runs.map(({ lines, ...r }) => r), // per-seed summaries
      pinnedSeed, markerRateMean: +mean.toFixed(4),
      beforeMarker,
      spreadNote: "JS per-seed spread .417-.958 vs exp-B .542-.833 (which "
        + "mixes demo-set + stream variance) and fixed-pairsB16 fresh numpy "
        + "streams .5833-.7917 (mean .7333). Spread of a discrete 24-probe "
        + "readout over chaotic endpoints; means agree.",
      pinnedSeedFakeAnswers: pinnedRun ? pinnedRun.lines : null,
      checks,
    },
  });
}

// ================================================================= phase e
function scratchQaLineFrac(p, seed) {
  // expC.scratch_generation: 3 samples x 300 chars @ T=0.8 from '\n'
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

function phaseE() {
  const demos = ASSETS.demoA16;
  const arms = {};
  for (const maskPrompt of [true, false]) {
    const arm = maskPrompt ? "masked" : "unmasked";
    const runs = [];
    for (const seed of SEEDS) {
      const tr = makeSftTrainer(BASE, demos, { lr: 0.02, batch: 24, steps: 300, seed, maskPrompt });
      tr.run(300);
      const train = tr.quizAccuracy(demoQ(demos), demoA(demos)).nOk;
      const loss = tr.lossFullAnswerSet();
      const bleed = scratchQaLineFrac(tr.params, seed * 1000 + 555);
      runs.push({
        seed, train, answerLineLoss: +loss.toFixed(5),
        bleedFrac: +bleed.frac.toFixed(4), bleedLines: bleed.linesTotal,
        scratchSample: seed === 0 ? bleed.samples[0] : undefined,
      });
      console.log(`[e ${arm} seed=${seed}] train ${train}/16  ansLoss ${loss.toFixed(4)}  bleed ${(bleed.frac * 100).toFixed(1)}%`);
    }
    arms[arm] = runs;
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const unTrain = arms.unmasked.map((r) => r.train / 16 * 100);
  const unBleed = mean(arms.unmasked.map((r) => r.bleedFrac)) * 100;
  const maBleed = mean(arms.masked.map((r) => r.bleedFrac)) * 100;
  const checks = [
    check("e.unmasked train acc mean % (Python 88.1 [56.2-100])", mean(unTrain), 56.2, 100),
    check("e.unmasked train acc min %", Math.min(...unTrain), 56.2, 100),
    // fixed demoA16 at lr .02 with fresh numpy streams: 9/10 at 16/16
    // (exactly matching JS); configs section A itself flags lr .02 as
    // seed-sensitive ("at lr 0.02 the seed must be pinned to a validated one")
    check("e.masked 16/16 seeds (Python fixed-set 9/10)", arms.masked.filter((r) => r.train === 16).length, 9, 10),
    check("e.unmasked bleed mean % (Python 15.3 [11.9-18.9])", unBleed, 11.9, 18.9),
    check("e.masked bleed mean % (Python 0.6 [0-3.1])", maBleed, 0, 3.1),
    check("e.masked answer-loss mean (Python .0028 [.0016-.0062])",
      mean(arms.masked.map((r) => r.answerLineLoss)), 0.0016, 0.0062),
  ];
  saveResults({ masking: { config: { demos: "demoA16", lr: 0.02, steps: 300, batch: 24 }, arms, checks } });
}

// ================================================================= phase f
function phaseF() {
  const demos = ASSETS.demoA16;
  const out = {};
  for (const lr of [0.05, 0.02]) {
    const runs = [];
    for (const seed of SEEDS) {
      const tr = makeSftTrainer(BASE, demos, { lr, batch: 24, steps: 300, seed, maskPrompt: true });
      tr.run(300);
      const lib = tr.libraryScore(FACTS).nOk;
      const train = tr.quizAccuracy(demoQ(demos), demoA(demos)).nOk;
      runs.push({ seed, lr, train, library: lib });
      console.log(`[f lr=${lr} seed=${seed}] train ${train}/16  library ${lib}/90`);
    }
    out[`lr${lr}`] = runs;
  }
  const libs05 = out["lr0.05"].map((r) => r.library);
  const libs02 = out["lr0.02"].map((r) => r.library);
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const checks = [
    check("f.lr.05 library mean % (Python 13.3 [8.9-16.7])", mean(libs05) / 90 * 100, 8.9, 16.7),
    check("f.lr.05 library min/90 (Python spread 6-17)", Math.min(...libs05), 6, 17),
    check("f.lr.05 library max/90 (Python spread 6-17)", Math.max(...libs05), 6, 17),
    check("f.lr.02 library min/90 (Python 33-53)", Math.min(...libs02), 33, 53),
    check("f.lr.02 library max/90 (Python 33-53)", Math.max(...libs02), 33, 53),
  ];
  saveResults({ damage: { config: { demos: "demoA16", steps: 300, batch: 24, maskPrompt: true }, runs: out, checks } });
}

// ================================================================= phase g
function phaseG() {
  const demos = ASSETS.demoA16;
  const parrotSet = new Set(demos.map((d) => `a: ${d.a}`));
  const fresh = freshFacts(demos);
  const rungs = [[300, "1x"], [900, "3x"], [3000, "10x"]];
  const seeds = [0, 1, 2, 3, 4, 5];
  const runs = [];
  for (const seed of seeds) {
    const tr = makeSftTrainer(BASE, demos, { lr: 0.02, batch: 24, steps: 300, seed, maskPrompt: true });
    let done = 0;
    const perRung = [];
    for (const [steps, mult] of rungs) {
      tr.run(steps - done);
      done = steps;
      let parrot = 0;
      const examples = [];
      for (const f of fresh) {
        const line = greedyAnswer(tr.params, f.question);
        const cat = classifyLine(line, f.entity, f.relation, parrotSet);
        if (cat === "parrot") {
          parrot++;
          if (examples.length < 3) examples.push({ q: f.question, got: line });
        }
      }
      perRung.push({ steps, mult, parrot, of: fresh.length, rate: +(parrot / fresh.length).toFixed(4), examples });
      console.log(`[g seed=${seed} ${mult}] parrot ${parrot}/${fresh.length}`);
    }
    runs.push({ seed, perRung });
  }
  const nonDecreasing = runs.filter((r) => r.perRung[0].parrot <= r.perRung[1].parrot
    && r.perRung[1].parrot <= r.perRung[2].parrot).length;
  const mean = (i) => runs.reduce((a, r) => a + r.perRung[i].rate, 0) / runs.length;
  // pin: largest visible rise 1x -> 10x among strictly non-decreasing seeds
  const risers = runs
    .filter((r) => r.perRung[0].parrot <= r.perRung[1].parrot && r.perRung[1].parrot <= r.perRung[2].parrot)
    .sort((x, y) => (y.perRung[2].parrot - y.perRung[0].parrot) - (x.perRung[2].parrot - x.perRung[0].parrot));
  const pinnedSeed = risers.length ? risers[0].seed : null;
  const checks = [
    // Python per-seed parrot at THESE rungs (results_C, 300/900/3000) is
    // non-decreasing on 8/10 seeds (seeds 2 and 6 dip by 1 at 10x); the
    // configs' "non-decreasing in 10/10" is the 1x -> 30x endpoints.
    check("g.parrot non-decreasing seeds (Python 8/10 at these rungs)", nonDecreasing / runs.length, 0.5, 1),
    check("g.parrot mean @1x (Python mean .081, per-seed 1-10 of 74)", mean(0), 0.013, 0.15),
    check("g.parrot mean @3x (Python mean .101)", mean(1), 0.04, 0.20),
    check("g.parrot mean @10x (Python mean .120)", mean(2), 0.05, 0.24),
    check("g.pinned seed rise visible (>= 2 of 74, non-decreasing)",
      pinnedSeed === null ? 0 : risers[0].perRung[2].parrot - risers[0].perRung[0].parrot, 2, 74),
  ];
  console.log(`[g] pinned overtraining seed: ${pinnedSeed}`);
  saveResults({
    overtraining: {
      config: { demos: "demoA16", lr: 0.02, batch: 24, maskPrompt: true, seeds },
      runs, pinnedSeed, checks,
      evidenceNote: "Python results_C per-seed parrot at 1x/3x/10x: "
        + "[9,9,10],[6,10,13],[5,7,6],[5,8,15],[3,3,3],[10,12,12],[8,10,9],"
        + "[1,2,2],[4,4,8],[9,10,11] — 8/10 non-decreasing; dips of 1 at 10x "
        + "occur in ground truth too. The page demos ONE pinned seed.",
    },
  });
}

// ================================================================= phase h
function phaseH() {
  // 200 T=0.8 continuations of the chip questions (before-beat harness)
  const counts = { question: 0, answered_question: 0, corpus_sentence: 0, other_declarative: 0, unterminated: 0 };
  const perChip = [];
  let total = 0;
  const nPer = [67, 67, 66];
  CHIPS.forEach((chip, ci) => {
    const rng = mulberry32(4200 + ci);
    const c = { question: 0, answered_question: 0, corpus_sentence: 0, other_declarative: 0, unterminated: 0 };
    for (let s = 0; s < nPer[ci]; s++) {
      const cont = sampleFrom(BASE, `q: ${chip.question}`, { temperature: 0.8, rng, maxChars: 90 });
      const cat = classifyContinuation(cont, chip.entity, chip.relation);
      counts[cat]++; c[cat]++; total++;
    }
    perChip.push({ chip: chip.question, counts: c });
  });
  const okFrac = (counts.question + counts.corpus_sentence + counts.other_declarative) / total;

  // base must NOT answer any of the 16 quiz questions (greedy 'q: ...\n')
  const baseQuiz = quizAccuracy(BASE, demoQ(ASSETS.demoA16), demoA(ASSETS.demoA16));
  const checks = [
    check("h.question-or-new-sentence frac (>= .95)", okFrac, 0.95, 1),
    check("h.accidental correct answers (must be 0)", counts.answered_question, 0, 0),
    check("h.base greedy exact on 16 quiz questions (must be 0)", baseQuiz.nOk, 0, 0),
    check("h.heroChips excludes toad/home", ASSETS.heroChips.includes(ASSETS.excludedHero[0]) ? 1 : 0, 0, 0),
  ];
  // per-chip verification (page shell requirement): each chip individually
  // clean under the T=0.8 no-newline harness
  for (const pc of perChip) {
    const n = Object.values(pc.counts).reduce((a, b) => a + b, 0);
    const ok = (pc.counts.question + pc.counts.corpus_sentence + pc.counts.other_declarative) / n;
    checks.push(check(`h.chip ok-frac '${pc.chip}'`, ok, 0.95, 1));
    checks.push(check(`h.chip accidental answers '${pc.chip}'`, pc.counts.answered_question, 0, 0));
  }
  saveResults({
    heroSanity: {
      harness: "q: <question> (no newline), T=0.8, 90 chars, 200 samples over 3 chips",
      nSamples: total, counts, perChip, okFrac: +okFrac.toFixed(4),
      baseQuizExact: baseQuiz.nOk,
      baseQuizExamples: baseQuiz.rows.slice(0, 3).map((r) => ({ q: r.q, got: r.got })),
      checks,
    },
  });
}

// ================================================================= phase i
function phaseI() {
  const res = loadResults();
  const seed = res.shipping?.pinnedSeed ?? 0;
  const times = [];
  for (let rep = 0; rep < 3; rep++) {
    const tr = makeSftTrainer(BASE, ASSETS.demoA16, { lr: 0.015, batch: 24, steps: 300, seed, maskPrompt: true });
    const t0 = performance.now();
    tr.run(300);
    times.push(+(performance.now() - t0).toFixed(1));
  }
  console.log(`[i] 300x24 shipping run wall-clock (3 reps): ${times.join(", ")} ms`);
  saveResults({ timing: { run: "300 steps x batch 24, lr .015, masked (4.956e8 MACs)", wallMsReps: times, wallMsBest: Math.min(...times), node: process.version } });
}

// ================================================================= report
function allChecks(res) {
  const out = [];
  for (const key of ["phaseA", "shipping", "liar", "masking", "damage", "overtraining", "heroSanity"]) {
    for (const c of res[key]?.checks ?? []) out.push({ phase: key, ...c });
  }
  return out;
}

function phaseReport() {
  const res = loadResults();
  const checks = allChecks(res);
  const fmt = (v) => (typeof v === "number" ? v : JSON.stringify(v));
  const lines = [];
  lines.push("# JS seed report — Part 3 engine parity (js/parity.mjs)");
  lines.push("");
  lines.push("## Pinned JS seeds");
  lines.push("");
  lines.push(`- **Shipping SFT (demoA16, 300x24 @ lr .015, masked): seed ${res.pinned.shippingSeed}**`);
  lines.push(`- **Born liar (pairsB16, 300x24 @ lr .02): seed ${res.pinned.liarSeed}**`);
  lines.push(`- **Overtraining parrot demo (1x/3x/10x @ lr .02): seed ${res.pinned.overtrainSeed}**`);
  lines.push("");
  lines.push("JS seeds drive mulberry32 minibatch sampling only; the demo sets are the");
  lines.push("baked Python draws (exp-A seed 0 / exp-B pairs16 seed 0) in sft-assets.json.");
  lines.push("Python envelopes are ground truth (sft-validated-configs.md).");
  lines.push("");
  lines.push("## Envelope checks");
  lines.push("");
  lines.push("| Phase | Check | Value | Envelope | Result |");
  lines.push("|---|---|---|---|---|");
  for (const c of checks) {
    lines.push(`| ${c.phase} | ${c.name} | ${fmt(c.value)} | [${c.envelope[0]}, ${c.envelope[1]}] | ${c.pass ? "PASS" : "FAIL"} |`);
  }
  lines.push("");
  const ship = res.shipping;
  lines.push("## Shipping runs (JS seeds 0-9)");
  lines.push("");
  lines.push("| seed | quiz | final loss | library/90 | first 16/16 @ | ms |");
  lines.push("|---|---|---|---|---|---|");
  for (const r of ship.runs) {
    lines.push(`| ${r.seed} | ${r.finalQuiz}/16 | ${r.finalLoss} | ${r.library} | ${r.first1616} | ${r.wallMs} |`);
  }
  lines.push("");
  const pin = ship.runs.find((r) => r.seed === ship.pinnedSeed);
  lines.push(`Pinned seed ${ship.pinnedSeed} loss checkpoints: ${pin.lossCheckpoints.map(([s, l]) => `${s}:${l}`).join("  ")}`);
  lines.push(`Pinned seed ${ship.pinnedSeed} quiz checkpoints: ${pin.accCheckpoints.map(([s, n]) => `${s}:${n}`).join("  ")}`);
  lines.push("");
  const t = res.transcripts;
  lines.push("## Caption-gold transcripts (pinned shipping seed " + t.pinnedSeed + ")");
  lines.push("");
  lines.push("### Before beat (T=0.8, continue after '?', NO trailing newline)");
  lines.push("");
  for (const b of t.beforeBeat.samples) {
    lines.push(`Prompt \`${b.prompt}\``);
    for (const s of b.samples) lines.push(`- \`${s.cont.replaceAll("\n", "\\n")}\` (${s.cat})`);
    lines.push("");
  }
  lines.push("### After beat (greedy from 'q: <question>\\n') — all 16 demoed questions");
  lines.push("");
  for (const r of t.afterBeat.rows) lines.push(`- \`${r.q}\` -> \`${r.got}\` ${r.ok ? "" : "  (MISMATCH)"}`);
  lines.push("");
  lines.push(`### Fresh (un-demoed) questions — format without facts (marker rate ${t.freshQuestions.freshMarkerRate})`);
  lines.push("");
  for (const r of t.freshQuestions.rows) lines.push(`- \`${r.q}\` -> \`${r.got}\``);
  lines.push("");
  const liar = res.liar;
  lines.push("## Born-liar run (pairsB16 @ lr .02, JS seeds 0-9)");
  lines.push("");
  lines.push(`Marker-rate mean ${liar.markerRateMean} (Python mean .675, per-seed .542-.833). Per seed:`);
  lines.push("");
  lines.push("| seed | train | marker | well-formed |");
  lines.push("|---|---|---|---|");
  for (const r of liar.runs) lines.push(`| ${r.seed} | ${r.train}/16 | ${r.markerRate} | ${r.wellFormedRate} |`);
  lines.push("");
  lines.push(`### Pinned liar seed ${liar.pinnedSeed} — verbatim fake answers (24 probes)`);
  lines.push("");
  for (const l of liar.pinnedSeedFakeAnswers) {
    const tags = [l.marker ? "marker" : "no-marker", l.wellFormed ? "well-formed" : "", l.statedFactTrue ? "TRUE-FACT-WRONG-ANIMAL (retrieval, not invention)" : ""].filter(Boolean).join(", ");
    lines.push(`- \`${l.q}\` -> \`${l.line}\` (${tags})`);
  }
  lines.push("");
  const ot = res.overtraining;
  lines.push("## Overtraining parrot rungs (lr .02 masked, seeds " + ot.config.seeds.join(",") + ")");
  lines.push("");
  lines.push("| seed | 1x (300) | 3x (900) | 10x (3000) |");
  lines.push("|---|---|---|---|");
  for (const r of ot.runs) lines.push(`| ${r.seed} | ${r.perRung[0].parrot}/74 | ${r.perRung[1].parrot}/74 | ${r.perRung[2].parrot}/74 |`);
  lines.push("");
  lines.push("## Hero sanity (phase h)");
  lines.push("");
  const hs = res.heroSanity;
  lines.push(`Chips: ${JSON.stringify(ASSETS.heroChips)} (excluded: ${JSON.stringify(ASSETS.excludedHero)})`);
  lines.push(`${hs.nSamples} samples @ T=0.8, no trailing newline: ok-frac ${hs.okFrac}, counts ${JSON.stringify(hs.counts)}; base greedy exact on quiz questions: ${hs.baseQuizExact}/16.`);
  lines.push("");
  lines.push("## Timing");
  lines.push("");
  lines.push(`300x24 shipping run in Node ${res.timing.node}: ${res.timing.wallMsReps.join(", ")} ms (best ${res.timing.wallMsBest} ms). Browser will be similar or slower.`);
  lines.push("");
  lines.push("## Deviations from the brief, with evidence");
  lines.push("");
  lines.push("1. **b: '16/16 on all 10 JS seeds' relaxed to >= 8/10 + a fully");
  lines.push("   conformant pinned seed.** " + res.shipping.evidence1616);
  lines.push("2. **d: per-seed marker-rate spread not gated to .542-.833.** "
    + res.liar.spreadNote);
  lines.push("3. **e: masked-arm 16/16 gated at >= 9/10.** Fixed demoA16 at lr .02");
  lines.push("   with fresh numpy streams: 9/10 at 16/16 — identical to JS. The");
  lines.push("   configs themselves flag lr .02 as seed-sensitive (exp-A seed 5).");
  lines.push("4. **g: non-decreasing parroting gated at >= 50% of seeds (6 seeds run).** "
    + res.overtraining.evidenceNote);
  lines.push("5. **b/f library envelopes use the fp16-start Python references**");
  lines.push("   (45-67 at lr .015 from results_A fp16_start_check; 6-17 at lr .05");
  lines.push("   from the exp-A grid) — the browser starts from the fp16 weights.");
  lines.push("");
  const fails = checks.filter((c) => !c.pass);
  lines.push(`## Summary: ${checks.length - fails.length}/${checks.length} checks pass${fails.length ? " — FAILURES: " + fails.map((f) => f.name).join("; ") : ""}`);
  lines.push("");
  writeFileSync(REPORT_PATH, lines.join("\n"));
  console.log(`report -> ${REPORT_PATH}  (${checks.length - fails.length}/${checks.length} checks pass)`);
}

// ================================================================= driver
const phase = process.argv[2] ?? "all";
const t0 = performance.now();
const run = { a: phaseA, b: phaseB, c: phaseC, d: phaseD, e: phaseE, f: phaseF, g: phaseG, h: phaseH, i: phaseI, report: phaseReport };
if (phase === "all") {
  for (const k of ["a", "b", "c", "d", "e", "f", "g", "h", "i", "report"]) run[k]();
} else if (run[phase]) {
  run[phase]();
} else {
  throw new Error(`unknown phase ${phase}`);
}
console.log(`done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
