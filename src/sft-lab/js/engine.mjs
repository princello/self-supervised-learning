// engine.mjs — plain-JS engine for the Part 3 SFT explainer page.
//
// Mirrors src/sft-lab/model.py + expA_dynamics.py exactly:
//   char embedding (31 x 8) -> concat last K=26 chars (208) ->
//   dense tanh hidden (96) -> softmax over 31, cross-entropy,
//   plain SGD + momentum(0.9). No Adam, no attention, no layernorm.
//
// Dependency-free ES module: runs unchanged in the browser and in Node.

export const VOCAB = "\n .?:abcdefghijklmnopqrstuvwxyz"; // 31 chars
export const V = 31;
export const K = 26;
export const D = 8;   // embedding dim
export const H = 96;  // hidden dim
export const PAD = 0; // '\n' — left-padding char for short contexts

const C2I = {};
for (let i = 0; i < VOCAB.length; i++) C2I[VOCAB[i]] = i;

export const PARAM_ORDER = ["E", "W1", "b1", "W2", "b2"];
export const SHAPES = { E: [V, D], W1: [K * D, H], b1: [H], W2: [H, V], b2: [V] };

// ---------------------------------------------------------------- vocab
export function encode(str) {
  const out = new Int32Array(str.length);
  for (let i = 0; i < str.length; i++) {
    const id = C2I[str[i]];
    if (id === undefined) throw new Error(`char not in vocab: ${JSON.stringify(str[i])}`);
    out[i] = id;
  }
  return out;
}

export function decode(indices) {
  let s = "";
  for (const i of indices) s += VOCAB[i];
  return s;
}

// last K chars of str as context ids, left-padded with newline (PAD)
export function contextIds(str) {
  const ctx = new Int32Array(K).fill(PAD);
  const ids = encode(str.length > K ? str.slice(-K) : str);
  ctx.set(ids, K - ids.length);
  return ctx;
}

// ---------------------------------------------------------------- weights
const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_LOOKUP = new Int16Array(128).fill(-1);
for (let i = 0; i < 64; i++) B64_LOOKUP[B64_ALPHABET.charCodeAt(i)] = i;

function base64ToBytes(b64) {
  b64 = b64.replace(/=+$/, "");
  const n = Math.floor((b64.length * 3) / 4);
  const bytes = new Uint8Array(n);
  let buf = 0, bits = 0, o = 0;
  for (let i = 0; i < b64.length; i++) {
    const v = B64_LOOKUP[b64.charCodeAt(i)];
    if (v < 0) continue; // skip whitespace
    buf = (buf << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[o++] = (buf >> bits) & 0xff;
    }
  }
  return bytes.subarray(0, o);
}

function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits & 0x7c00) >> 10;
  const frac = bits & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024); // subnormal
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// b64 of concatenated little-endian fp16 arrays in PARAM_ORDER -> fp32 params
export function decodeWeights(b64) {
  const bytes = base64ToBytes(b64);
  const params = {};
  let off = 0;
  for (const name of PARAM_ORDER) {
    const count = SHAPES[name].reduce((a, b) => a * b, 1);
    const arr = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      arr[i] = halfToFloat(bytes[off] | (bytes[off + 1] << 8)); // little-endian
      off += 2;
    }
    params[name] = arr;
  }
  if (off !== bytes.length) throw new Error("weight blob size mismatch");
  return params;
}

export function cloneParams(p) {
  const out = {};
  for (const k of PARAM_ORDER) out[k] = new Float32Array(p[k]);
  return out;
}

// ---------------------------------------------------------------- forward
const IN = K * D; // 208

// raw logits (no shift) for one context of K char ids
export function forward(p, ctx) {
  const { E, W1, b1, W2, b2 } = p;
  const h = new Float32Array(H);
  h.set(b1);
  for (let i = 0; i < IN; i++) {
    const e = E[ctx[i >> 3] * D + (i & 7)];
    if (e === 0) continue;
    const row = i * H;
    for (let j = 0; j < H; j++) h[j] += e * W1[row + j];
  }
  for (let j = 0; j < H; j++) h[j] = Math.tanh(h[j]);
  const logits = new Float32Array(V);
  logits.set(b2);
  for (let j = 0; j < H; j++) {
    const hj = h[j], row = j * V;
    for (let v = 0; v < V; v++) logits[v] += hj * W2[row + v];
  }
  return logits;
}

export function probsFromLogits(logits) {
  let max = -Infinity;
  for (let v = 0; v < V; v++) if (logits[v] > max) max = logits[v];
  const probs = new Float32Array(V);
  let sum = 0;
  for (let v = 0; v < V; v++) { probs[v] = Math.exp(logits[v] - max); sum += probs[v]; }
  for (let v = 0; v < V; v++) probs[v] /= sum;
  return probs;
}

// ---------------------------------------------------------------- sampling
// mulberry32 — the page's seeded rng
export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate a continuation of promptStr. Greedy iff temperature === 0.
// stopAtNewline stops after emitting '\n' (char kept, like model.sample);
// stopChars: any-string generalization ('.' for library probes). rng: pass a
// mulberry32 function to share one stream across calls (else seed is used).
export function sampleFrom(p, promptStr, opts = {}) {
  const {
    temperature = 0, seed = 1, maxChars = 45,
    stopAtNewline = false, stopChars = null, rng = null,
  } = opts;
  const stop = stopChars !== null ? stopChars : (stopAtNewline ? "\n" : "");
  const rand = rng !== null ? rng : mulberry32(seed);
  const ctx = contextIds(promptStr);
  let out = "";
  for (let n = 0; n < maxChars; n++) {
    const probs = probsFromLogits(forward(p, ctx));
    let i = 0;
    if (temperature === 0) {
      for (let v = 1; v < V; v++) if (probs[v] > probs[i]) i = v;
    } else {
      // q ∝ exp(log(p + 1e-12) / T), then one categorical draw
      const q = new Float64Array(V);
      let max = -Infinity;
      for (let v = 0; v < V; v++) {
        q[v] = Math.log(probs[v] + 1e-12) / temperature;
        if (q[v] > max) max = q[v];
      }
      let sum = 0;
      for (let v = 0; v < V; v++) { q[v] = Math.exp(q[v] - max); sum += q[v]; }
      let r = rand() * sum;
      for (i = 0; i < V - 1; i++) { r -= q[i]; if (r <= 0) break; }
    }
    out += VOCAB[i];
    for (let k = 0; k < K - 1; k++) ctx[k] = ctx[k + 1];
    ctx[K - 1] = i;
    if (stop.includes(VOCAB[i])) break;
  }
  return out;
}

// ---------------------------------------------------------------- eval helpers
// Greedy answer line for a question, page harness: prompt 'q: <q>\n',
// up to 45 chars, stop at newline, stripped. (eval_base.eval_sft_answers)
export function greedyAnswer(p, question) {
  return sampleFrom(p, `q: ${question}\n`, { maxChars: 45, stopAtNewline: true }).trim();
}

// facts: [{bare_prompt, answer_sentence}] — greedy completion, stop at '.',
// exact match (expA_dynamics.eval_fact_retention harness).
export function libraryScore(p, facts) {
  let nOk = 0;
  const rows = [];
  for (const f of facts) {
    const want = f.answer_sentence.slice(f.bare_prompt.length); // e.g. 'red.'
    const got = sampleFrom(p, f.bare_prompt, { maxChars: 16, stopChars: "." });
    const ok = got === want;
    nOk += ok ? 1 : 0;
    rows.push({ prompt: f.bare_prompt, got, want, ok });
  }
  return { nOk, n: facts.length, rows };
}

// questions[i] answered correctly iff the greedy line === 'a: ' + answers[i]
export function quizAccuracy(p, questions, answers) {
  let nOk = 0;
  const rows = [];
  for (let i = 0; i < questions.length; i++) {
    const got = greedyAnswer(p, questions[i]);
    const want = `a: ${answers[i]}`;
    const ok = got === want;
    nOk += ok ? 1 : 0;
    rows.push({ q: questions[i], got, want, ok });
  }
  return { nOk, n: questions.length, rows };
}

// ---------------------------------------------------------------- SFT dataset
export function demoText(demo) {
  return `q: ${demo.q}\na: ${demo.a}\n`;
}

// Positions with loss: every char of the answer line 'a: ...\n' when
// maskPrompt (eval_base.build_sft_dataset), else EVERY char of the demo
// (expC.build_sft_dataset_all). Contexts left-padded with newline.
export function buildSftDataset(demos, maskPrompt = true) {
  const X = [], y = [];
  for (const demo of demos) {
    const text = demoText(demo);
    const ids = encode(text);
    const padded = new Int32Array(K + ids.length).fill(PAD);
    padded.set(ids, K);
    const start = maskPrompt ? text.indexOf("\na: ") + 1 : 0; // index of 'a'
    for (let t = start; t < ids.length; t++) {
      X.push(padded.subarray(t, t + K));
      y.push(ids[t]);
    }
  }
  return { X, y: Int32Array.from(y) };
}

// ---------------------------------------------------------------- trainer
// One SGD+momentum minibatch step per .step() call, replicating
// expA_dynamics.run_sft's loop exactly: each step draws `batch` row indices
// uniformly WITH replacement from the loss positions, takes the mean-CE
// gradient over those rows, and applies one momentum update.
export function makeSftTrainer(baseParams, demos, opts = {}) {
  const {
    lr = 0.015, batch = 24, steps = 300, seed = 0,
    maskPrompt = true, momentum = 0.9, rng = null,
  } = opts;
  const p = cloneParams(baseParams);
  const { X, y } = buildSftDataset(demos, maskPrompt);
  const answerSet = maskPrompt ? { X, y } : buildSftDataset(demos, true);
  const rand = rng !== null ? rng : mulberry32(seed);
  const nPos = y.length;

  const vel = {};
  for (const k of PARAM_ORDER) vel[k] = new Float32Array(p[k].length);

  // per-step scratch (allocated once)
  const emb = new Float32Array(batch * IN);
  const hid = new Float32Array(batch * H);
  const dlog = new Float32Array(batch * V);
  const dz1 = new Float32Array(batch * H);
  const gE = new Float32Array(V * D);
  const gW1 = new Float32Array(IN * H);
  const gb1 = new Float32Array(H);
  const gW2 = new Float32Array(H * V);
  const gb2 = new Float32Array(V);
  const rows = new Int32Array(batch);

  let stepCount = 0;

  function step() {
    const { E, W1, b1, W2, b2 } = p;
    for (let b = 0; b < batch; b++) rows[b] = Math.floor(rand() * nPos);

    // ---- forward
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      const ctx = X[rows[b]];
      const eo = b * IN, ho = b * H, po = b * V;
      for (let i = 0; i < IN; i++) emb[eo + i] = E[ctx[i >> 3] * D + (i & 7)];
      for (let j = 0; j < H; j++) hid[ho + j] = b1[j];
      for (let i = 0; i < IN; i++) {
        const e = emb[eo + i];
        if (e === 0) continue;
        const w1r = i * H;
        for (let j = 0; j < H; j++) hid[ho + j] += e * W1[w1r + j];
      }
      for (let j = 0; j < H; j++) hid[ho + j] = Math.tanh(hid[ho + j]);
      // logits -> probs (into dlog as scratch)
      for (let v = 0; v < V; v++) dlog[po + v] = b2[v];
      for (let j = 0; j < H; j++) {
        const hj = hid[ho + j], w2r = j * V;
        for (let v = 0; v < V; v++) dlog[po + v] += hj * W2[w2r + v];
      }
      let max = -Infinity;
      for (let v = 0; v < V; v++) if (dlog[po + v] > max) max = dlog[po + v];
      let sum = 0;
      for (let v = 0; v < V; v++) { dlog[po + v] = Math.exp(dlog[po + v] - max); sum += dlog[po + v]; }
      for (let v = 0; v < V; v++) dlog[po + v] /= sum;
      loss -= Math.log(dlog[po + y[rows[b]]] + 1e-12);
    }
    loss /= batch;

    // ---- backward (mean CE: dlogits = (probs - onehot) / batch)
    const w = 1 / batch;
    for (let b = 0; b < batch; b++) {
      const po = b * V;
      for (let v = 0; v < V; v++) dlog[po + v] *= w;
      dlog[po + y[rows[b]]] -= w;
    }
    gW2.fill(0); gb2.fill(0);
    for (let b = 0; b < batch; b++) {
      const ho = b * H, po = b * V;
      for (let j = 0; j < H; j++) {
        const hj = hid[ho + j], w2r = j * V;
        for (let v = 0; v < V; v++) gW2[w2r + v] += hj * dlog[po + v];
      }
      for (let v = 0; v < V; v++) gb2[v] += dlog[po + v];
    }
    // dz1 = (dlogits @ W2^T) * (1 - h^2)
    for (let b = 0; b < batch; b++) {
      const ho = b * H, po = b * V;
      for (let j = 0; j < H; j++) {
        let dh = 0;
        const w2r = j * V;
        for (let v = 0; v < V; v++) dh += dlog[po + v] * W2[w2r + v];
        const hj = hid[ho + j];
        dz1[ho + j] = dh * (1 - hj * hj);
      }
    }
    gW1.fill(0); gb1.fill(0); gE.fill(0);
    for (let b = 0; b < batch; b++) {
      const ctx = X[rows[b]];
      const eo = b * IN, ho = b * H;
      for (let i = 0; i < IN; i++) {
        const e = emb[eo + i], w1r = i * H;
        let demb = 0;
        if (e !== 0) {
          for (let j = 0; j < H; j++) {
            gW1[w1r + j] += e * dz1[ho + j];
            demb += dz1[ho + j] * W1[w1r + j];
          }
        } else {
          for (let j = 0; j < H; j++) demb += dz1[ho + j] * W1[w1r + j];
        }
        gE[ctx[i >> 3] * D + (i & 7)] += demb;
      }
      for (let j = 0; j < H; j++) gb1[j] += dz1[ho + j];
    }

    // ---- SGD + momentum update (model.make_sgd)
    const grads = { E: gE, W1: gW1, b1: gb1, W2: gW2, b2: gb2 };
    for (const k of PARAM_ORDER) {
      const pk = p[k], vk = vel[k], gk = grads[k];
      for (let i = 0; i < pk.length; i++) {
        vk[i] = momentum * vk[i] - lr * gk[i];
        pk[i] += vk[i];
      }
    }
    stepCount++;
    return loss;
  }

  return {
    params: p,
    demos,
    config: { lr, batch, steps, seed, maskPrompt, momentum },
    nLossPositions: nPos,
    nAnswerPositions: answerSet.y.length,
    get stepCount() { return stepCount; },
    step,
    run(n = steps) { const losses = []; for (let i = 0; i < n; i++) losses.push(step()); return losses; },
    // mean CE over ALL answer-line positions (expA.full_answer_loss /
    // expC.answer_line_loss — always the masked position set)
    lossFullAnswerSet() {
      let total = 0;
      for (let i = 0; i < answerSet.y.length; i++) {
        const probs = probsFromLogits(forward(p, answerSet.X[i]));
        total -= Math.log(probs[answerSet.y[i]] + 1e-12);
      }
      return total / answerSet.y.length;
    },
    quizAccuracy(questions, answers) { return quizAccuracy(p, questions, answers); },
    libraryScore(facts) { return libraryScore(p, facts); },
  };
}
