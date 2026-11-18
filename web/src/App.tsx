import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Stage from "./Stage";
import type { StageControls } from "./Stage";
import "./style.css";

type Round = { round: number };
type Verdict = { correct: boolean; reply: string; subject: string | null; guesses_left?: number };
type Reveal = { won: boolean; subject: string; reply: string };

const ROUND_MS = 30_000;

function decodeReplay(buffer: ArrayBuffer) {
  const header = new Int32Array(buffer, 0, 2);
  const [count, vertices] = header;
  const raw = new Int16Array(buffer, 8);
  const floats = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) floats[i] = raw[i] / 10000;
  const side = vertices * 3;
  const frames: Float32Array[][] = [];
  for (let f = 0; f < count; f++) {
    frames.push([
      floats.subarray(f * 2 * side, (f * 2 + 1) * side),
      floats.subarray((f * 2 + 1) * side, (f * 2 + 2) * side),
    ]);
  }
  return frames;
}

function blend(frames: Float32Array[][], t: number) {
  const pos = t * (frames.length - 1);
  const i = Math.floor(pos);
  const f = pos - i;
  const a = frames[i];
  const b = frames[Math.min(i + 1, frames.length - 1)];
  return a.map((ma, side) => {
    const mb = b[side];
    const out = new Float32Array(ma.length);
    for (let j = 0; j < ma.length; j++) out[j] = ma[j] + (mb[j] - ma[j]) * f;
    return out;
  });
}

function App() {
  const [error, setError] = useState("");
  const [guess, setGuess] = useState("");
  const [notice, setNotice] = useState("");
  const [judging, setJudging] = useState(false);
  const [active, setActive] = useState(false);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const stage = useRef<StageControls | null>(null);
  const slot = useRef<number | null>(null);
  const previous = useRef<string[]>([]);
  const runId = useRef(0);
  const timers = useRef<number[]>([]);
  const finished = useRef(false);
  const bare = new URLSearchParams(window.location.search).has("bare");
  const grid = new URLSearchParams(window.location.search).has("grid");

  const clearTimers = () => {
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current = [];
  };
  useEffect(
    () => () => {
      timers.current.forEach((timer) => window.clearTimeout(timer));
    },
    [],
  );

  const onReady = useCallback((controls: StageControls) => {
    stage.current = controls;
    setReady(true);
    if (!new URLSearchParams(window.location.search).has("bare")) return;
    if (window.parent !== window) {
      window.parent.postMessage("shadow:ready", "*");
      window.addEventListener("message", (event) => {
        if (event.data === "shadow:start") start();
      });
    } else start();
  }, []);

  async function start() {
    if (!stage.current) return;
    clearTimers();
    finished.current = false;
    const run = ++runId.current;
    slot.current = null;
    previous.current = [];
    setError("");
    setGuess("");
    setNotice("");
    setJudging(false);
    setReveal(null);
    setActive(false);
    setLoading(true);
    try {
      const round = (await (await fetch("/api/round")).json()) as Round;
      const frames = decodeReplay(
        await (await fetch(`/replays/${round.round}.bin`)).arrayBuffer(),
      );
      slot.current = round.round;
      setActive(true);
      stage.current.reset();
      const t0 = performance.now();
      const tick = (now: number) => {
        if (finished.current || runId.current !== run) return;
        const t = Math.max(0, Math.min(1, (now - t0) / ROUND_MS));
        stage.current?.pose(blend(frames, t), true);
        if (t < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      timers.current.push(
        window.setTimeout(() => {
          if (runId.current === run) {
            setLoading(false);
            setNotice((n) => n || "Any guesses?");
          }
        }, ROUND_MS),
      );
    } catch {
      setLoading(false);
      setError("Could not load the replay. Please try again.");
    }
  }

  async function submit(giveUp = false) {
    const text = guess.trim();
    if (slot.current === null || reveal || judging || (!giveUp && !text)) return;
    setJudging(true);
    setError("");
    try {
      const response = await fetch("/api/guess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          round: slot.current,
          guess: text,
          give_up: giveUp,
          previous: previous.current,
        }),
      });
      if (!response.ok) throw new Error("guess failed");
      const verdict = (await response.json()) as Verdict;
      if (giveUp || verdict.correct || verdict.subject !== null) {
        finished.current = true;
        setReveal({
          won: verdict.correct,
          subject: verdict.subject ?? "",
          reply: verdict.reply,
        });
        setActive(false);
        setLoading(false);
      } else {
        previous.current = [...previous.current, text];
        setNotice(verdict.reply);
      }
      if (!giveUp) setGuess("");
    } catch {
      setError("Could not check that guess. Please try again.");
    } finally {
      setJudging(false);
    }
  }

  if (grid) return <Grid />;

  return (
    <main>
      <Stage onError={setError} onReady={onReady} />
      <div className="game-control">
        {loading && !bare && (
          <svg className="spin" aria-label="Loading replay" role="status" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.2-8.56" /></svg>
        )}
        {!bare && (!active || reveal) && !judging && !loading && (
          <button type="button" className="start" disabled={!ready} onClick={start}>
            {reveal || slot.current !== null ? "Play again" : "Start"}
          </button>
        )}
        {judging && (
          <svg className="spin" aria-label="Judging" role="status" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.2-8.56" /></svg>
        )}
      </div>
      {!bare && <header>
        <div className="head-meta">
          <span className="eyebrow"><a href="https://modal.com/library/deepseek/deepseek-v4-1-flash-1017" target="_blank" rel="noreferrer">DeepSeek V4.1 Flash 1017</a></span>
          <span className="modal-line">on <a className="modal-link" href="https://modal.com" target="_blank" rel="noreferrer" aria-label="Modal"><img className="modal-logo" src="/modal-logo.png" alt="Modal" /></a></span>
        </div>
      </header>}
      {!bare && (active || reveal || error) && (
        <form className="prompt" onSubmit={(event) => { event.preventDefault(); submit(); }}>
          <div className={`reply${reveal ? " revealed" : ""}`} role="status">
            {reveal ? reveal.reply : notice || " "}
          </div>
          {active && !reveal && (
            <div className="composer">
              <input aria-label="Guess the shadow" placeholder="Your guess…" maxLength={200} enterKeyHint="send" value={guess} onChange={(event) => setGuess(event.target.value)} />
              <button aria-label="Send" disabled={!guess.trim() || judging}>{judging ? <svg className="spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.2-8.56" /></svg> : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 2 11 13" /><path d="M22 2 15 22l-4-9-9-4Z" /></svg>}</button>
              <button type="button" className="give-up" aria-label="Give up" disabled={judging} onClick={() => submit(true)}><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 22V3c5-4 11 4 16 0v12c-5 4-11-4-16 0" fill="white" /><path d="M4 15V3" /></svg></button>
            </div>
          )}
          {error && <div className="speech-error" role="alert">{error}</div>}
        </form>
      )}
    </main>
  );
}

function Grid() {
  const [, setError] = useState("");
  const stages = useRef<(StageControls | null)[]>(Array.from({ length: 9 }));
  const ready = useMemo(
    () =>
      Array.from({ length: 9 }, (_, i) => (controls: StageControls) => {
        stages.current[i] = controls;
      }),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      const { rounds } = (await (await fetch("/api/grid")).json()) as { rounds: number[] };
      const sets = await Promise.all(
        rounds.map(async (slot) =>
          decodeReplay(await (await fetch(`/replays/${slot}.bin`)).arrayBuffer()),
        ),
      );
      if (cancelled) return;
      stages.current.forEach((s) => s?.reset());
      const t0 = performance.now();
      const tick = (now: number) => {
        if (cancelled) return;
        const t = Math.max(0, Math.min(1, (now - t0) / ROUND_MS));
        sets.forEach((frames, k) => stages.current[k]?.pose(blend(frames, t), true));
        if (t < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };
    const poll = () => {
      if (cancelled) return;
      if (stages.current.every(Boolean)) run().catch(() => setError("Could not load the replays."));
      else window.setTimeout(poll, 100);
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="grid">
      {ready.map((onReady, i) => <Stage key={i} onError={setError} onReady={onReady} />)}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
