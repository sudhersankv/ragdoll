const LIMITS = {
  head: 8,
  hic: 250,
  clearance: 2,
  landing: 0.65,
  score: 82,
  framing: 85,
};

function bestOf(wave) {
  return wave.results.find((result) => result.ghost_id === wave.best_ghost_id) || wave.results[0];
}

function metric(wave, key) {
  return bestOf(wave)?.metrics?.[key] ?? 0;
}

function Chart({ title, values, limit, max, digits }) {
  const width = 320;
  const height = 96;
  const pad = 8;
  const top = Math.max(max, limit, ...values, 1);
  const coords = values.map((value, index) => {
    const x = pad + (values.length === 1 ? (width - pad * 2) / 2 : (index / (values.length - 1)) * (width - pad * 2));
    const y = height - pad - (value / top) * (height - pad * 2);
    return [x, y];
  });
  const limitY = height - pad - (limit / top) * (height - pad * 2);
  const line = coords.map((point) => point.join(',')).join(' ');
  return (
    <figure>
      <figcaption className="text-xs text-neutral-400">{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-1 w-full h-24" role="img" aria-label={title}>
        <line x1={pad} x2={width - pad} y1={limitY} y2={limitY} stroke="#a3a3a3" strokeDasharray="3 3" />
        <polyline fill="none" stroke="#f5f5f4" strokeWidth="2" points={line} />
        {coords.map((point, index) => (
          <circle key={index} cx={point[0]} cy={point[1]} r="3" fill="#fbbf24" />
        ))}
      </svg>
      <p className="text-xs text-neutral-500 tabular-nums">
        {values.map((value) => value.toFixed(digits)).join('  ·  ')} · limit {limit}
      </p>
    </figure>
  );
}

function Row({ label, value, limit, pass, unit }) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto] gap-3 py-2 border-b border-neutral-800 text-sm">
      <span className="text-neutral-300">{label}</span>
      <span className="tabular-nums text-neutral-100">{value}{unit}</span>
      <span className={pass ? 'text-neutral-400' : 'text-red-300'}>{pass ? 'Pass' : 'Fail'} · {limit}{unit}</span>
    </div>
  );
}

export default function CritiquePanel({ waves, onClose, onNext, busy }) {
  const latest = waves[waves.length - 1];
  const wave = latest?.wave;
  const best = wave ? bestOf(wave) : null;
  const metrics = best?.metrics || {};
  const scores = waves.map((entry) => entry.wave.best_score);
  const heads = waves.map((entry) => metric(entry.wave, 'peak_head_g'));
  const clearances = waves.map((entry) => metric(entry.wave, 'min_clearance_cm'));

  return (
    <section className="absolute inset-y-0 right-0 z-20 w-[420px] max-w-full bg-neutral-950 border-l border-neutral-800 flex flex-col">
      <header className="h-14 px-4 flex items-center justify-between border-b border-neutral-800">
        <h2 className="text-sm font-semibold text-white text-balance">Critique · wave {wave?.wave_index || 1}</h2>
        <button type="button" onClick={onClose} className="text-sm text-neutral-300">Close</button>
      </header>
      <div className="flex-1 overflow-auto px-4 py-4">
        {best && (
          <>
            <p className="text-2xl font-semibold tabular-nums text-white">{wave.best_score}<span className="text-sm font-normal text-neutral-500"> / 100</span></p>
            <p className="mt-1 text-xs text-neutral-400 text-pretty">{best.diagnostic}</p>
            <div className="mt-4">
              <Row label="Head load" value={metrics.peak_head_g} limit={LIMITS.head} unit=" g" pass={metrics.peak_head_g < LIMITS.head} />
              <Row label="HIC15" value={metrics.hic15} limit={LIMITS.hic} unit="" pass={metrics.hic15 < LIMITS.hic} />
              <Row label="Clearance" value={metrics.min_clearance_cm} limit={LIMITS.clearance} unit=" cm" pass={metrics.min_clearance_cm >= LIMITS.clearance} />
              <Row label="Landing error" value={metrics.landing_error_m} limit={LIMITS.landing} unit=" m" pass={metrics.landing_error_m <= LIMITS.landing} />
              <Row label="Framing" value={metrics.camera_framing_pct} limit={LIMITS.framing} unit="%" pass={metrics.camera_framing_pct >= LIMITS.framing} />
              <Row label="Score" value={wave.best_score} limit={LIMITS.score} unit="" pass={Number(wave.best_score) >= LIMITS.score} />
            </div>
            <div className="mt-6 space-y-4">
              <Chart title="Score by wave" values={scores} limit={LIMITS.score} max={100} digits={0} />
              <Chart title="Peak head load" values={heads} limit={LIMITS.head} max={12} digits={2} />
              <Chart title="Clearance" values={clearances} limit={LIMITS.clearance} max={20} digits={1} />
            </div>
            <ol className="mt-6">
              {wave.results.map((result) => (
                <li key={result.ghost_id} className="flex justify-between py-1.5 text-sm border-b border-neutral-900">
                  <span className="text-neutral-300">{result.label}</span>
                  <span className="tabular-nums text-neutral-100">{result.stunt_score} · {result.gate_1_passed ? 'Gate 1 pass' : 'Gate 1 fail'}</span>
                </li>
              ))}
            </ol>
            <div className="mt-6">
              <h3 className="text-xs text-neutral-400">Critic</h3>
              <p className="mt-1 text-sm text-neutral-200 text-pretty">
                {latest.critique?.assistant_critique || 'The critic has not answered this wave yet.'}
              </p>
              {latest.critique?.key_mutations?.length > 0 && (
                <ul className="mt-3 space-y-1">
                  {latest.critique.key_mutations.map((note) => (
                    <li key={note} className="text-xs text-neutral-400 text-pretty">{note}</li>
                  ))}
                </ul>
              )}
            </div>
          </>
        )}
      </div>
      <div className="border-t border-neutral-800 p-3">
        <button
          type="button"
          onClick={onNext}
          disabled={busy}
          className="w-full bg-amber-400 hover:bg-amber-300 disabled:opacity-40 text-neutral-950 font-semibold text-sm py-2.5 rounded-md"
        >
          {busy ? 'Running…' : 'Run the next wave'}
        </button>
      </div>
    </section>
  );
}
