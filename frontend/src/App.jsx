import React, { useEffect, useRef, useState } from 'react';
import WebGLViewport from './WebGLViewport.jsx';
import CritiquePanel from './CritiquePanel.jsx';

const API_BASE = '';

function fmtMb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fullSize(part) {
  return part.size.map((value) => (Number(value) * 2).toFixed(2)).join(' × ');
}

function withoutFrames(wave) {
  return {
    ...wave,
    results: (wave.results || []).map(({ frames, trajectory_points, humanoid_geoms_meta, ...rest }) => rest),
  };
}

export default function App() {
  const viewportRef = useRef(null);
  const fileRef = useRef(null);

  const [videoMeta, setVideoMeta] = useState(null);
  const [scene, setScene] = useState(null);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [stuntPrompt, setStuntPrompt] = useState('');
  const [waves, setWaves] = useState([]);
  const [critiqueOpen, setCritiqueOpen] = useState(false);
  const [frame, setFrame] = useState(0);

  const latest = waves[waves.length - 1]?.wave || null;

  useEffect(() => {
    const count = latest?.results?.[0]?.frames?.length || 0;
    if (!count) return undefined;
    const timer = window.setInterval(() => {
      setFrame((current) => (current + 1) % count);
    }, 32);
    return () => window.clearInterval(timer);
  }, [latest]);

  async function postVideo(file, extra = {}) {
    const body = new FormData();
    body.append('video', file, file.name);
    body.append('stunt_brief', 'Replicate this location. Do not add a person.');
    if (extra.env_prompt) body.append('env_prompt', extra.env_prompt);
    if (extra.current_obstacles) body.append('current_obstacles', JSON.stringify(extra.current_obstacles));
    const res = await fetch(`${API_BASE}/api/scene-from-video`, { method: 'POST', body });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(payload.detail || `The server rejected the video (${res.status}).`);
    }
    return payload;
  }

  async function replicate(file, extra = {}) {
    setBusy(true);
    setError('');
    setStatus(extra.env_prompt ? 'Checking the video again.' : 'Watching the video and building the set.');
    try {
      const next = await postVideo(file, extra);
      if (next.alignment_status === 'replica_failed' || !(next.scene_obstacles || []).length) {
        setScene(null);
        setError(next.alignment_notes || 'The location did not come back.');
        setStatus('');
        return;
      }
      setScene(next);
      setWaves([]);
      setCritiqueOpen(false);
      setStatus('');
      setPrompt('');
    } catch (err) {
      setScene(null);
      setError(err.message || 'The location did not build.');
      setStatus('');
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file) {
    if (!file) return;
    const looksLikeVideo = file.type.startsWith('video/') || /\.(mp4|mov|webm|m4v)$/i.test(file.name);
    if (!looksLikeVideo) {
      setError('Choose a video of the location.');
      return;
    }
    fileRef.current = file;
    setVideoMeta({ name: file.name, size: file.size });
    setScene(null);
    setWaves([]);
    setCritiqueOpen(false);
    await replicate(file);
  }

  async function correctReplica() {
    const text = prompt.trim();
    if (!text || !fileRef.current || !scene || busy) return;
    await replicate(fileRef.current, {
      env_prompt: text,
      current_obstacles: scene.scene_parts || scene.scene_obstacles,
    });
  }

  async function readCritique(wave) {
    const res = await fetch(`${API_BASE}/api/argon-critic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        wave_data: withoutFrames(wave),
        stunt_brief: stuntPrompt.trim(),
        director_steer: stuntPrompt.trim(),
        history: waves.map((entry) => ({
          wave_index: entry.wave.wave_index,
          best_score: entry.wave.best_score,
          gate_1_passed: entry.wave.any_gate_1_passed,
        })),
      }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || 'The critic did not answer.');
    return payload;
  }

  async function runStunt(candidates, waveIndex) {
    const brief = stuntPrompt.trim();
    if (!brief || !scene || busy) {
      setError('Describe the stunt first.');
      return;
    }
    setBusy(true);
    setError('');
    setStatus(waveIndex > 1 ? `Running wave ${waveIndex}.` : 'Running the stunt in this set.');
    setFrame(0);
    try {
      const simRes = await fetch(`${API_BASE}/api/simulate-wave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_index: waveIndex,
          scene_obstacles: scene.scene_obstacles,
          start_pos: scene.start_pos,
          camera: scene.camera,
          candidates: candidates || scene.wave_1_candidates,
          duration_sec: 2.4,
          enable_local_refinement: waveIndex >= 2,
        }),
      });
      const wave = await simRes.json().catch(() => ({}));
      if (!simRes.ok) throw new Error(wave.detail || 'The stunt did not run.');
      setWaves((current) => [...current, { wave, critique: null }]);
      setStatus('');
      setBusy(false);
      try {
        const critique = await readCritique(wave);
        setWaves((current) => current.map((entry) => (entry.wave === wave ? { wave, critique } : entry)));
      } catch (err) {
        setWaves((current) => current.map((entry) => (
          entry.wave === wave
            ? { wave, critique: { assistant_critique: err.message, key_mutations: [], gate_2_passed: false } }
            : entry
        )));
      }
      return;
    } catch (err) {
      setError(err.message || 'The stunt did not run.');
      setStatus('');
    } finally {
      setBusy(false);
    }
  }

  async function nextWave() {
    const last = waves[waves.length - 1];
    if (!last || busy) return;
    let candidates = last.critique?.next_candidates;
    if (!candidates) {
      setStatus('Reading this wave before the next one.');
      try {
        const critique = await readCritique(last.wave);
        candidates = critique.next_candidates;
        setWaves((current) => current.map((entry) => (entry.wave === last.wave ? { wave: last.wave, critique } : entry)));
      } catch (err) {
        setError(err.message || 'The critic did not answer.');
        setStatus('');
        return;
      }
    }
    await runStunt(candidates, (last.wave.wave_index || 1) + 1);
  }

  const parts = scene?.scene_parts?.length ? scene.scene_parts : scene?.scene_obstacles || [];
  const viewScene = latest
    ? { ...scene, ...latest, scene_parts: scene.scene_parts, scene_obstacles: scene.scene_obstacles }
    : scene;

  return (
    <div className="h-dvh bg-neutral-950 text-neutral-100 flex flex-col">
      <header className="h-14 shrink-0 border-b border-neutral-800 px-5 flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-sm font-semibold text-white text-balance">RAGDOLL</h1>
          <p className="text-xs text-neutral-400 text-pretty">Environment</p>
        </div>
        <div className="flex items-center gap-4">
          {latest && (
            <button type="button" onClick={() => setCritiqueOpen(true)} className="text-sm text-neutral-100 tabular-nums">
              Critique · {latest.best_score}
            </button>
          )}
          {videoMeta && (
            <p className="text-xs text-neutral-400 tabular-nums truncate max-w-[220px]">
              {videoMeta.name} · {fmtMb(videoMeta.size)}
            </p>
          )}
          {scene && (
            <label className="text-xs text-neutral-200 cursor-pointer">
              Replace
              <input
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/*"
                className="hidden"
                disabled={busy}
                onChange={(event) => onFile(event.target.files?.[0])}
              />
            </label>
          )}
        </div>
      </header>

      {!scene && (
        <main className="flex-1 flex flex-col items-center justify-center px-6 text-center">
          <h2 className="text-2xl font-semibold text-white text-balance">Build the location</h2>
          <p className="mt-2 max-w-md text-sm text-neutral-400 text-pretty">
            Upload a video of the place. Argon turns it into a solid 3D set you can orbit.
          </p>
          <label className="mt-6 cursor-pointer bg-amber-400 hover:bg-amber-300 text-neutral-950 font-semibold text-sm px-4 py-2.5 rounded-md">
            {busy ? 'Building…' : 'Upload a video'}
            <input
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/*"
              className="hidden"
              disabled={busy}
              onChange={(event) => onFile(event.target.files?.[0])}
            />
          </label>
          {status && <p className="mt-4 text-sm text-neutral-300 text-pretty max-w-md">{status}</p>}
          {error && <p role="alert" className="mt-4 max-w-lg text-sm text-red-300 text-pretty">{error}</p>}
        </main>
      )}

      {scene && (
        <main className="flex-1 min-h-0 flex relative">
          <div className="flex-1 min-w-0 relative">
            <WebGLViewport
              ref={viewportRef}
              envScene={viewScene}
              frameIndex={frame}
              selectedGhostId={latest?.best_ghost_id || ''}
              showAllGhosts
              cameraMode="match"
            />
            {status && <p className="absolute left-4 bottom-4 text-sm text-neutral-200">{status}</p>}
          </div>
          <aside className="w-80 shrink-0 border-l border-neutral-800 flex flex-col min-h-0">
            <div className="px-4 py-4 border-b border-neutral-800">
              <h2 className="text-sm font-semibold text-white text-balance">{scene.name || 'Location'}</h2>
              <p className="mt-1 text-xs text-neutral-400 text-pretty">{scene.description}</p>
            </div>
            <form
              className="px-4 py-3 border-b border-neutral-800"
              onSubmit={(event) => {
                event.preventDefault();
                runStunt(null, 1);
              }}
            >
              <label htmlFor="stunt-prompt" className="text-xs text-neutral-400">Stunt</label>
              <textarea
                id="stunt-prompt"
                value={stuntPrompt}
                onChange={(event) => setStuntPrompt(event.target.value)}
                disabled={busy}
                rows={3}
                placeholder="Vault the sofa and land on the open floor"
                className="mt-2 w-full bg-neutral-950 border border-neutral-700 rounded-md px-2.5 py-2 text-sm focus:outline-none focus:border-amber-400"
              />
              <button
                type="submit"
                disabled={busy || !stuntPrompt.trim()}
                className="mt-2 w-full bg-amber-400 hover:bg-amber-300 disabled:opacity-40 text-neutral-950 font-semibold text-sm py-2 rounded-md"
              >
                {busy ? 'Running…' : 'Run'}
              </button>
              {error && <p role="alert" className="mt-2 text-xs text-red-300 text-pretty">{error}</p>}
            </form>
            <ol className="flex-1 overflow-auto px-2 py-2">
              {parts.map((part) => (
                <li key={part.id} className="px-2 py-2 border-b border-neutral-900">
                  <div className="text-sm text-neutral-100">{String(part.id).replaceAll('_', ' ')}</div>
                  <div className="mt-0.5 text-xs text-neutral-500 tabular-nums">
                    {part.material || 'solid'} · {fullSize(part)} m
                  </div>
                </li>
              ))}
            </ol>
            <form
              className="border-t border-neutral-800 p-3"
              onSubmit={(event) => {
                event.preventDefault();
                correctReplica();
              }}
            >
              <label htmlFor="replica-correction" className="text-xs text-neutral-500">Correct the set</label>
              <div className="mt-2 flex gap-2">
                <input
                  id="replica-correction"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  disabled={busy}
                  placeholder="The screen is wider"
                  className="flex-1 min-w-0 bg-neutral-950 border border-neutral-700 rounded-md px-2.5 py-2 text-sm focus:outline-none focus:border-amber-400"
                />
                <button
                  type="submit"
                  disabled={busy || !prompt.trim()}
                  className="text-sm text-neutral-200 px-2"
                >
                  Apply
                </button>
              </div>
            </form>
          </aside>
          {critiqueOpen && waves.length > 0 && (
            <CritiquePanel waves={waves} busy={busy} onClose={() => setCritiqueOpen(false)} onNext={nextWave} />
          )}
        </main>
      )}
    </div>
  );
}
