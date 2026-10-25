import React, { useRef, useState } from 'react';
import WebGLViewport from './WebGLViewport.jsx';

const API_BASE = '';

function fmtMb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fullSize(obs) {
  return obs.size.map((value) => (Number(value) * 2).toFixed(2)).join(' × ');
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
    setStatus(extra.env_prompt ? 'Argon is checking the video again.' : 'Uploading the video. Argon watches it once the file is ready.');
    try {
      const next = await postVideo(file, extra);
      if (next.alignment_status === 'replica_failed' || !(next.scene_obstacles || []).length) {
        setScene(null);
        setError(next.alignment_notes || 'Argon did not return a location.');
        setStatus('');
        return;
      }
      setScene(next);
      setStatus('');
      setPrompt('');
    } catch (err) {
      setScene(null);
      setError(err.message || 'The location replica failed.');
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
    setVideoMeta({ name: file.name, size: file.size, mime: file.type || 'video/mp4' });
    setScene(null);
    await replicate(file);
  }

  async function correctReplica() {
    const text = prompt.trim();
    if (!text || !fileRef.current || !scene || busy) return;
    await replicate(fileRef.current, {
      env_prompt: text,
      current_obstacles: scene.scene_obstacles,
    });
  }

  const obstacles = scene?.scene_obstacles || [];

  return (
    <div className="min-h-dvh bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900 px-5 py-3 flex items-center justify-between gap-3">
        <div>
          <h1 className="font-bold text-base text-white text-balance">RAGDOLL</h1>
          <p className="text-xs text-slate-400 text-pretty">Location video in. MuJoCo replica of that place out. No body yet.</p>
        </div>
        {videoMeta && (
          <div className="text-right text-[11px] font-mono text-slate-400 tabular-nums">
            <div className="text-slate-200 truncate max-w-[240px]">{videoMeta.name}</div>
            <div>{fmtMb(videoMeta.size)}</div>
          </div>
        )}
      </header>

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 flex flex-col gap-3">
        {!scene && (
          <section className="flex-1 min-h-[70dvh] border border-dashed border-slate-700 rounded-2xl bg-slate-900 flex flex-col items-center justify-center text-center px-6">
            <h2 className="text-2xl font-semibold text-white text-balance">Upload the location</h2>
            <p className="mt-2 max-w-md text-sm text-slate-400 text-pretty">
              Gemini 4 Argon watches the whole video and builds boxes for the objects that are actually there.
            </p>
            <label className="mt-6 cursor-pointer bg-amber-400 hover:bg-amber-300 text-slate-950 font-bold text-sm px-5 py-3 rounded-lg">
              {busy ? 'Building the replica…' : 'Choose a video'}
              <input
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/*"
                className="hidden"
                disabled={busy}
                onChange={(event) => onFile(event.target.files?.[0])}
              />
            </label>
            {status && <p className="mt-4 text-sm text-amber-300 text-pretty max-w-md">{status}</p>}
            {error && <p role="alert" className="mt-4 max-w-lg text-sm text-rose-300 text-pretty">{error}</p>}
          </section>
        )}

        {scene && (
          <>
            <section className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3 min-h-0">
              <div className="h-[68dvh] min-h-[420px]">
                <WebGLViewport
                  ref={viewportRef}
                  envScene={scene}
                  frameIndex={0}
                  selectedGhostId=""
                  showAllGhosts={false}
                  cameraMode="match"
                />
              </div>
              <aside className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col gap-3 min-h-0">
                <div>
                  <h2 className="text-sm font-semibold text-white text-balance">{scene.name || 'Location replica'}</h2>
                  <p className="mt-1 text-xs text-slate-400 text-pretty">{scene.description}</p>
                </div>
                <p className="text-[11px] text-slate-300 text-pretty">{scene.alignment_notes}</p>
                <ol className="flex-1 overflow-auto space-y-1.5">
                  {obstacles.map((obs) => (
                    <li key={obs.id} className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-2">
                      <div className="text-xs font-semibold text-slate-100">{String(obs.id).replaceAll('_', ' ')}</div>
                      <div className="mt-0.5 text-[11px] font-mono tabular-nums text-slate-400">
                        {fullSize(obs)} m
                      </div>
                    </li>
                  ))}
                </ol>
                <label className="text-[11px] font-mono text-slate-500 cursor-pointer">
                  Replace video
                  <input
                    type="file"
                    accept="video/mp4,video/webm,video/quicktime,video/*"
                    className="hidden"
                    disabled={busy}
                    onChange={(event) => onFile(event.target.files?.[0])}
                  />
                </label>
              </aside>
            </section>

            <section className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <label htmlFor="replica-correction" className="text-xs font-semibold text-slate-200">
                Correct the replica against the same video
              </label>
              <div className="mt-2 flex gap-2">
                <input
                  id="replica-correction"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') correctReplica();
                  }}
                  disabled={busy}
                  placeholder="The table is wider, and there is a railing on the left"
                  className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-amber-400"
                />
                <button
                  type="button"
                  onClick={correctReplica}
                  disabled={busy || !prompt.trim()}
                  className="bg-amber-400 hover:bg-amber-300 disabled:opacity-40 text-slate-950 font-bold text-xs px-3 rounded-lg"
                >
                  {busy ? 'Checking…' : 'Apply'}
                </button>
              </div>
              {status && <p className="mt-2 text-xs text-amber-300">{status}</p>}
              {error && <p role="alert" className="mt-2 text-xs text-rose-300 text-pretty">{error}</p>}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
