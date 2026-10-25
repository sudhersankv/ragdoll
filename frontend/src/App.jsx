import React, { useState, useEffect, useRef } from 'react';
import WebGLViewport from './WebGLViewport.jsx';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const viewportRef = useRef(null);

  const [presets, setPresets] = useState({});
  const [activePresetId, setActivePresetId] = useState('couch_vault');
  const [stuntBrief, setStuntBrief] = useState(
    'Launch over the couch, clear the backrest, and land on the crash mat without head/neck injury.'
  );
  const [directorSteer, setDirectorSteer] = useState('');

  // Wave & Harness State
  const [waveHistory, setWaveHistory] = useState([]);
  const [activeWaveIdx, setActiveWaveIdx] = useState(0);
  const [selectedGhostId, setSelectedGhostId] = useState('ghost_C');
  const [showAllGhosts, setShowAllGhosts] = useState(true);
  const [isSimulating, setIsSimulating] = useState(false);
  const [autoRunning, setAutoRunning] = useState(false);
  const [directorApproved, setDirectorApproved] = useState(false);

  // Playback & Bullet-Time Controls
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.5);
  const [cameraMode, setCameraMode] = useState('match');

  // Ticket 04: Location Scout Video/Photo + Keyframes + Visual Self-Alignment
  const [bgMediaUrl, setBgMediaUrl] = useState(null);
  const [bgMediaType, setBgMediaType] = useState('image');
  const [extractedKeyframes, setExtractedKeyframes] = useState([]);
  const [alignmentNote, setAlignmentNote] = useState('');
  const [isAligning, setIsAligning] = useState(false);

  // Ticket 05: Multi-Scene Stitcher (3-Shot Sequence)
  const [stitchedShots, setStitchedShots] = useState([null, null, null]);
  const [playingMasterSequence, setPlayingMasterSequence] = useState(false);

  const currentEntry = waveHistory[activeWaveIdx] || null;
  const currentWaveData = currentEntry?.waveData || null;
  const currentCritique = currentEntry?.critique || null;

  const selectedGhost =
    currentWaveData?.results?.find((r) => r.ghost_id === selectedGhostId) ||
    currentWaveData?.results?.[0] ||
    null;

  const totalFrames = selectedGhost?.frames?.length || 150;

  useEffect(() => {
    async function init() {
      try {
        const res = await fetch(`${API_BASE}/api/presets`);
        const data = await res.json();
        setPresets(data.presets || {});
        await runInitialWave('couch_vault', data.presets?.couch_vault);
      } catch (err) {
        console.error('Failed to connect to FALLGUY backend:', err);
      }
    }
    init();
  }, []);

  useEffect(() => {
    if (!isPlaying || !currentWaveData) return;
    const intervalMs = Math.max(14, Math.round(16 / playbackSpeed));
    const timer = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % Math.max(1, totalFrames));
    }, intervalMs);
    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, currentWaveData, totalFrames]);

  async function runInitialWave(presetId, presetObj, customObstacles = null) {
    setIsSimulating(true);
    setDirectorApproved(false);
    try {
      const p = presetObj || presets[presetId];
      if (p?.description && !customObstacles) setStuntBrief(p.description);

      const simRes = await fetch(`${API_BASE}/api/simulate-wave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_index: 1,
          preset_id: presetId,
          scene_obstacles: customObstacles || undefined,
        }),
      });
      const wave1 = await simRes.json();

      const critRes = await fetch(`${API_BASE}/api/argon-critic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_data: wave1,
          stunt_brief: p?.description || stuntBrief,
        }),
      });
      const critique1 = await critRes.json();

      const firstEntry = { waveData: wave1, critique: critique1 };
      setWaveHistory([firstEntry]);
      setActiveWaveIdx(0);
      setSelectedGhostId(wave1.best_ghost_id || 'ghost_C');
      setFrameIndex(0);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSimulating(false);
    }
  }

  async function runNextRecursiveWave(customSteer = null, baseEntryOverride = null) {
    const activeItem = baseEntryOverride || currentEntry;
    if (!activeItem || isSimulating) return null;
    setIsSimulating(true);
    const steerText = customSteer !== null ? customSteer : directorSteer;

    try {
      const snapshotB64 = viewportRef.current?.captureSnapshot?.() || null;

      const critRes = await fetch(`${API_BASE}/api/argon-critic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_data: activeItem.waveData,
          stunt_brief: stuntBrief,
          director_steer: steerText || null,
          canvas_image_b64: snapshotB64,
        }),
      });
      const updatedCritique = await critRes.json();

      const nextWaveNumber = (activeItem.waveData.wave_index || 1) + 1;
      const simRes = await fetch(`${API_BASE}/api/simulate-wave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_index: nextWaveNumber,
          preset_id: activePresetId,
          scene_obstacles: activeItem.waveData.scene_obstacles,
          start_pos: activeItem.waveData.start_pos,
          camera: activeItem.waveData.camera,
          candidates: updatedCritique.next_candidates,
          enable_local_refinement: true,
        }),
      });
      const nextWaveData = await simRes.json();

      const nextCritRes = await fetch(`${API_BASE}/api/argon-critic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wave_data: nextWaveData,
          stunt_brief: stuntBrief,
          director_steer: null,
        }),
      });
      const nextCritique = await nextCritRes.json();

      const newEntry = {
        waveData: nextWaveData,
        critique: nextCritique,
        appliedSteer: steerText || null,
      };

      setWaveHistory((prev) => {
        const updated = [...prev, newEntry];
        setActiveWaveIdx(updated.length - 1);
        return updated;
      });
      setSelectedGhostId(nextWaveData.best_ghost_id);
      setFrameIndex(0);
      return newEntry;
    } catch (err) {
      console.error('Recursive wave error:', err);
      return null;
    } finally {
      setIsSimulating(false);
    }
  }

  // Ticket 02: Auto-Run up to Max 4 Waves, stopping early once Gate 1 & Gate 2 pass
  async function handleAutoHarnessLoop() {
    if (autoRunning || isSimulating || !currentEntry) return;
    setAutoRunning(true);
    let latest = currentEntry;
    while ((latest?.waveData?.wave_index || 1) < 4) {
      if (latest?.critique?.gate_2_passed && latest?.waveData?.any_gate_1_passed) {
        break;
      }
      latest = await runNextRecursiveWave(null, latest);
      if (!latest) break;
      await new Promise((r) => setTimeout(r, 350));
    }
    setAutoRunning(false);
  }

  // Ticket 03: Nudge 3D Obstacle or Crash Mat position directly via UI sliders
  async function handleNudgeObstacle(obsId, axisIdx, newValue) {
    if (!currentWaveData || isSimulating) return;
    const updatedObs = currentWaveData.scene_obstacles.map((obs) => {
      if (obs.id !== obsId) return obs;
      const nextPos = [...obs.pos];
      nextPos[axisIdx] = Number(newValue);
      return { ...obs, pos: nextPos };
    });

    const simRes = await fetch(`${API_BASE}/api/simulate-wave`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        wave_index: currentWaveData.wave_index,
        preset_id: activePresetId,
        scene_obstacles: updatedObs,
        start_pos: currentWaveData.start_pos,
        camera: currentWaveData.camera,
        candidates: currentWaveData.results.map((r) => ({
          ...r.parameters,
          ghost_id: r.ghost_id,
          label: r.label,
          color: r.color,
        })),
        enable_local_refinement: false,
      }),
    });
    const reSim = await simRes.json();
    setWaveHistory((prev) => {
      const copy = [...prev];
      copy[activeWaveIdx] = { ...copy[activeWaveIdx], waveData: reSim };
      return copy;
    });
  }

  // Ticket 04: Upload Video/Photo, Extract Start/Mid/End Keyframes, and Reconstruct 3D Scene
  async function handleMediaUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const isVid = file.type.startsWith('video');
    setBgMediaType(isVid ? 'video' : 'image');
    setBgMediaUrl(url);

    let framesB64 = [];
    if (!isVid) {
      const b64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(file);
      });
      framesB64 = [b64];
    } else {
      framesB64 = await extractVideoKeyframes(url);
    }
    setExtractedKeyframes(framesB64);

    try {
      const res = await fetch(`${API_BASE}/api/scene-from-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keyframes_b64: framesB64,
          stunt_brief: stuntBrief,
        }),
      });
      const sceneOut = await res.json();
      setAlignmentNote(sceneOut.alignment_notes || '3D Scene Blockout extracted.');
      await runInitialWave(activePresetId, sceneOut, sceneOut.scene_obstacles);
    } catch (err) {
      console.error('Scene extraction error:', err);
    }
  }

  async function extractVideoKeyframes(videoUrl) {
    return new Promise((resolve) => {
      const vid = document.createElement('video');
      vid.src = videoUrl;
      vid.muted = true;
      vid.onloadedmetadata = async () => {
        const dur = Math.max(0.5, vid.duration || 2.0);
        const times = [0.05, dur * 0.5, Math.max(0.1, dur - 0.1)];
        const canvas = document.createElement('canvas');
        canvas.width = 480;
        canvas.height = 270;
        const ctx = canvas.getContext('2d');
        const out = [];
        for (const t of times) {
          vid.currentTime = t;
          await new Promise((r) => {
            vid.onseeked = r;
          });
          ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
          out.push(canvas.toDataURL('image/jpeg', 0.7));
        }
        resolve(out);
      };
      vid.onerror = () => resolve([]);
    });
  }

  // Ticket 04: 1-Click Argon Visual Self-Alignment Pass
  async function handleVisualSelfAlignment() {
    if (!currentWaveData || isAligning) return;
    setIsAligning(true);
    try {
      const wireframeSnap = viewportRef.current?.captureSnapshot?.() || 'data:image/jpeg;base64,/9j/4AAQSkZJRg==';
      const res = await fetch(`${API_BASE}/api/scene-from-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keyframes_b64: extractedKeyframes.length ? extractedKeyframes : [wireframeSnap],
          wireframe_overlay_b64: wireframeSnap,
          current_obstacles: currentWaveData.scene_obstacles,
          stunt_brief: stuntBrief,
        }),
      });
      const aligned = await res.json();
      setAlignmentNote(aligned.alignment_notes || 'Visual Self-Alignment Locked.');
      await runInitialWave(activePresetId, aligned, aligned.scene_obstacles);
    } catch (err) {
      console.error('Self-alignment error:', err);
    } finally {
      setIsAligning(false);
    }
  }

  // Ticket 05: Lock & Stitch Multi-Scene Sequence
  function lockShotToTimeline(slotIndex) {
    if (!currentEntry || !selectedGhost) return;
    const updated = [...stitchedShots];
    updated[slotIndex] = {
      slot: slotIndex + 1,
      presetName: presets[activePresetId]?.name || activePresetId,
      waveIndex: currentEntry.waveData.wave_index,
      ghost: selectedGhost,
      waveData: currentEntry.waveData,
    };
    setStitchedShots(updated);
  }

  async function playStitchedMasterSequence() {
    const validShots = stitchedShots.filter(Boolean);
    if (!validShots.length || playingMasterSequence) return;
    setPlayingMasterSequence(true);
    for (const shot of validShots) {
      setWaveHistory((prev) => [
        ...prev,
        {
          waveData: shot.waveData,
          critique: {
            gate_2_passed: true,
            assistant_critique: `Playing Locked Shot #${shot.slot}: ${shot.presetName}`,
          },
        },
      ]);
      setActiveWaveIdx((prev) => prev + 1);
      setSelectedGhostId(shot.ghost.ghost_id);
      setFrameIndex(0);
      await new Promise((r) => setTimeout(r, 2400));
    }
    setPlayingMasterSequence(false);
  }

  function exportSimToRealPolicy() {
    if (!selectedGhost) return;
    const payload = {
      platform: 'FALLGUY_SIM2REAL_V1',
      target_rigs: ['Human_75kg_Techvis', 'Unitree_G1_23DOF'],
      stunt_score: selectedGhost.stunt_score,
      gate_1_passed: selectedGhost.gate_1_passed,
      metrics: selectedGhost.metrics,
      control_parameters: selectedGhost.parameters,
      frames_60hz: selectedGhost.frames,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fallguy_verified_stunt_${selectedGhost.ghost_id}.json`;
    a.click();
  }

  const gate1Passed = Boolean(currentWaveData?.any_gate_1_passed);
  const gate2Passed = Boolean(currentCritique?.gate_2_passed && gate1Passed);

  const stressBadgeColor = (level) => {
    if (level === 'critical') return 'bg-rose-500/20 text-rose-400 border-rose-500/50';
    if (level === 'warn') return 'bg-amber-500/20 text-amber-300 border-amber-500/50';
    return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50';
  };

  return (
    <div className="min-h-screen bg-[#060911] text-slate-100 flex flex-col">
      {/* Top Command Header */}
      <header className="border-b border-slate-800/90 bg-[#0a0f1d]/90 backdrop-blur px-5 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center font-black text-slate-950 text-lg shadow-lg shadow-cyan-500/20">
            FG
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold tracking-tight text-base text-white">FALLGUY</h1>
              <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                Gemini 4 Argon × MuJoCo 500Hz
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Autonomous 3-Gate Recursive Stunt &amp; Biomechanical Safety Harness
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            value={activePresetId}
            onChange={(e) => {
              const pid = e.target.value;
              setActivePresetId(pid);
              runInitialWave(pid, presets[pid]);
            }}
            className="bg-slate-900 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-400"
          >
            {Object.values(presets).map((p) => (
              <option key={p.id} value={p.id}>
                🎬 Scene: {p.name}
              </option>
            ))}
          </select>

          <label className="cursor-pointer bg-slate-800/90 hover:bg-slate-700 border border-slate-700 text-xs px-3 py-2 rounded-lg text-slate-200 transition flex items-center gap-1.5">
            <span>📱 Upload Scout Video/Photo</span>
            <input
              type="file"
              accept="video/*,image/*"
              onChange={handleMediaUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={handleVisualSelfAlignment}
            disabled={isAligning || isSimulating}
            className="bg-slate-800 hover:bg-slate-700 border border-indigo-500/40 text-indigo-300 text-xs font-medium px-3 py-2 rounded-lg transition"
          >
            {isAligning ? '🎯 Aligning...' : '🎯 Argon 1-Step Self-Align'}
          </button>

          <button
            onClick={() => runNextRecursiveWave()}
            disabled={isSimulating || autoRunning}
            className="bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-cyan-500/40 text-cyan-300 text-xs font-semibold px-3.5 py-2 rounded-lg transition"
          >
            {isSimulating ? '⚙️ Simulating...' : '🔄 Step Wave (+1)'}
          </button>

          <button
            onClick={handleAutoHarnessLoop}
            disabled={isSimulating || autoRunning}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 text-slate-950 font-bold text-xs px-4 py-2 rounded-lg shadow-lg shadow-cyan-500/20 transition"
          >
            {autoRunning ? '⚡ Harness Converging...' : '⚡ Auto-Run 3-Gate Harness'}
          </button>
        </div>
      </header>

      {alignmentNote && (
        <div className="bg-indigo-950/50 border-b border-indigo-500/30 px-5 py-1.5 text-xs text-indigo-200 flex items-center justify-between">
          <span>🌐 {alignmentNote}</span>
          <button onClick={() => setAlignmentNote('')} className="text-indigo-400 hover:text-white">
            ✕
          </button>
        </div>
      )}

      {/* Main 12-Column Workspace */}
      <main className="flex-1 grid grid-cols-12 gap-4 p-4 max-w-[1800px] w-full mx-auto">
        {/* LEFT / CENTER (8 Cols) */}
        <section className="col-span-12 lg:col-span-8 flex flex-col gap-3">
          <div className="relative h-[470px] w-full">
            <WebGLViewport
              ref={viewportRef}
              waveData={currentWaveData}
              frameIndex={frameIndex}
              selectedGhostId={selectedGhostId}
              showAllGhosts={showAllGhosts}
              cameraMode={cameraMode}
              backgroundMediaUrl={bgMediaUrl}
              backgroundMediaType={bgMediaType}
            />

            {/* Floating Top-Left HUD */}
            <div className="absolute top-3 left-3 z-20 flex flex-col gap-2 pointer-events-none">
              <div className="bg-slate-950/85 backdrop-blur border border-slate-800 rounded-lg px-3.5 py-2.5 shadow-xl">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono uppercase text-cyan-400 font-bold">
                    WAVE #{currentWaveData?.wave_index || 1} ({currentWaveData?.total_mujoco_rollouts || 4} MuJoCo Rollouts)
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded font-bold bg-slate-800 text-white">
                    Best Score: {currentWaveData?.best_score || 0}%
                  </span>
                </div>
                {selectedGhost && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs">
                    <div>
                      <span className="text-slate-400">Head Load: </span>
                      <span
                        className={`font-mono font-bold ${
                          selectedGhost.metrics.peak_head_g < 15.0
                            ? 'text-emerald-400'
                            : 'text-rose-400'
                        }`}
                      >
                        {selectedGhost.metrics.peak_head_g}g (HIC15: {selectedGhost.metrics.hic15})
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400">Clearance: </span>
                      <span
                        className={`font-mono font-bold ${
                          selectedGhost.metrics.min_clearance_cm >= 2.0
                            ? 'text-cyan-300'
                            : 'text-rose-400'
                        }`}
                      >
                        {selectedGhost.metrics.min_clearance_cm > 0 ? '+' : ''}
                        {selectedGhost.metrics.min_clearance_cm}cm
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400">Mat Error: </span>
                      <span className="font-mono font-bold text-slate-200">
                        {selectedGhost.metrics.landing_error_m}m
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Floating Top-Right View Controls */}
            <div className="absolute top-3 right-3 z-20 flex flex-wrap items-center gap-1.5 bg-slate-950/85 backdrop-blur border border-slate-800 rounded-lg p-1.5">
              <button
                onClick={() => setShowAllGhosts((v) => !v)}
                className={`text-xs px-2.5 py-1 rounded font-medium transition ${
                  showAllGhosts
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {showAllGhosts ? '👻 4 Ghosts Active' : '🎯 Winner Only'}
              </button>
              {['match', 'hero', 'top'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setCameraMode(mode)}
                  className={`text-xs px-2.5 py-1 rounded capitalize transition ${
                    cameraMode === mode
                      ? 'bg-slate-800 text-white font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {mode === 'match' ? '🎥 Cam Match' : mode === 'hero' ? '📐 Hero 3/4' : '🛰️ Top View'}
                </button>
              ))}
            </div>

            {/* Floating Bottom Bullet-Time Scrubber */}
            <div className="absolute bottom-3 inset-x-3 z-20 bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg px-3.5 py-2 flex items-center gap-3">
              <button
                onClick={() => setIsPlaying((p) => !p)}
                className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs px-3 py-1.5 rounded transition"
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>

              <input
                type="range"
                min={0}
                max={Math.max(1, totalFrames - 1)}
                value={frameIndex}
                onChange={(e) => {
                  setIsPlaying(false);
                  setFrameIndex(Number(e.target.value));
                }}
                className="flex-1 accent-cyan-400 cursor-pointer"
              />

              <span className="text-xs font-mono text-slate-300 w-16 text-right">
                {((frameIndex / Math.max(1, totalFrames)) * 2.4).toFixed(2)}s
              </span>

              <div className="flex items-center gap-1 border-l border-slate-800 pl-2">
                {[
                  { label: '0.25x Bullet', val: 0.25 },
                  { label: '0.5x Slow', val: 0.5 },
                  { label: '1x Real', val: 1.0 },
                ].map((spd) => (
                  <button
                    key={spd.val}
                    onClick={() => setPlaybackSpeed(spd.val)}
                    className={`text-[11px] px-2 py-1 rounded ${
                      playbackSpeed === spd.val
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {spd.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* 4 Candidate Ghost Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {(currentWaveData?.results || []).map((ghost) => {
              const isSelected = ghost.ghost_id === selectedGhostId;
              return (
                <div
                  key={ghost.ghost_id}
                  onClick={() => setSelectedGhostId(ghost.ghost_id)}
                  className={`cursor-pointer rounded-xl p-3 border transition ${
                    isSelected
                      ? 'bg-slate-900/95 border-cyan-400 shadow-lg shadow-cyan-500/10'
                      : 'bg-slate-900/50 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full inline-block"
                        style={{ backgroundColor: ghost.color }}
                      />
                      <span className="text-xs font-bold uppercase text-white">
                        {ghost.ghost_id.replace('_', ' ')}
                      </span>
                    </div>
                    <span
                      className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${
                        ghost.gate_1_passed
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : 'bg-rose-500/20 text-rose-300'
                      }`}
                    >
                      {ghost.stunt_score}%
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-300 font-medium mt-1 truncate">
                    {ghost.label}
                  </p>
                  <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>Head: {ghost.metrics.peak_head_g}g</span>
                    <span>Clear: {ghost.metrics.min_clearance_cm}cm</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Ticket 03: Direct 3D Obstacle & Crash-Mat Sliders */}
          {currentWaveData?.scene_obstacles && (
            <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-3.5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
                📐 3D Stage &amp; Crash-Mat Rigging Controls (Live MuJoCo Re-Sim)
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {currentWaveData.scene_obstacles.map((obs) => (
                  <div
                    key={obs.id}
                    className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 flex flex-col gap-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-200">{obs.id}</span>
                      <span className="font-mono text-[11px] text-cyan-400">
                        X: {obs.pos[0].toFixed(2)}m
                      </span>
                    </div>
                    <input
                      type="range"
                      min={obs.id === 'crash_mat' ? 1.1 : -1.2}
                      max={obs.id === 'crash_mat' ? 3.2 : 1.4}
                      step={0.05}
                      value={obs.pos[0]}
                      onChange={(e) => handleNudgeObstacle(obs.id, 0, e.target.value)}
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ticket 05: Multi-Scene Sequence Stitcher */}
          <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-3.5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  🎞️ Multi-Scene Sequence Stitcher (8–10s Action Cut)
                </h3>
                <p className="text-[11px] text-slate-400">
                  Lock verified stunt beats across 3 shots to play a continuous multi-obstacle stunt sequence.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={exportSimToRealPolicy}
                  className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition"
                >
                  📥 Export Trajectory JSON
                </button>
                <button
                  onClick={playStitchedMasterSequence}
                  disabled={!stitchedShots.some(Boolean) || playingMasterSequence}
                  className="text-xs px-3.5 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-slate-950 font-bold transition"
                >
                  {playingMasterSequence ? '▶ Playing Master Cut...' : '▶ Play Stitched Sequence'}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5">
              {[0, 1, 2].map((slotIdx) => {
                const shot = stitchedShots[slotIdx];
                return (
                  <div
                    key={slotIdx}
                    className="rounded-lg border border-slate-800 bg-slate-950/60 p-2.5 flex items-center justify-between"
                  >
                    {shot ? (
                      <div>
                        <div className="text-xs font-bold text-emerald-400">
                          Shot #{slotIdx + 1}: {shot.presetName}
                        </div>
                        <div className="text-[11px] font-mono text-slate-400">
                          Wave {shot.waveIndex} • Score {shot.ghost.stunt_score}% •{' '}
                          {shot.ghost.metrics.peak_head_g}g
                        </div>
                      </div>
                    ) : (
                      <span className="text-xs text-slate-500">
                        Shot #{slotIdx + 1} Empty (2.4s Beat)
                      </span>
                    )}
                    <button
                      onClick={() => lockShotToTimeline(slotIdx)}
                      className="text-[11px] px-2.5 py-1 rounded bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/30 font-semibold"
                    >
                      {shot ? 'Replace' : '+ Lock Shot'}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* RIGHT PANEL (4 Cols) */}
        <aside className="col-span-12 lg:col-span-4 flex flex-col gap-3">
          {/* 1. 3-Gate Verification Harness Status */}
          <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-2.5">
              🛡️ 3-Gate Stunt Verification Harness
            </h2>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <div>
                  <div className="text-xs font-bold text-white">
                    GATE 1: Deterministic Physics &amp; Safety
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Head &lt; 15g, HIC15 &lt; 250, Clearance &ge; 2cm, On Mat
                  </div>
                </div>
                <span
                  className={`text-xs font-mono font-bold px-2.5 py-1 rounded ${
                    gate1Passed
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  }`}
                >
                  {gate1Passed ? '✅ PASS' : '❌ FAIL'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <div>
                  <div className="text-xs font-bold text-white">
                    GATE 2: Argon Multimodal Visual Critic
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Camera Framing &ge; 85% &amp; Viability Score &ge; 82%
                  </div>
                </div>
                <span
                  className={`text-xs font-mono font-bold px-2.5 py-1 rounded ${
                    gate2Passed
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  }`}
                >
                  {gate2Passed ? '✅ PASS' : '🔄 REFINING'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <div>
                  <div className="text-xs font-bold text-white">
                    GATE 3: Lead Stunt Director Sign-Off
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Human-in-the-Loop Live Steer or Greenlight
                  </div>
                </div>
                <button
                  onClick={() => setDirectorApproved((v) => !v)}
                  className={`text-xs font-mono font-bold px-2.5 py-1 rounded transition ${
                    directorApproved
                      ? 'bg-emerald-500 text-slate-950'
                      : 'bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/40'
                  }`}
                >
                  {directorApproved ? '🎬 LOCKED' : 'Approve Shot'}
                </button>
              </div>
            </div>
          </div>

          {/* 2. Quantitative Self-Improvement Wave Curve */}
          <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                📈 Recursive Self-Improvement Curve
              </h2>
              <span className="text-[11px] font-mono text-cyan-400">
                {waveHistory.length} / 4 Max Auto-Waves
              </span>
            </div>

            <div className="flex items-end gap-2 h-20 pt-2 px-2 bg-slate-950/70 rounded-lg border border-slate-800/80">
              {waveHistory.map((item, idx) => {
                const score = item.waveData.best_score || 0;
                const isCurr = idx === activeWaveIdx;
                return (
                  <button
                    key={idx}
                    onClick={() => {
                      setActiveWaveIdx(idx);
                      setSelectedGhostId(item.waveData.best_ghost_id);
                    }}
                    className={`flex-1 flex flex-col items-center gap-1 ${
                      isCurr ? 'opacity-100' : 'opacity-65 hover:opacity-100'
                    }`}
                  >
                    <span className="text-[11px] font-mono font-bold text-white">
                      {score}%
                    </span>
                    <div
                      className={`w-full rounded-t transition-all ${
                        score >= 82
                          ? 'bg-emerald-400'
                          : score >= 60
                          ? 'bg-amber-400'
                          : 'bg-rose-500'
                      } ${isCurr ? 'ring-2 ring-white' : ''}`}
                      style={{ height: `${Math.max(12, Math.round(score * 0.48))}px` }}
                    />
                    <span className="text-[10px] font-mono text-slate-400">
                      Wave #{item.waveData.wave_index}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 3. Anatomical Injury & G-Force Stress Heatmap */}
          {selectedGhost && (
            <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  🦴 Biomechanical Injury Telemetry ({selectedGhost.ghost_id})
                </h2>
                <span className="text-[11px] font-mono text-slate-400">
                  HIC15: {selectedGhost.metrics.hic15}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {[
                  { key: 'head', label: 'Cranium (Head)' },
                  { key: 'neck', label: 'C3 Cervical Spine' },
                  { key: 'spine', label: 'Lumbar Spine' },
                  { key: 'knees', label: 'Tibial / Knee Load' },
                ].map((part) => {
                  const status = selectedGhost.joint_stress?.[part.key] || 'safe';
                  return (
                    <div
                      key={part.key}
                      className={`px-3 py-2 rounded-lg border text-xs flex items-center justify-between ${stressBadgeColor(
                        status
                      )}`}
                    >
                      <span className="font-medium">{part.label}</span>
                      <span className="font-mono uppercase font-bold text-[10px]">
                        {status}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="mt-2.5 p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300">
                {selectedGhost.diagnostic}
              </div>
            </div>
          )}

          {/* 4. Assistant Stunt Director Critique & Gate 3 Live Steer */}
          <div className="bg-[#0b101e] border border-slate-800/90 rounded-xl p-4 flex-1 flex flex-col justify-between gap-3">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                  🤖 Assistant Stunt Director Critique
                </h2>
                <span className="text-[10px] font-mono text-slate-400">
                  {currentCritique?.used_model || 'gemini-4-argon'}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/75 p-2.5 rounded-lg border border-slate-800">
                {currentCritique?.assistant_critique ||
                  'Analyzing 500Hz MuJoCo contact forces and camera frustum framing...'}
              </p>

              {currentCritique?.key_mutations?.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {currentCritique.key_mutations.map((m, i) => (
                    <li
                      key={i}
                      className="text-[11px] font-mono text-cyan-300 bg-cyan-950/30 border border-cyan-500/20 rounded px-2 py-1"
                    >
                      ↳ {m}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="border-t border-slate-800 pt-3">
              <label className="block text-xs font-bold uppercase tracking-wider text-white mb-1.5">
                🎬 Gate 3: Lead Stunt Director Live Steer
              </label>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {[
                  'Make clearance tighter (skim edge)',
                  'Higher dramatic flip arc',
                  'Shift crash mat back',
                  'Softer landing absorption',
                ].map((quick) => (
                  <button
                    key={quick}
                    onClick={() => {
                      setDirectorSteer(quick);
                      runNextRecursiveWave(quick);
                    }}
                    disabled={isSimulating}
                    className="text-[11px] px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 transition"
                  >
                    + {quick}
                  </button>
                ))}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={directorSteer}
                  onChange={(e) => setDirectorSteer(e.target.value)}
                  placeholder="Steer agent (e.g. 'Skim the couch & tuck earlier')..."
                  className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400"
                />
                <button
                  onClick={() => runNextRecursiveWave()}
                  disabled={isSimulating}
                  className="bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs px-3 py-1.5 rounded-lg transition"
                >
                  Steer ↵
                </button>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
