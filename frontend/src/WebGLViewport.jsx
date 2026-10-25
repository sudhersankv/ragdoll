import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';

/**
 * Dispose all geometries and materials inside a Three.js Object3D subtree
 * to prevent GPU memory leaks across optimization waves and unmount.
 */
function disposeObject3D(root) {
  if (!root) return;
  root.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m && m.dispose && m.dispose());
      } else if (child.material.dispose) {
        child.material.dispose();
      }
    }
  });
}

/**
 * Default 14-bone MuJoCo humanoid metadata + standing pose offsets relative to pelvis
 * so when the user is in Phase 1 (3D Environment Architect) before running a stunt wave,
 * the articulated stunt mannequin stands on the launch mark in the MuJoCo world.
 */
const DEFAULT_HUMANOID_META = [
  { name: 'geom_head', region: 'head', type: 'sphere', size: [0.105, 0, 0], relPos: [0, 0, 0.67], quat: [0, 0, 0, 1] },
  { name: 'geom_neck', region: 'neck', type: 'capsule', size: [0.05, 0.04, 0], relPos: [0, 0, 0.54], quat: [0, 0, 0, 1] },
  { name: 'geom_torso', region: 'spine', type: 'capsule', size: [0.105, 0.145, 0], relPos: [0, 0, 0.315], quat: [0, 0, 0, 1] },
  { name: 'geom_pelvis', region: 'spine', type: 'capsule', size: [0.09, 0.10, 0], relPos: [0, 0, 0.0], quat: [0.7071, 0, 0, 0.7071] },
  { name: 'geom_upper_arm_l', region: 'arms', type: 'capsule', size: [0.045, 0.12, 0], relPos: [0, 0.19, 0.30], quat: [0, 0, 0, 1] },
  { name: 'geom_lower_arm_l', region: 'arms', type: 'capsule', size: [0.038, 0.11, 0], relPos: [0.03, 0.19, 0.05], quat: [0, 0, 0, 1] },
  { name: 'geom_upper_arm_r', region: 'arms', type: 'capsule', size: [0.045, 0.12, 0], relPos: [0, -0.19, 0.30], quat: [0, 0, 0, 1] },
  { name: 'geom_lower_arm_r', region: 'arms', type: 'capsule', size: [0.038, 0.11, 0], relPos: [0.03, -0.19, 0.05], quat: [0, 0, 0, 1] },
  { name: 'geom_thigh_l', region: 'knees', type: 'capsule', size: [0.065, 0.18, 0], relPos: [0, 0.11, -0.23], quat: [0, 0, 0, 1] },
  { name: 'geom_shin_l', region: 'knees', type: 'capsule', size: [0.05, 0.17, 0], relPos: [0, 0.11, -0.60], quat: [0, 0, 0, 1] },
  { name: 'geom_foot_l', region: 'knees', type: 'box', size: [0.09, 0.045, 0.03], relPos: [0.05, 0.11, -0.81], quat: [0, 0, 0, 1] },
  { name: 'geom_thigh_r', region: 'knees', type: 'capsule', size: [0.065, 0.18, 0], relPos: [0, -0.11, -0.23], quat: [0, 0, 0, 1] },
  { name: 'geom_shin_r', region: 'knees', type: 'capsule', size: [0.05, 0.17, 0], relPos: [0, -0.11, -0.60], quat: [0, 0, 0, 1] },
  { name: 'geom_foot_r', region: 'knees', type: 'box', size: [0.09, 0.045, 0.03], relPos: [0.05, -0.11, -0.81], quat: [0, 0, 0, 1] },
];

/**
 * Anatomical bone adjacency links between the 14 MuJoCo humanoid geoms:
 *   0: geom_head, 1: geom_neck, 2: geom_torso, 3: geom_pelvis,
 *   4: geom_upper_arm_l, 5: geom_lower_arm_l,
 *   6: geom_upper_arm_r, 7: geom_lower_arm_r,
 *   8: geom_thigh_l, 9: geom_shin_l, 10: geom_foot_l,
 *  11: geom_thigh_r, 12: geom_shin_r, 13: geom_foot_r
 */
const ANATOMICAL_JOINTS = [
  [0, new THREE.Vector3(0, -0.08, 0), 1, new THREE.Vector3(0, 0.04, 0), 0.046, 'neck'],
  [1, new THREE.Vector3(0, -0.04, 0), 2, new THREE.Vector3(0, 0.145, 0), 0.058, 'neck'],
  [2, new THREE.Vector3(0, -0.145, 0), 3, new THREE.Vector3(0, 0.0, 0), 0.088, 'spine'],
  [2, new THREE.Vector3(0, 0.11, -0.175), 4, new THREE.Vector3(0, -0.12, 0), 0.052, 'arms'],
  [2, new THREE.Vector3(0, 0.11, 0.175), 6, new THREE.Vector3(0, -0.12, 0), 0.052, 'arms'],
  [4, new THREE.Vector3(0, 0.12, 0), 5, new THREE.Vector3(0, -0.11, 0), 0.044, 'arms'],
  [6, new THREE.Vector3(0, 0.12, 0), 7, new THREE.Vector3(0, -0.11, 0), 0.044, 'arms'],
  [3, new THREE.Vector3(0, -0.08, 0), 8, new THREE.Vector3(0, -0.18, 0), 0.068, 'spine'],
  [3, new THREE.Vector3(0, 0.08, 0), 11, new THREE.Vector3(0, -0.18, 0), 0.068, 'spine'],
  [8, new THREE.Vector3(0, 0.18, 0), 9, new THREE.Vector3(0, -0.17, 0), 0.056, 'knees'],
  [11, new THREE.Vector3(0, 0.18, 0), 12, new THREE.Vector3(0, -0.17, 0), 0.056, 'knees'],
  [9, new THREE.Vector3(0, 0.17, 0), 10, new THREE.Vector3(-0.03, 0.02, 0), 0.046, 'knees'],
  [12, new THREE.Vector3(0, 0.17, 0), 13, new THREE.Vector3(-0.03, 0.02, 0), 0.046, 'knees'],
];

/**
 * Build a detailed, articulated Stunt-Double / Crash-Test Mannequin segment
 * for one of the 14 MuJoCo humanoid geoms.
 */
function createArticulatedBoneGroup(gmeta, primaryMat, trimMat, visorMat) {
  const group = new THREE.Group();
  const name = gmeta.name || '';
  const s0 = gmeta.size?.[0] || 0.05;
  const s1 = gmeta.size?.[1] || 0.12;
  const s2 = gmeta.size?.[2] || 0.04;

  if (name === 'geom_head' || gmeta.type === 'sphere') {
    const r = s0;
    const cranium = new THREE.Mesh(new THREE.SphereGeometry(r * 1.04, 24, 24), primaryMat);
    cranium.scale.set(1.08, 1.04, 0.96);
    group.add(cranium);

    const visor = new THREE.Mesh(
      new THREE.CylinderGeometry(r * 0.96, r * 0.96, r * 0.62, 20, 1, false, -Math.PI * 0.38, Math.PI * 0.76),
      visorMat
    );
    visor.rotation.y = Math.PI * 0.5;
    visor.position.set(r * 0.16, r * 0.08, 0);
    group.add(visor);

    const chinGuard = new THREE.Mesh(
      new THREE.BoxGeometry(r * 0.95, r * 0.42, r * 1.25),
      trimMat
    );
    chinGuard.position.set(r * 0.28, -r * 0.46, 0);
    group.add(chinGuard);

    const crest = new THREE.Mesh(
      new THREE.BoxGeometry(r * 1.35, r * 0.22, r * 0.24),
      trimMat
    );
    crest.position.set(-r * 0.05, r * 0.82, 0);
    group.add(crest);
  } else if (name === 'geom_neck') {
    const radius = s0;
    const halfLen = s1;
    const core = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 0.92, halfLen * 2, 8, 16),
      primaryMat
    );
    group.add(core);

    [-0.5, 0, 0.5].forEach((frac) => {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 1.05, radius * 0.18, 8, 18),
        trimMat
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.y = frac * halfLen * 1.4;
      group.add(ring);
    });
  } else if (name === 'geom_torso') {
    const radius = s0;
    const halfLen = s1;
    const torsoCore = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius, halfLen * 1.9, 12, 20),
      primaryMat
    );
    torsoCore.scale.set(0.92, 1.0, 1.38);
    group.add(torsoCore);

    const chestPlate = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 1.55, halfLen * 1.28, radius * 2.55),
      primaryMat
    );
    chestPlate.position.set(radius * 0.26, halfLen * 0.22, 0);
    group.add(chestPlate);

    const sternumPod = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.42, radius * 0.42, radius * 0.25, 16),
      visorMat
    );
    sternumPod.rotation.z = Math.PI / 2;
    sternumPod.position.set(radius * 0.98, halfLen * 0.3, 0);
    group.add(sternumPod);

    [-1, 1].forEach((side) => {
      const deltoid = new THREE.Mesh(
        new THREE.SphereGeometry(radius * 0.64, 16, 16),
        primaryMat
      );
      deltoid.scale.set(1.0, 0.95, 1.15);
      deltoid.position.set(0, halfLen * 0.72, side * radius * 1.58);
      group.add(deltoid);
    });

    const spineRidge = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 0.45, halfLen * 1.95, radius * 0.48),
      trimMat
    );
    spineRidge.position.set(-radius * 0.85, 0, 0);
    group.add(spineRidge);

    const belt = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 1.65, halfLen * 0.35, radius * 2.3),
      trimMat
    );
    belt.position.set(0, -halfLen * 0.58, 0);
    group.add(belt);
  } else if (name === 'geom_pelvis') {
    const radius = s0;
    const halfLen = s1;
    const pelvisCore = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 0.98, halfLen * 2.0, 10, 18),
      primaryMat
    );
    group.add(pelvisCore);

    const hipBelt = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 1.95, halfLen * 1.65, radius * 1.85),
      trimMat
    );
    group.add(hipBelt);
  } else if (name.startsWith('geom_upper_arm')) {
    const radius = s0;
    const halfLen = s1;
    const bicep = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 1.04, halfLen * 1.92, 10, 16),
      primaryMat
    );
    group.add(bicep);

    const band = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 1.14, radius * 1.12, halfLen * 0.45, 16),
      trimMat
    );
    band.position.y = -halfLen * 0.15;
    group.add(band);
  } else if (name.startsWith('geom_lower_arm')) {
    const radius = s0;
    const halfLen = s1;
    const forearm = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 0.98, halfLen * 1.9, 10, 16),
      primaryMat
    );
    group.add(forearm);

    const gauntlet = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 1.12, radius * 0.95, halfLen * 0.95, 14),
      trimMat
    );
    gauntlet.position.y = halfLen * 0.15;
    group.add(gauntlet);

    const hand = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.18, 14, 14),
      visorMat
    );
    hand.scale.set(1.15, 1.35, 0.78);
    hand.position.set(0, halfLen + radius * 0.75, 0);
    group.add(hand);
  } else if (name.startsWith('geom_thigh')) {
    const radius = s0;
    const halfLen = s1;
    const quad = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 1.05, halfLen * 1.92, 10, 18),
      primaryMat
    );
    quad.scale.set(1.08, 1.0, 0.96);
    group.add(quad);

    const thighGuard = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 1.14, radius * 1.04, halfLen * 0.85, 16),
      trimMat
    );
    thighGuard.position.y = -halfLen * 0.1;
    group.add(thighGuard);
  } else if (name.startsWith('geom_shin')) {
    const radius = s0;
    const halfLen = s1;
    const calf = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius * 0.98, halfLen * 1.92, 10, 18),
      primaryMat
    );
    group.add(calf);

    const shinGuard = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 1.35, halfLen * 1.35, radius * 1.55),
      trimMat
    );
    shinGuard.position.set(radius * 0.25, 0, 0);
    group.add(shinGuard);
  } else if (gmeta.type === 'box' || name.startsWith('geom_foot')) {
    const bootUpper = new THREE.Mesh(
      new THREE.BoxGeometry(s0 * 2.0, s2 * 2.0, s1 * 2.0),
      primaryMat
    );
    group.add(bootUpper);

    const toeCap = new THREE.Mesh(
      new THREE.SphereGeometry(s1 * 1.02, 14, 14),
      trimMat
    );
    toeCap.scale.set(1.15, (s2 / Math.max(0.01, s1)) * 0.95, 1.0);
    toeCap.position.set(s0 * 0.85, -s2 * 0.1, 0);
    group.add(toeCap);

    const sole = new THREE.Mesh(
      new THREE.BoxGeometry(s0 * 2.15, s2 * 0.55, s1 * 2.1),
      visorMat
    );
    sole.position.set(s0 * 0.05, -s2 * 0.85, 0);
    group.add(sole);
  } else {
    const geomObj = new THREE.CapsuleGeometry(s0, s1 * 2, 10, 16);
    group.add(new THREE.Mesh(geomObj, primaryMat));
  }

  return group;
}

/**
 * Pure 3D MuJoCo World Viewport for RAGDOLL
 * - Renders the generated 3D MuJoCo collision obstacles (wireframe + solid tactical shading + crash mat target bullseye)
 * - In Phase 1 (Environment Architect before waves run), renders the stand-by articulated stunt mannequin on the launch pad
 * - In Phase 2/3 (Choreography & 3-Gate Simulation), renders all 4 Parallel Ghost Stuntmen + Parabolic Trajectory Ribbons
 * - Dynamic Bone G-Force Stress Highlighting (Critical Rose-Red >= 8.0G, Warning Amber >= 5.0G)
 */
const WebGLViewport = forwardRef(function WebGLViewport(
  {
    waveData,
    envScene,
    frameIndex,
    selectedGhostId,
    showAllGhosts,
    cameraMode,
  },
  ref
) {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const ghostGroupsRef = useRef({});
  const orbitStateRef = useRef({
    isDragging: false,
    lastX: 0,
    lastY: 0,
    theta: 0.15,
    phi: 1.24,
    radius: 4.7,
    target: new THREE.Vector3(0.45, 0.72, 0.0),
  });

  const activeData = waveData || envScene || null;

  // Convert MuJoCo Z-up [x, y, z] -> Three.js Y-up [x, z, -y]
  const mjPosToThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);

  /**
   * Convert MuJoCo Quaternion [qx, qy, qz, qw] (in Z-up) -> Three.js Y-up.
   */
  const mjQuatToThree = (q) => new THREE.Quaternion(q[0], q[2], -q[1], q[3]).normalize();

  const updateCameraFromOrbit = () => {
    const cam = cameraRef.current;
    if (!cam) return;
    const { theta, phi, radius, target } = orbitStateRef.current;
    const x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = target.y + radius * Math.cos(phi);
    const z = target.z + radius * Math.sin(phi) * Math.cos(theta);
    cam.position.set(x, y, z);
    cam.lookAt(target);
  };

  const syncCameraToMatchSpec = (camSpec) => {
    const cam = cameraRef.current;
    if (!cam) return;
    if (camSpec && Array.isArray(camSpec.pos) && Array.isArray(camSpec.lookat)) {
      const posThree = mjPosToThree(camSpec.pos);
      const lookThree = mjPosToThree(camSpec.lookat);
      const offset = new THREE.Vector3().subVectors(posThree, lookThree);
      const radius = Math.max(1.2, offset.length());
      const phi = Math.max(0.15, Math.min(Math.PI - 0.15, Math.acos(offset.y / radius)));
      const theta = Math.atan2(offset.x, offset.z);

      orbitStateRef.current.target.copy(lookThree);
      orbitStateRef.current.radius = radius;
      orbitStateRef.current.phi = phi;
      orbitStateRef.current.theta = theta;

      cam.fov = Number(camSpec.fovy) || 50;
      cam.updateProjectionMatrix();
      cam.position.copy(posThree);
      cam.lookAt(lookThree);
    } else {
      orbitStateRef.current.target.set(0.45, 0.72, 0.0);
      orbitStateRef.current.theta = 0.15;
      orbitStateRef.current.phi = 1.24;
      orbitStateRef.current.radius = 4.7;
      cam.fov = 48;
      cam.updateProjectionMatrix();
      updateCameraFromOrbit();
    }
  };

  useImperativeHandle(ref, () => ({
    captureSnapshot: () => {
      if (!rendererRef.current) return null;
      const glCanvas = rendererRef.current.domElement;
      if (!glCanvas) return null;

      if (sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }

      try {
        const width = glCanvas.width || 960;
        const height = glCanvas.height || 540;
        const offscreen = document.createElement('canvas');
        offscreen.width = width;
        offscreen.height = height;
        const ctx = offscreen.getContext('2d');
        if (!ctx) return glCanvas.toDataURL('image/jpeg', 0.85);

        ctx.fillStyle = '#060a12';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(glCanvas, 0, 0, width, height);
        return offscreen.toDataURL('image/jpeg', 0.85);
      } catch {
        return glCanvas.toDataURL('image/jpeg', 0.85);
      }
    },
    resetCamera: () => {
      if (cameraMode === 'match' && activeData?.camera) {
        syncCameraToMatchSpec(activeData.camera);
      } else {
        orbitStateRef.current.target.set(0.45, 0.72, 0.0);
        orbitStateRef.current.theta = 0.15;
        orbitStateRef.current.phi = 1.24;
        orbitStateRef.current.radius = 4.7;
        updateCameraFromOrbit();
      }
    },
  }));

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      48,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.05,
      400
    );
    cameraRef.current = camera;
    updateCameraFromOrbit();

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const ambient = new THREE.AmbientLight(0xdbeafe, 0.92);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.55);
    keyLight.position.set(3.5, 7.5, 5.0);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
    rimLight.position.set(-4, 3.5, -4);
    scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0x818cf8, 0.45);
    fillLight.position.set(0, -3, 4);
    scene.add(fillLight);

    // MuJoCo Physics Stage Floor & Coordinate Grid
    const floorGeom = new THREE.PlaneGeometry(14, 10);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x090f1d,
      roughness: 0.85,
      metalness: 0.15,
    });
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = -0.002;
    scene.add(floorMesh);

    const grid = new THREE.GridHelper(14, 28, 0x0ea5e9, 0x1e293b);
    grid.position.y = 0.002;
    scene.add(grid);

    const dom = renderer.domElement;
    const onPointerDown = (e) => {
      orbitStateRef.current.isDragging = true;
      orbitStateRef.current.lastX = e.clientX;
      orbitStateRef.current.lastY = e.clientY;
    };
    const onPointerMove = (e) => {
      if (!orbitStateRef.current.isDragging) return;
      const dx = e.clientX - orbitStateRef.current.lastX;
      const dy = e.clientY - orbitStateRef.current.lastY;
      orbitStateRef.current.lastX = e.clientX;
      orbitStateRef.current.lastY = e.clientY;
      orbitStateRef.current.theta -= dx * 0.007;
      orbitStateRef.current.phi = Math.max(
        0.18,
        Math.min(1.52, orbitStateRef.current.phi - dy * 0.006)
      );
      updateCameraFromOrbit();
    };
    const onPointerUp = () => {
      orbitStateRef.current.isDragging = false;
    };
    const onWheel = (e) => {
      e.preventDefault();
      orbitStateRef.current.radius = Math.max(
        1.8,
        Math.min(40, orbitStateRef.current.radius + e.deltaY * 0.004)
      );
      updateCameraFromOrbit();
    };

    dom.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    const onResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width < 2 || height < 2) return;
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height, false);
    };
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    onResize();
    const resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(container);

    let animId;
    const renderLoop = () => {
      animId = requestAnimationFrame(renderLoop);
      renderer.render(scene, camera);
    };
    renderLoop();

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      dom.removeEventListener('wheel', onWheel);
      resizeObserver.disconnect();
      disposeObject3D(scene);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (cameraMode === 'match') {
      syncCameraToMatchSpec(activeData?.camera);
    } else if (cameraMode === 'top') {
      orbitStateRef.current.theta = 0.0;
      orbitStateRef.current.phi = 0.25;
      orbitStateRef.current.radius = 5.8;
      updateCameraFromOrbit();
    } else if (cameraMode === 'hero') {
      orbitStateRef.current.theta = 0.65;
      orbitStateRef.current.phi = 1.2;
      orbitStateRef.current.radius = 4.1;
      updateCameraFromOrbit();
    }
  }, [cameraMode, activeData?.camera]);

  // Build 3D MuJoCo obstacles + articulated humanoid stuntman rigs
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    const oldDynamic = scene.getObjectByName('dynamic_wave_root');
    if (oldDynamic) {
      scene.remove(oldDynamic);
      disposeObject3D(oldDynamic);
    }

    if (!activeData) return;

    const rootGroup = new THREE.Group();
    rootGroup.name = 'dynamic_wave_root';
    ghostGroupsRef.current = {};

    // 1. Render 3D MuJoCo Obstacles & Crash Mat
    (activeData.scene_obstacles || []).forEach((obs) => {
      const [sx, sy, sz] = obs.size;
      const geom = new THREE.BoxGeometry(sx * 2, sz * 2, sy * 2);
      const isMat = obs.id === 'crash_mat';
      const isHazard = Boolean(obs.is_hazard);
      const rgba = obs.rgba || (isMat ? [0.06, 0.72, 0.5, 0.55] : isHazard ? [0.95, 0.35, 0.2, 0.55] : [0.25, 0.55, 0.95, 0.5]);
      const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);

      const mat = new THREE.MeshPhysicalMaterial({
        color,
        transparent: true,
        opacity: isMat ? 0.52 : 0.48,
        roughness: 0.28,
        metalness: 0.15,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(mjPosToThree(obs.pos));

      const labelCanvas = document.createElement('canvas');
      labelCanvas.width = 512;
      labelCanvas.height = 96;
      const labelCtx = labelCanvas.getContext('2d');
      if (labelCtx) {
        labelCtx.fillStyle = 'rgba(2, 6, 23, 0.88)';
        labelCtx.fillRect(0, 16, 512, 64);
        labelCtx.fillStyle = '#f8fafc';
        labelCtx.font = '32px ui-monospace, monospace';
        labelCtx.textAlign = 'center';
        labelCtx.textBaseline = 'middle';
        labelCtx.fillText(String(obs.id || 'object').replaceAll('_', ' ').slice(0, 22), 256, 48);
        const label = new THREE.Sprite(
          new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(labelCanvas), transparent: true, depthTest: false })
        );
        label.position.y = sz + 0.16;
        label.scale.set(1.15, 0.22, 1);
        mesh.add(label);
      }

      const edges = new THREE.EdgesGeometry(geom);
      const lineMat = new THREE.LineBasicMaterial({
        color: isMat ? 0x10b981 : isHazard ? 0xf59e0b : 0x38bdf8,
        transparent: true,
        opacity: 0.95,
      });
      const wireframe = new THREE.LineSegments(edges, lineMat);
      mesh.add(wireframe);

      if (isMat) {
        const ringRadius = Math.min(sx, sy) * 0.72;
        const outerRing = new THREE.Mesh(
          new THREE.RingGeometry(ringRadius * 0.78, ringRadius * 0.95, 32),
          new THREE.MeshBasicMaterial({
            color: 0x10b981,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.82,
          })
        );
        outerRing.rotation.x = -Math.PI / 2;
        outerRing.position.y = sz + 0.004;
        mesh.add(outerRing);

        const innerRing = new THREE.Mesh(
          new THREE.RingGeometry(ringRadius * 0.28, ringRadius * 0.42, 24),
          new THREE.MeshBasicMaterial({
            color: 0x34d399,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.9,
          })
        );
        innerRing.rotation.x = -Math.PI / 2;
        innerRing.position.y = sz + 0.005;
        mesh.add(innerRing);
      }

      rootGroup.add(mesh);
    });

    // Location replica has boxes only. The humanoid appears after a stunt wave exists.
    const ghostList = activeData.results || [];

    ghostList.forEach((ghostRes) => {
      const ghostId = ghostRes.ghost_id;
      const baseColor = new THREE.Color(ghostRes.color || '#3b82f6');
      const isSelected = selectedGhostId === ghostId || ghostId === activeData.best_ghost_id;

      // Parabolic trajectory ribbon
      const pts = (ghostRes.trajectory_points || []).map((p) => mjPosToThree(p));
      if (pts.length > 2) {
        const curve = new THREE.CatmullRomCurve3(pts);
        const tubeGeom = new THREE.TubeGeometry(curve, 56, isSelected ? 0.02 : 0.011, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: baseColor,
          transparent: true,
          opacity: isSelected ? 0.88 : 0.35,
        });
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        rootGroup.add(tubeMesh);
      }

      const rigGroup = new THREE.Group();
      const boneGroups = [];

      const trimColor = baseColor.clone().lerp(new THREE.Color(0x0f172a), 0.55);
      const visorMat = new THREE.MeshStandardMaterial({
        color: 0x090d16,
        emissive: baseColor.clone().multiplyScalar(0.15),
        roughness: 0.12,
        metalness: 0.85,
        transparent: true,
        opacity: 0.95,
      });

      (ghostRes.humanoid_geoms_meta || DEFAULT_HUMANOID_META).forEach((gmeta) => {
        const primaryMat = new THREE.MeshStandardMaterial({
          color: baseColor,
          emissive: baseColor,
          emissiveIntensity: 0.28,
          roughness: 0.28,
          metalness: 0.38,
          transparent: true,
          opacity: 0.94,
        });
        const trimMat = new THREE.MeshStandardMaterial({
          color: trimColor,
          emissive: baseColor,
          emissiveIntensity: 0.14,
          roughness: 0.4,
          metalness: 0.55,
          transparent: true,
          opacity: 0.94,
        });

        const boneGroup = createArticulatedBoneGroup(gmeta, primaryMat, trimMat, visorMat);
        boneGroup.userData = {
          region: gmeta.region,
          name: gmeta.name,
          baseColor: baseColor.clone(),
          trimColor: trimColor.clone(),
          primaryMat,
          trimMat,
          visorMat,
        };
        rigGroup.add(boneGroup);
        boneGroups.push(boneGroup);
      });

      const jointConnectors = ANATOMICAL_JOINTS.map(
        ([idxA, anchorA, idxB, anchorB, jRadius, region]) => {
          const jMat = new THREE.MeshStandardMaterial({
            color: trimColor,
            emissive: baseColor,
            emissiveIntensity: 0.22,
            roughness: 0.32,
            metalness: 0.55,
            transparent: true,
            opacity: 0.94,
          });
          const ballMesh = new THREE.Mesh(
            new THREE.SphereGeometry(jRadius, 12, 12),
            jMat
          );
          const bridgeMesh = new THREE.Mesh(
            new THREE.CylinderGeometry(jRadius * 0.72, jRadius * 0.72, 1.0, 10),
            jMat
          );
          rigGroup.add(ballMesh);
          rigGroup.add(bridgeMesh);
          return {
            idxA,
            anchorA,
            idxB,
            anchorB,
            region,
            jMat,
            ballMesh,
            bridgeMesh,
            baseColor: baseColor.clone(),
            trimColor: trimColor.clone(),
          };
        }
      );

      const shadowDisc = new THREE.Mesh(
        new THREE.CircleGeometry(0.28, 24),
        new THREE.MeshBasicMaterial({
          color: baseColor,
          transparent: true,
          opacity: 0.22,
          depthWrite: false,
        })
      );
      shadowDisc.rotation.x = -Math.PI / 2;
      shadowDisc.position.y = 0.01;
      rigGroup.add(shadowDisc);

      rootGroup.add(rigGroup);
      ghostGroupsRef.current[ghostId] = {
        rigGroup,
        boneGroups,
        jointConnectors,
        shadowDisc,
        frames: ghostRes.frames || [],
      };
    });

    if (!(activeData.results || []).length && rootGroup.children.length) {
      const bounds = new THREE.Box3().setFromObject(rootGroup);
      if (!bounds.isEmpty()) {
        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        const fit = Math.max(size.x, size.y, size.z, 2);
        orbitStateRef.current.target.copy(center);
        orbitStateRef.current.radius = Math.min(36, Math.max(3.2, fit * 1.7));
        orbitStateRef.current.phi = 1.02;
        orbitStateRef.current.theta = 0.62;
        const cam = cameraRef.current;
        if (cam) {
          cam.fov = 46;
          cam.far = Math.max(400, fit * 14);
          cam.updateProjectionMatrix();
        }
        updateCameraFromOrbit();
      }
    }

    scene.add(rootGroup);
  }, [activeData]);

  // Animate articulated 14-bone rigs, anatomical joints, and 8.0G biomechanical stress coloring
  useEffect(() => {
    if (!activeData) return;
    const upAxis = new THREE.Vector3(0, 1, 0);
    const tmpWorldA = new THREE.Vector3();
    const tmpWorldB = new THREE.Vector3();
    const tmpMid = new THREE.Vector3();
    const tmpDir = new THREE.Vector3();

    Object.entries(ghostGroupsRef.current).forEach(([ghostId, gdata]) => {
      const { rigGroup, boneGroups, jointConnectors, shadowDisc, frames } = gdata;
      const visible = showAllGhosts || ghostId === selectedGhostId || Object.keys(ghostGroupsRef.current).length === 1;
      rigGroup.visible = visible;
      if (!visible || !frames.length) return;

      const clampedIdx = Math.min(frameIndex, frames.length - 1);
      const frame = frames[clampedIdx];
      if (!frame) return;

      const isPrimary = ghostId === selectedGhostId || Object.keys(ghostGroupsRef.current).length === 1;
      const gLoad = frame.g || 1.0;
      const opacity = isPrimary ? 0.96 : 0.4;

      boneGroups.forEach((bGroup, idx) => {
        const tf = frame.geoms[idx];
        if (!tf) return;
        bGroup.position.copy(mjPosToThree(tf.p));
        bGroup.quaternion.copy(mjQuatToThree(tf.q));

        const { region, baseColor, trimColor, primaryMat, trimMat, visorMat } = bGroup.userData;
        if (gLoad >= 8.0 && (region === 'head' || region === 'neck' || region === 'spine')) {
          primaryMat.color.setHex(0xf43f5e);
          primaryMat.emissive.setHex(0xe11d48);
          primaryMat.emissiveIntensity = 0.85;
          trimMat.color.setHex(0xbe123c);
          trimMat.emissive.setHex(0xf43f5e);
          trimMat.emissiveIntensity = 0.65;
        } else if (gLoad >= 5.0 && (region === 'head' || region === 'neck' || region === 'spine')) {
          primaryMat.color.setHex(0xf59e0b);
          primaryMat.emissive.setHex(0xd97706);
          primaryMat.emissiveIntensity = 0.58;
          trimMat.color.setHex(0xb45309);
          trimMat.emissive.setHex(0xf59e0b);
          trimMat.emissiveIntensity = 0.42;
        } else {
          primaryMat.color.copy(baseColor);
          primaryMat.emissive.copy(baseColor);
          primaryMat.emissiveIntensity = isPrimary ? 0.34 : 0.14;
          trimMat.color.copy(trimColor);
          trimMat.emissive.copy(baseColor);
          trimMat.emissiveIntensity = isPrimary ? 0.18 : 0.08;
        }

        primaryMat.opacity = opacity;
        trimMat.opacity = opacity;
        if (visorMat) visorMat.opacity = isPrimary ? 0.96 : 0.45;
      });

      jointConnectors.forEach((jc) => {
        const boneA = boneGroups[jc.idxA];
        const boneB = boneGroups[jc.idxB];
        if (!boneA || !boneB) return;

        tmpWorldA.copy(jc.anchorA).applyQuaternion(boneA.quaternion).add(boneA.position);
        tmpWorldB.copy(jc.anchorB).applyQuaternion(boneB.quaternion).add(boneB.position);
        tmpMid.addVectors(tmpWorldA, tmpWorldB).multiplyScalar(0.5);

        jc.ballMesh.position.copy(tmpMid);

        tmpDir.subVectors(tmpWorldB, tmpWorldA);
        const dist = tmpDir.length();
        if (dist > 0.004) {
          jc.bridgeMesh.visible = true;
          jc.bridgeMesh.position.copy(tmpMid);
          jc.bridgeMesh.scale.set(1, dist, 1);
          jc.bridgeMesh.quaternion.setFromUnitVectors(upAxis, tmpDir.normalize());
        } else {
          jc.bridgeMesh.visible = false;
        }

        if (gLoad >= 8.0 && (jc.region === 'head' || jc.region === 'neck' || jc.region === 'spine')) {
          jc.jMat.color.setHex(0xf43f5e);
          jc.jMat.emissive.setHex(0xe11d48);
          jc.jMat.emissiveIntensity = 0.75;
        } else if (
          gLoad >= 5.0 &&
          (jc.region === 'head' || jc.region === 'neck' || jc.region === 'spine')
        ) {
          jc.jMat.color.setHex(0xf59e0b);
          jc.jMat.emissive.setHex(0xd97706);
          jc.jMat.emissiveIntensity = 0.5;
        } else {
          jc.jMat.color.copy(jc.trimColor);
          jc.jMat.emissive.copy(jc.baseColor);
          jc.jMat.emissiveIntensity = isPrimary ? 0.22 : 0.1;
        }
        jc.jMat.opacity = opacity;
      });

      const pelvisBone = boneGroups[3];
      if (shadowDisc && pelvisBone) {
        shadowDisc.position.x = pelvisBone.position.x;
        shadowDisc.position.z = pelvisBone.position.z;
        const heightAboveGround = Math.max(0.1, pelvisBone.position.y);
        const scale = Math.max(0.45, Math.min(1.25, 1.35 / (0.6 + heightAboveGround * 0.45)));
        shadowDisc.scale.set(scale, scale, 1);
        shadowDisc.material.opacity = isPrimary ? 0.26 : 0.08;
      }
    });
  }, [frameIndex, activeData, selectedGhostId, showAllGhosts]);

  return (
    <div className="relative w-full h-full overflow-hidden rounded-xl border border-slate-800 bg-[#060a12]">
      <div ref={mountRef} className="relative z-10 w-full h-full cursor-grab active:cursor-grabbing" />
    </div>
  );
});

export default WebGLViewport;
