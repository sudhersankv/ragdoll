import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';

/**
 * Interactive Three.js WebGL Viewport for FALLGUY
 * - Renders 3D MuJoCo scene obstacles (wireframe + translucent tactical fill)
 * - Renders 4 Parallel Colored Ghost Stuntmen (articulated 14-geom rigs)
 * - Renders 3D Parabolic Trajectory Ribbons for each ghost
 * - Dynamic Bone G-Force Stress Highlighting (Red >= 15g, Amber >= 9g)
 * - 360° Bullet-Time Orbit Camera + Location Video/Image Background Plate
 */
const WebGLViewport = forwardRef(function WebGLViewport(
  {
    waveData,
    frameIndex,
    selectedGhostId,
    showAllGhosts,
    cameraMode,
    backgroundMediaUrl,
    backgroundMediaType,
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
    theta: 0.0,
    phi: 1.28,
    radius: 4.8,
    target: new THREE.Vector3(0.5, 0.75, 0.0),
  });

  useImperativeHandle(ref, () => ({
    captureSnapshot: () => {
      if (!rendererRef.current) return null;
      return rendererRef.current.domElement.toDataURL('image/jpeg', 0.75);
    },
    resetCamera: () => {
      orbitStateRef.current.theta = 0.0;
      orbitStateRef.current.phi = 1.28;
      orbitStateRef.current.radius = 4.8;
      updateCameraFromOrbit();
    },
  }));

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

  // Convert MuJoCo Z-up [x, y, z] -> Three.js Y-up [x, z, -y]
  const mjPosToThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);

  // Convert MuJoCo Quaternion [x, y, z, w] in Z-up -> Three.js Y-up
  const mjQuatToThree = (q) => {
    const basis = new THREE.Quaternion(q[0], q[2], -q[1], q[3]);
    const zToY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    return basis.multiply(zToY);
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      48,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.1,
      100
    );
    cameraRef.current = camera;
    updateCameraFromOrbit();

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const ambient = new THREE.AmbientLight(0xdbeafe, 0.85);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(3, 7, 5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.7);
    rimLight.position.set(-4, 3, -4);
    scene.add(rimLight);

    const grid = new THREE.GridHelper(12, 24, 0x0ea5e9, 0x1e293b);
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
        0.25,
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
        2.2,
        Math.min(9.5, orbitStateRef.current.radius + e.deltaY * 0.003)
      );
      updateCameraFromOrbit();
    };

    dom.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    const onResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      cameraRef.current.aspect = container.clientWidth / Math.max(1, container.clientHeight);
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', onResize);

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
      window.removeEventListener('resize', onResize);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (cameraMode === 'match') {
      orbitStateRef.current.theta = 0.0;
      orbitStateRef.current.phi = 1.32;
      orbitStateRef.current.radius = 4.6;
      updateCameraFromOrbit();
    } else if (cameraMode === 'top') {
      orbitStateRef.current.theta = 0.0;
      orbitStateRef.current.phi = 0.32;
      orbitStateRef.current.radius = 5.6;
      updateCameraFromOrbit();
    } else if (cameraMode === 'hero') {
      orbitStateRef.current.theta = 0.68;
      orbitStateRef.current.phi = 1.22;
      orbitStateRef.current.radius = 3.9;
      updateCameraFromOrbit();
    }
  }, [cameraMode]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !waveData) return;

    const oldDynamic = scene.getObjectByName('dynamic_wave_root');
    if (oldDynamic) scene.remove(oldDynamic);

    const rootGroup = new THREE.Group();
    rootGroup.name = 'dynamic_wave_root';
    ghostGroupsRef.current = {};

    (waveData.scene_obstacles || []).forEach((obs) => {
      const [sx, sy, sz] = obs.size;
      const geom = new THREE.BoxGeometry(sx * 2, sz * 2, sy * 2);
      const rgba = obs.rgba || [0, 0.8, 1, 0.4];
      const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);

      const mat = new THREE.MeshPhysicalMaterial({
        color,
        transparent: true,
        opacity: backgroundMediaUrl ? 0.28 : rgba[3] || 0.45,
        roughness: 0.25,
        metalness: 0.1,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(mjPosToThree(obs.pos));

      const edges = new THREE.EdgesGeometry(geom);
      const lineMat = new THREE.LineBasicMaterial({
        color: obs.is_hazard ? 0x38bdf8 : 0x10b981,
        transparent: true,
        opacity: 0.92,
      });
      const wireframe = new THREE.LineSegments(edges, lineMat);
      mesh.add(wireframe);
      rootGroup.add(mesh);
    });

    (waveData.results || []).forEach((ghostRes) => {
      const ghostId = ghostRes.ghost_id;
      const baseColor = new THREE.Color(ghostRes.color || '#3b82f6');
      const isSelected = selectedGhostId === ghostId || ghostId === waveData.best_ghost_id;

      const pts = (ghostRes.trajectory_points || []).map((p) => mjPosToThree(p));
      if (pts.length > 2) {
        const curve = new THREE.CatmullRomCurve3(pts);
        const tubeGeom = new THREE.TubeGeometry(curve, 48, isSelected ? 0.022 : 0.012, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: baseColor,
          transparent: true,
          opacity: isSelected ? 0.9 : 0.38,
        });
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        rootGroup.add(tubeMesh);
      }

      const rigGroup = new THREE.Group();
      const boneMeshes = [];

      (ghostRes.humanoid_geoms_meta || []).forEach((gmeta) => {
        let geomObj;
        if (gmeta.type === 'sphere') {
          geomObj = new THREE.SphereGeometry(gmeta.size[0], 16, 16);
        } else if (gmeta.type === 'box') {
          geomObj = new THREE.BoxGeometry(
            gmeta.size[0] * 2,
            gmeta.size[2] * 2,
            gmeta.size[1] * 2
          );
        } else {
          const radius = gmeta.size[0] || 0.05;
          const halfLen = gmeta.size[1] || 0.12;
          geomObj = new THREE.CapsuleGeometry(radius, halfLen * 2, 8, 16);
        }

        const mat = new THREE.MeshStandardMaterial({
          color: baseColor,
          emissive: baseColor,
          emissiveIntensity: 0.25,
          roughness: 0.3,
          metalness: 0.35,
          transparent: true,
          opacity: 0.92,
        });
        const boneMesh = new THREE.Mesh(geomObj, mat);
        boneMesh.userData = { region: gmeta.region, baseColor: baseColor.clone() };
        rigGroup.add(boneMesh);
        boneMeshes.push(boneMesh);
      });

      rootGroup.add(rigGroup);
      ghostGroupsRef.current[ghostId] = {
        rigGroup,
        boneMeshes,
        frames: ghostRes.frames || [],
      };
    });

    scene.add(rootGroup);
  }, [waveData, backgroundMediaUrl]);

  useEffect(() => {
    if (!waveData) return;
    Object.entries(ghostGroupsRef.current).forEach(([ghostId, gdata]) => {
      const { rigGroup, boneMeshes, frames } = gdata;
      const visible = showAllGhosts || ghostId === selectedGhostId;
      rigGroup.visible = visible;
      if (!visible || !frames.length) return;

      const clampedIdx = Math.min(frameIndex, frames.length - 1);
      const frame = frames[clampedIdx];
      if (!frame) return;

      const isPrimary = ghostId === selectedGhostId;
      const gLoad = frame.g || 1.0;

      boneMeshes.forEach((mesh, idx) => {
        const tf = frame.geoms[idx];
        if (!tf) return;
        mesh.position.copy(mjPosToThree(tf.p));
        mesh.quaternion.copy(mjQuatToThree(tf.q));

        const baseCol = mesh.userData.baseColor;
        const region = mesh.userData.region;
        if (gLoad >= 15.0 && (region === 'head' || region === 'neck' || region === 'spine')) {
          mesh.material.color.setHex(0xff1e42);
          mesh.material.emissive.setHex(0xff002b);
          mesh.material.emissiveIntensity = 0.85;
        } else if (gLoad >= 9.0 && (region === 'head' || region === 'neck')) {
          mesh.material.color.setHex(0xf59e0b);
          mesh.material.emissive.setHex(0xd97706);
          mesh.material.emissiveIntensity = 0.55;
        } else {
          mesh.material.color.copy(baseCol);
          mesh.material.emissive.copy(baseCol);
          mesh.material.emissiveIntensity = isPrimary ? 0.35 : 0.15;
        }
        mesh.material.opacity = isPrimary ? 0.96 : 0.42;
      });
    });
  }, [frameIndex, waveData, selectedGhostId, showAllGhosts]);

  return (
    <div className="relative w-full h-full overflow-hidden rounded-xl border border-slate-800 bg-[#070b14]">
      {backgroundMediaUrl && (
        <div className="absolute inset-0 pointer-events-none z-0 opacity-65">
          {backgroundMediaType === 'video' ? (
            <video
              src={backgroundMediaUrl}
              autoPlay
              loop
              muted
              playsInline
              className="w-full h-full object-cover"
            />
          ) : (
            <img
              src={backgroundMediaUrl}
              alt="Scouted Location Background"
              className="w-full h-full object-cover"
            />
          )}
        </div>
      )}
      <div ref={mountRef} className="relative z-10 w-full h-full cursor-grab active:cursor-grabbing" />
    </div>
  );
});

export default WebGLViewport;
