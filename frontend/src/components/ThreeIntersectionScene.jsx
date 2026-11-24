import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LANES, LightColor, SignalState } from '../simulation/stateMachine';
import IntersectionDiagram from './IntersectionDiagram';

const CAMERA_PRESETS = {
  ISO: { pos: [34, 30, 34], target: [0, 0, 0] },
  PLAN: { pos: [0, 56, 0.01], target: [0, 0, 0] },
  'N-S': { pos: [7, 15, 42], target: [0, 1, 0] },
  'E-W': { pos: [42, 15, 7], target: [0, 1, 0] },
};

const CAR_PALETTE = [0x94a3b8, 0x475569, 0x3b82f6, 0xd97706, 0xcbd5e1];

const MAST_SPECS = {
  N: { post: [10.4, -10.4], armDir: [-1, 0], head: [5.4, -10.4], rotY: 0, hudAnchor: [6.2, 8.2, -11.2] },
  E: { post: [10.4, 10.4], armDir: [0, -1], head: [10.4, 5.4], rotY: -Math.PI / 2, hudAnchor: [11.2, 8.2, 6.2] },
  S: { post: [-10.4, 10.4], armDir: [1, 0], head: [-5.4, 10.4], rotY: Math.PI, hudAnchor: [-6.2, 8.2, 11.2] },
  W: { post: [-10.4, -10.4], armDir: [0, 1], head: [-10.4, -5.4], rotY: Math.PI / 2, hudAnchor: [-11.2, 8.2, -6.2] },
};

const STOP_STRIP_POS = {
  N: { x: 4.5, z: -10.5, rotY: 0 },
  E: { x: 10.5, z: 4.5, rotY: Math.PI / 2 },
  S: { x: -4.5, z: 10.5, rotY: 0 },
  W: { x: -10.5, z: -4.5, rotY: Math.PI / 2 },
};

const LANE_TRAVEL = {
  N: { yaw: 0, dx: 0, dz: 1, subX: [2.6, 6.4], startZ: -13.2, stepZ: -5.4, exitZ: 28 },
  S: { yaw: Math.PI, dx: 0, dz: -1, subX: [-2.6, -6.4], startZ: 13.2, stepZ: 5.4, exitZ: -28 },
  E: { yaw: -Math.PI / 2, dx: -1, dz: 0, subZ: [2.6, 6.4], startX: 13.2, stepX: 5.4, exitX: -28 },
  W: { yaw: Math.PI / 2, dx: 1, dz: 0, subZ: [-2.6, -6.4], startX: -13.2, stepX: -5.4, exitX: 28 },
};

function getQueueTransform(lane, slotIdx) {
  const col = slotIdx % 2;
  const row = Math.floor(slotIdx / 2);
  const tr = LANE_TRAVEL[lane];
  if (lane === 'N' || lane === 'S') {
    return { x: tr.subX[col], z: tr.startZ + row * tr.stepZ, yaw: tr.yaw };
  }
  return { x: tr.startX + row * tr.stepX, z: tr.subZ[col], yaw: tr.yaw };
}

function getCrossingOrigin(lane, subIdx) {
  const tr = LANE_TRAVEL[lane];
  if (lane === 'N') return { x: tr.subX[subIdx % 2], z: -10.6 - subIdx * 2.4, yaw: tr.yaw };
  if (lane === 'S') return { x: tr.subX[subIdx % 2], z: 10.6 + subIdx * 2.4, yaw: tr.yaw };
  if (lane === 'E') return { x: 10.6 + subIdx * 2.4, z: tr.subZ[subIdx % 2], yaw: tr.yaw };
  return { x: -10.6 - subIdx * 2.4, z: tr.subZ[subIdx % 2], yaw: tr.yaw };
}

function buildStandardCarMesh(colorHex) {
  const group = new THREE.Group();

  const bodyMat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: 0.38,
    metalness: 0.45,
  });
  const cabinMat = new THREE.MeshStandardMaterial({
    color: 0x111827,
    roughness: 0.25,
    metalness: 0.65,
  });
  const headMat = new THREE.MeshBasicMaterial({ color: 0xfef08a });
  const tailMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x090b0e, roughness: 0.85 });

  const chassis = new THREE.Mesh(new THREE.BoxGeometry(1.95, 0.72, 3.85), bodyMat);
  chassis.position.y = 0.56;
  chassis.castShadow = true;
  chassis.receiveShadow = true;
  group.add(chassis);

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.62, 0.56, 2.0), cabinMat);
  cabin.position.set(0, 1.15, -0.18);
  cabin.castShadow = true;
  group.add(cabin);

  // Headlights (+Z front in local space)
  for (const sx of [-0.62, 0.62]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.16, 0.08), headMat);
    hl.position.set(sx, 0.6, 1.94);
    group.add(hl);
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.16, 0.08), tailMat);
    tl.position.set(sx, 0.6, -1.94);
    group.add(tl);
  }

  for (const sx of [-0.96, 0.96]) {
    for (const sz of [-1.15, 1.15]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.22, 12), wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(sx, 0.34, sz);
      group.add(wheel);
    }
  }

  return group;
}

function buildAmbulanceMesh() {
  const group = new THREE.Group();

  const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.32, metalness: 0.2 });
  const stripeMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.4 });
  const glassMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.2, metalness: 0.7 });
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x090b0e, roughness: 0.85 });

  const body = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.48, 4.55), whiteMat);
  body.position.y = 0.98;
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);

  const stripe = new THREE.Mesh(new THREE.BoxGeometry(2.24, 0.24, 4.58), stripeMat);
  stripe.position.y = 0.82;
  group.add(stripe);

  const crossA = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.05, 1.1), stripeMat);
  crossA.position.set(0, 1.74, -0.35);
  group.add(crossA);
  const crossB = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.05, 0.32), stripeMat);
  crossB.position.set(0, 1.74, -0.35);
  group.add(crossB);

  const windshield = new THREE.Mesh(new THREE.BoxGeometry(1.88, 0.56, 0.14), glassMat);
  windshield.position.set(0, 1.25, 2.26);
  group.add(windshield);

  const redBeaconMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0xef4444,
    emissiveIntensity: 2.0,
  });
  const blueBeaconMat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    emissive: 0x3b82f6,
    emissiveIntensity: 2.0,
  });
  const beaconRed = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.24, 0.38), redBeaconMat);
  beaconRed.position.set(-0.42, 1.84, 1.3);
  group.add(beaconRed);
  const beaconBlue = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.24, 0.38), blueBeaconMat);
  beaconBlue.position.set(0.42, 1.84, 1.3);
  group.add(beaconBlue);

  for (const sx of [-1.08, 1.08]) {
    for (const sz of [-1.4, 1.4]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.24, 12), wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(sx, 0.38, sz);
      group.add(wheel);
    }
  }

  group.userData = { redBeaconMat, blueBeaconMat };
  return group;
}

function buildIntersectionEnvironment(scene) {
  // Deep base ground & subtle architectural grid
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x0b0e13, roughness: 0.95, metalness: 0.05 });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(180, 180), groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.04;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(140, 28, 0x1e2531, 0x141821);
  grid.position.y = -0.02;
  scene.add(grid);

  // Asphalt roads (N-S and E-W)
  const asphaltMat = new THREE.MeshStandardMaterial({ color: 0x171b22, roughness: 0.88, metalness: 0.06 });
  const nsRoad = new THREE.Mesh(new THREE.BoxGeometry(18, 0.08, 124), asphaltMat);
  nsRoad.position.y = 0;
  nsRoad.receiveShadow = true;
  scene.add(nsRoad);

  const ewRoad = new THREE.Mesh(new THREE.BoxGeometry(124, 0.08, 18), asphaltMat);
  ewRoad.position.y = 0.002;
  ewRoad.receiveShadow = true;
  scene.add(ewRoad);

  // Corner concrete curbs & architectural massing blocks
  const curbMat = new THREE.MeshStandardMaterial({ color: 0x151921, roughness: 0.85, metalness: 0.1 });
  const curbEdgeMat = new THREE.MeshStandardMaterial({ color: 0x28303e, roughness: 0.7 });
  const bldgMat = new THREE.MeshStandardMaterial({ color: 0x10141b, roughness: 0.8, metalness: 0.15 });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x232a36, roughness: 0.6 });

  const corners = [
    [1, 1],
    [-1, 1],
    [1, -1],
    [-1, -1],
  ];
  for (const [sx, sz] of corners) {
    const pad = new THREE.Mesh(new THREE.BoxGeometry(48, 0.34, 48), curbMat);
    pad.position.set(sx * 33.2, 0.15, sz * 33.2);
    pad.receiveShadow = true;
    scene.add(pad);

    // Architectural block set back from junction
    const bldg = new THREE.Mesh(new THREE.BoxGeometry(18, 4.2, 18), bldgMat);
    bldg.position.set(sx * 27, 2.3, sz * 27);
    bldg.castShadow = true;
    bldg.receiveShadow = true;
    scene.add(bldg);

    const roofTrim = new THREE.Mesh(new THREE.BoxGeometry(18.3, 0.22, 18.3), trimMat);
    roofTrim.position.set(sx * 27, 4.45, sz * 27);
    scene.add(roofTrim);

    const curbPost = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.45, 1.2), curbEdgeMat);
    curbPost.position.set(sx * 9.8, 0.25, sz * 9.8);
    scene.add(curbPost);
  }

  // Yellow dashed median lines
  const yellowMat = new THREE.MeshBasicMaterial({ color: 0xca8a04 });
  for (let d = 12.5; d <= 56; d += 4.2) {
    for (const sign of [-1, 1]) {
      const dashNS = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 2.1), yellowMat);
      dashNS.rotation.x = -Math.PI / 2;
      dashNS.position.set(0, 0.052, sign * d);
      scene.add(dashNS);

      const dashEW = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 0.28), yellowMat);
      dashEW.rotation.x = -Math.PI / 2;
      dashEW.position.set(sign * d, 0.052, 0);
      scene.add(dashEW);
    }
  }

  // White crosswalk bars & stop lines
  const whiteMat = new THREE.MeshBasicMaterial({ color: 0xdbe2ec });
  for (const offset of [-6.8, -4.1, -1.4, 1.4, 4.1, 6.8]) {
    for (const sign of [-1, 1]) {
      const cwNS = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.8), whiteMat);
      cwNS.rotation.x = -Math.PI / 2;
      cwNS.position.set(offset, 0.052, sign * 9.3);
      scene.add(cwNS);

      const cwEW = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 1.1), whiteMat);
      cwEW.rotation.x = -Math.PI / 2;
      cwEW.position.set(sign * 9.3, 0.052, offset);
      scene.add(cwEW);
    }
  }

  // Stop lines on each incoming half
  for (const lane of LANES) {
    const sp = STOP_STRIP_POS[lane];
    const stopBar = new THREE.Mesh(new THREE.PlaneGeometry(8.1, 0.42), whiteMat);
    stopBar.rotation.x = -Math.PI / 2;
    stopBar.rotation.z = sp.rotY;
    stopBar.position.set(sp.x, 0.054, sp.z);
    scene.add(stopBar);
  }
}

function buildSignalRig(scene) {
  const poleMat = new THREE.MeshStandardMaterial({ color: 0x2b3340, roughness: 0.45, metalness: 0.72 });
  const housingMat = new THREE.MeshStandardMaterial({ color: 0x0c0f14, roughness: 0.5, metalness: 0.5 });

  const signalRefs = {};

  for (const lane of LANES) {
    const spec = MAST_SPECS[lane];
    const [px, pz] = spec.post;
    const [hx, hz] = spec.head;

    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.24, 6.4, 10), poleMat);
    pole.position.set(px, 3.2, pz);
    pole.castShadow = true;
    scene.add(pole);

    const armLength = Math.hypot(hx - px, hz - pz) + 0.6;
    const armGeo = new THREE.BoxGeometry(
      spec.armDir[0] !== 0 ? armLength : 0.22,
      0.22,
      spec.armDir[1] !== 0 ? armLength : 0.22
    );
    const arm = new THREE.Mesh(armGeo, poleMat);
    arm.position.set((px + hx) / 2, 6.2, (pz + hz) / 2);
    arm.castShadow = true;
    scene.add(arm);

    const headGroup = new THREE.Group();
    headGroup.position.set(hx, 6.2, hz);
    headGroup.rotation.y = spec.rotY;

    const housing = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.96, 0.64), housingMat);
    housing.castShadow = true;
    headGroup.add(housing);

    const makeLamp = (xOffset, colorHex) => {
      const mat = new THREE.MeshStandardMaterial({
        color: 0x1f252e,
        emissive: colorHex,
        emissiveIntensity: 0.06,
        roughness: 0.3,
      });
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.28, 14, 14), mat);
      sphere.position.set(xOffset, 0, 0);
      sphere.scale.set(1, 1, 1.3);
      headGroup.add(sphere);
      return mat;
    };

    const redMat = makeLamp(-0.76, 0xef4444);
    const amberMat = makeLamp(0.0, 0xf59e0b);
    const greenMat = makeLamp(0.76, 0x10b981);
    scene.add(headGroup);

    // Pavement signal strip behind stop line
    const sp = STOP_STRIP_POS[lane];
    const stripMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 0.7,
      roughness: 0.4,
    });
    const stripMesh = new THREE.Mesh(new THREE.BoxGeometry(7.9, 0.04, 0.28), stripMat);
    stripMesh.rotation.y = sp.rotY;
    stripMesh.position.set(
      lane === 'E' ? sp.x + 0.45 : lane === 'W' ? sp.x - 0.45 : sp.x,
      0.055,
      lane === 'N' ? sp.z - 0.45 : lane === 'S' ? sp.z + 0.45 : sp.z
    );
    scene.add(stripMesh);

    // Real point light wash onto the approach stop line
    const light = new THREE.PointLight(0xef4444, 2.2, 16, 1.7);
    light.position.set(hx, 4.2, hz);
    scene.add(light);

    signalRefs[lane] = { redMat, amberMat, greenMat, stripMat, light };
  }

  // Center junction boundary ring for ALL_RED / EMERGENCY_GREEN visual feedback
  const boxRingMat = new THREE.MeshStandardMaterial({
    color: 0x262d3a,
    emissive: 0x000000,
    emissiveIntensity: 0.0,
    roughness: 0.6,
  });
  const boxBorder = new THREE.Mesh(new THREE.RingGeometry(11.6, 12.1, 4), boxRingMat);
  boxBorder.rotation.x = -Math.PI / 2;
  boxBorder.rotation.z = Math.PI / 4;
  boxBorder.position.y = 0.053;
  scene.add(boxBorder);

  return { signalRefs, boxRingMat };
}

export default function ThreeIntersectionScene({ snapshot }) {
  const [viewMode, setViewMode] = useState('3d');
  const [activePreset, setActivePreset] = useState('ISO');
  const [webglFailed, setWebglFailed] = useState(() => {
    try {
      const c = document.createElement('canvas');
      return !(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
    } catch {
      return true;
    }
  });

  const containerRef = useRef(null);
  const snapshotRef = useRef(snapshot);
  const hudRefs = useRef({ N: null, E: null, S: null, W: null });
  const cameraLerpRef = useRef({ active: false, pos: new THREE.Vector3(), target: new THREE.Vector3() });

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  const handlePreset = (presetId) => {
    const preset = CAMERA_PRESETS[presetId];
    if (!preset) return;
    setActivePreset(presetId);
    cameraLerpRef.current = {
      active: true,
      pos: new THREE.Vector3(...preset.pos),
      target: new THREE.Vector3(...preset.target),
    };
  };

  useEffect(() => {
    if (viewMode !== '3d' || webglFailed) return undefined;
    const container = containerRef.current;
    if (!container) return undefined;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    } catch {
      queueMicrotask(() => setWebglFailed(true));
      return undefined;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    const initW = Math.max(container.clientWidth, 320);
    const initH = Math.max(container.clientHeight, 360);
    renderer.setSize(initW, initH);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090b0e);
    scene.fog = new THREE.Fog(0x090b0e, 68, 150);

    const camera = new THREE.PerspectiveCamera(44, initW / initH, 0.5, 220);
    camera.position.set(...CAMERA_PRESETS.ISO.pos);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.06;
    controls.minDistance = 18;
    controls.maxDistance = 85;
    controls.target.set(...CAMERA_PRESETS.ISO.target);

    const cancelLerp = () => {
      cameraLerpRef.current.active = false;
    };
    controls.addEventListener('start', cancelLerp);

    // Lighting
    const hemi = new THREE.HemisphereLight(0x9fb3c8, 0x141820, 0.9);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xe2e8f0, 1.45);
    keyLight.position.set(28, 46, 26);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 10;
    keyLight.shadow.camera.far = 120;
    keyLight.shadow.camera.left = -38;
    keyLight.shadow.camera.right = 38;
    keyLight.shadow.camera.top = 38;
    keyLight.shadow.camera.bottom = -38;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x60a5fa, 0.35);
    rimLight.position.set(-32, 22, -28);
    scene.add(rimLight);

    buildIntersectionEnvironment(scene);
    const { signalRefs, boxRingMat } = buildSignalRig(scene);

    // Dynamic vehicle pools
    const vehicleGroup = new THREE.Group();
    scene.add(vehicleGroup);

    const stdPool = { N: [], E: [], S: [], W: [] };
    const emgPool = { N: [], E: [], S: [], W: [] };
    for (const lane of LANES) {
      for (let i = 0; i < 10; i += 1) {
        const mesh = buildStandardCarMesh(CAR_PALETTE[(i + LANES.indexOf(lane)) % CAR_PALETTE.length]);
        mesh.visible = false;
        vehicleGroup.add(mesh);
        stdPool[lane].push(mesh);
      }
      for (let i = 0; i < 2; i += 1) {
        const emgMesh = buildAmbulanceMesh();
        emgMesh.visible = false;
        vehicleGroup.add(emgMesh);
        emgPool[lane].push(emgMesh);
      }
    }

    const crossingVehicles = [];
    const sirenPointLight = new THREE.PointLight(0xef4444, 0, 18, 1.6);
    scene.add(sirenPointLight);

    let lastObservedTimestamp = snapshotRef.current.timestamp;
    let prevClock = performance.now();
    let rafId = 0;
    const projVec = new THREE.Vector3();

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.max(entry.contentRect.width, 320);
        const h = Math.max(entry.contentRect.height, 360);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      }
    });
    resizeObserver.observe(container);

    const spawnCrossing = (lane, isEmergency, idxOffset) => {
      const origin = getCrossingOrigin(lane, idxOffset);
      const mesh = isEmergency
        ? buildAmbulanceMesh()
        : buildStandardCarMesh(CAR_PALETTE[(idxOffset + Math.floor(Math.random() * 5)) % CAR_PALETTE.length]);
      mesh.position.set(origin.x, 0.04, origin.z);
      mesh.rotation.y = origin.yaw;
      vehicleGroup.add(mesh);
      const tr = LANE_TRAVEL[lane];
      crossingVehicles.push({
        mesh,
        lane,
        isEmergency,
        vx: tr.dx * (isEmergency ? 20 : 16.5),
        vz: tr.dz * (isEmergency ? 20 : 16.5),
        life: 0,
      });
    };

    const disposeObject3D = (root) => {
      root.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      });
    };

    const animate = (now) => {
      rafId = requestAnimationFrame(animate);
      const dtSec = Math.min(0.05, Math.max(0.001, (now - prevClock) / 1000));
      prevClock = now;

      const snap = snapshotRef.current;

      // Camera preset smoothing
      if (cameraLerpRef.current.active) {
        const alpha = 1 - Math.exp(-6.5 * dtSec);
        camera.position.lerp(cameraLerpRef.current.pos, alpha);
        controls.target.lerp(cameraLerpRef.current.target, alpha);
        if (
          camera.position.distanceTo(cameraLerpRef.current.pos) < 0.08 &&
          controls.target.distanceTo(cameraLerpRef.current.target) < 0.08
        ) {
          camera.position.copy(cameraLerpRef.current.pos);
          controls.target.copy(cameraLerpRef.current.target);
          cameraLerpRef.current.active = false;
        }
      }
      controls.update();

      // Detect tick change to spawn discharging crossing vehicles
      if (snap.timestamp !== lastObservedTimestamp) {
        if (snap.timestamp < lastObservedTimestamp) {
          while (crossingVehicles.length > 0) {
            const item = crossingVehicles.pop();
            vehicleGroup.remove(item.mesh);
            disposeObject3D(item.mesh);
          }
        } else {
          for (const lane of LANES) {
            const emgDepart = snap.emergencyDischarging?.[lane] ?? 0;
            const stdDepart = snap.discharging?.[lane] ?? 0;
            for (let i = 0; i < Math.min(emgDepart, 1); i += 1) {
              spawnCrossing(lane, true, i);
            }
            for (let i = 0; i < Math.min(stdDepart, 2); i += 1) {
              spawnCrossing(lane, false, i);
            }
          }
        }
        lastObservedTimestamp = snap.timestamp;
      }

      // Update signal mast lights & pavement status strips
      for (const lane of LANES) {
        const lightColor = snap.lights[lane];
        const refs = signalRefs[lane];
        const isGreen = lightColor === LightColor.GREEN;
        const isAmber = lightColor === LightColor.AMBER;
        const isRed = lightColor === LightColor.RED;

        refs.redMat.emissiveIntensity = isRed ? 2.8 : 0.05;
        refs.amberMat.emissiveIntensity = isAmber ? 2.8 : 0.05;
        refs.greenMat.emissiveIntensity = isGreen ? 2.8 : 0.05;

        const hex = isGreen ? 0x10b981 : isAmber ? 0xf59e0b : 0xef4444;
        refs.stripMat.color.setHex(hex);
        refs.stripMat.emissive.setHex(hex);
        refs.stripMat.emissiveIntensity = isGreen ? 1.4 : isAmber ? 1.2 : 0.45;
        refs.light.color.setHex(hex);
        refs.light.intensity = isGreen || isAmber ? 3.2 : 1.4;
      }

      // Center box ring status
      if (snap.state === SignalState.ALL_RED) {
        boxRingMat.emissive.setHex(0xef4444);
        boxRingMat.emissiveIntensity = 1.2 + 0.5 * Math.sin(now * 0.012);
      } else if (snap.state === SignalState.EMERGENCY_GREEN) {
        boxRingMat.emissive.setHex(0xef4444);
        boxRingMat.emissiveIntensity = 0.9 + 0.6 * Math.sin(now * 0.015);
      } else if (snap.state === SignalState.TRUNCATED_GREEN) {
        boxRingMat.emissive.setHex(0xf59e0b);
        boxRingMat.emissiveIntensity = 1.0;
      } else {
        boxRingMat.emissive.setHex(0x000000);
        boxRingMat.emissiveIntensity = 0.0;
      }

      // Update queued vehicles smoothly
      const strobeRed = Math.sin(now * 0.022) > 0;
      let activeSirenTarget = null;
      const moveAlpha = 1 - Math.exp(-9 * dtSec);

      for (const lane of LANES) {
        const emgCount = Math.min(snap.emergencies[lane] ?? 0, 2);
        const stdCount = Math.min(snap.queues[lane] ?? 0, 10 - emgCount);

        for (let i = 0; i < 2; i += 1) {
          const mesh = emgPool[lane][i];
          if (i < emgCount) {
            const target = getQueueTransform(lane, i);
            if (!mesh.visible) {
              mesh.visible = true;
              mesh.position.set(target.x, 0.04, target.z);
            } else {
              mesh.position.x += (target.x - mesh.position.x) * moveAlpha;
              mesh.position.z += (target.z - mesh.position.z) * moveAlpha;
            }
            mesh.rotation.y = target.yaw;
            mesh.userData.redBeaconMat.emissiveIntensity = strobeRed ? 3.5 : 0.2;
            mesh.userData.blueBeaconMat.emissiveIntensity = strobeRed ? 0.2 : 3.5;
            if (!activeSirenTarget) activeSirenTarget = mesh.position;
          } else {
            mesh.visible = false;
          }
        }

        for (let i = 0; i < 10; i += 1) {
          const mesh = stdPool[lane][i];
          if (i < stdCount) {
            const target = getQueueTransform(lane, emgCount + i);
            if (!mesh.visible) {
              mesh.visible = true;
              mesh.position.set(target.x, 0.04, target.z);
            } else {
              mesh.position.x += (target.x - mesh.position.x) * moveAlpha;
              mesh.position.z += (target.z - mesh.position.z) * moveAlpha;
            }
            mesh.rotation.y = target.yaw;
          } else {
            mesh.visible = false;
          }
        }
      }

      // Update crossing vehicles moving through intersection
      for (let i = crossingVehicles.length - 1; i >= 0; i -= 1) {
        const cv = crossingVehicles[i];
        cv.life += dtSec;
        cv.mesh.position.x += cv.vx * dtSec;
        cv.mesh.position.z += cv.vz * dtSec;
        if (cv.isEmergency) {
          cv.mesh.userData.redBeaconMat.emissiveIntensity = strobeRed ? 3.8 : 0.2;
          cv.mesh.userData.blueBeaconMat.emissiveIntensity = strobeRed ? 0.2 : 3.8;
          activeSirenTarget = cv.mesh.position;
        }
        if (Math.abs(cv.mesh.position.x) > 34 || Math.abs(cv.mesh.position.z) > 34 || cv.life > 4.0) {
          vehicleGroup.remove(cv.mesh);
          disposeObject3D(cv.mesh);
          crossingVehicles.splice(i, 1);
        }
      }

      // Emergency strobe point light
      if (activeSirenTarget) {
        sirenPointLight.position.set(activeSirenTarget.x, 2.8, activeSirenTarget.z);
        sirenPointLight.color.setHex(strobeRed ? 0xef4444 : 0x3b82f6);
        sirenPointLight.intensity = 4.5;
      } else {
        sirenPointLight.intensity = 0;
      }

      renderer.render(scene, camera);

      // Project 3D anchor coordinates to 2D overlay badges
      const width = container.clientWidth;
      const height = container.clientHeight;
      for (const lane of LANES) {
        const el = hudRefs.current[lane];
        if (!el) continue;
        const [ax, ay, az] = MAST_SPECS[lane].hudAnchor;
        projVec.set(ax, ay, az).project(camera);
        if (projVec.z > 1) {
          el.style.opacity = '0';
          continue;
        }
        const px = Math.round((projVec.x * 0.5 + 0.5) * width);
        const py = Math.round((-projVec.y * 0.5 + 0.5) * height);
        el.style.opacity = '1';
        el.style.transform = `translate3d(${px}px, ${py}px, 0) translate(-50%, -100%)`;
      }
    };

    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      controls.removeEventListener('start', cancelLerp);
      controls.dispose();
      disposeObject3D(scene);
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [viewMode, webglFailed]);

  const {
    timestamp,
    state,
    activeLane,
    clockwiseLane,
    resumeLane,
    preemptTargetLane,
    remainingTime,
    lights,
    isEmergencyPreemption,
    queues,
    emergencies,
    carCountSwitch,
  } = snapshot;

  return (
    <div className="sim-card twin-panel">
      <div className="sim-card-header">
        <div>
          <h3>4-Way Chowk Digital Twin</h3>
          <p className="sim-card-sub">
            {viewMode === '3d' && !webglFailed
              ? 'Interactive Three.js WebGL scene · Drag to orbit, scroll to zoom'
              : '2D orthographic schematic view · Left-hand Indian chowk geometry'}
          </p>
        </div>

        <div className="twin-toolbar">
          <div className="twin-btn-group" role="group" aria-label="Viewport mode">
            <button
              type="button"
              className={`twin-btn ${viewMode === '3d' && !webglFailed ? 'active' : ''}`}
              disabled={webglFailed}
              onClick={() => setViewMode('3d')}
            >
              3D TWIN
            </button>
            <button
              type="button"
              className={`twin-btn ${viewMode === '2d' || webglFailed ? 'active' : ''}`}
              onClick={() => setViewMode('2d')}
            >
              2D PLAN
            </button>
          </div>

          {viewMode === '3d' && !webglFailed && (
            <div className="twin-btn-group" role="group" aria-label="Camera presets">
              {Object.keys(CAMERA_PRESETS).map((presetKey) => (
                <button
                  key={presetKey}
                  type="button"
                  className={`twin-btn ${activePreset === presetKey ? 'active' : ''}`}
                  onClick={() => handlePreset(presetKey)}
                >
                  {presetKey}
                </button>
              ))}
            </div>
          )}

          <span className="sim-clock-pill">T+{timestamp}s</span>
        </div>
      </div>

      {/* Compact Approach Status Strip */}
      <div className="sim-phase-strip">
        {LANES.map((lane) => {
          const isCurrentI = lane === clockwiseLane;
          const isActiveGreen = lights[lane] === LightColor.GREEN;
          const isEmgTarget = lane === preemptTargetLane || emergencies[lane] > 0;
          const isDemandTarget =
            !isEmergencyPreemption &&
            (carCountSwitch?.targetLane === lane || carCountSwitch?.nextDemandLane === lane);
          const isResumeNext = lane === resumeLane && lane !== activeLane;

          return (
            <div
              key={lane}
              className={`sim-phase-chip ${isActiveGreen ? 'is-green' : ''} ${
                isEmgTarget ? 'is-emg' : isDemandTarget ? 'is-demand' : ''
              }`}
            >
              <span className="sim-phase-lane">
                <span>LANE {lane}</span>
                <small className="sim-phase-count">C={queues[lane] ?? 0}</small>
              </span>
              <span className="sim-phase-tags">
                {isCurrentI && <span className="tag tag-i">PHASE i</span>}
                {isEmgTarget && <span className="tag tag-j">EMG j</span>}
                {isDemandTarget && <span className="tag tag-demand">HIGH COUNT</span>}
                {isResumeNext && <span className="tag tag-next">NEXT i+1</span>}
              </span>
            </div>
          );
        })}
      </div>

      {viewMode === '2d' || webglFailed ? (
        <IntersectionDiagram snapshot={snapshot} embedded />
      ) : (
        <div className="twin-viewport" ref={containerRef}>
          <div className="twin-hud-layer" aria-hidden="true">
            {LANES.map((lane) => {
              const lightColor = lights[lane];
              const isActive = lane === activeLane || state === SignalState.ALL_RED;
              const qCount = queues[lane] ?? 0;
              const eCount = emergencies[lane] ?? 0;
              const remLabel =
                state === SignalState.EMERGENCY_GREEN && lane === activeLane
                  ? 'HOLD'
                  : `${remainingTime.toFixed(0)}s`;

              return (
                <div
                  key={lane}
                  ref={(el) => {
                    hudRefs.current[lane] = el;
                  }}
                  className={`twin-hud-chip aspect-${lightColor.toLowerCase()} ${
                    eCount > 0 ? 'has-emg' : ''
                  }`}
                >
                  <span className={`hud-dot dot-${lightColor.toLowerCase()}`} />
                  <span className="hud-lane">{lane}</span>
                  <span className="hud-sep">|</span>
                  <span className="hud-q">C:{qCount}</span>
                  {eCount > 0 && <span className="hud-emg">E:{eCount}</span>}
                  {isActive && <span className="hud-rem">{remLabel}</span>}
                </div>
              );
            })}
          </div>

          <div className="twin-corner-legend">
            <span>
              <i className="legend-swatch swatch-green" /> GREEN
            </span>
            <span>
              <i className="legend-swatch swatch-amber" /> AMBER
            </span>
            <span>
              <i className="legend-swatch swatch-red" /> RED
            </span>
            <span>
              <i className="legend-swatch swatch-emg" /> EMERGENCY
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
