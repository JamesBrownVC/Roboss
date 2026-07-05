/* 3D voxel map view (Unitree-app style) — Three.js ES module */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const container = document.getElementById("lidar-3d");
const W = () => container.clientWidth || 320;
const H = 340;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);

const camera = new THREE.PerspectiveCamera(60, W() / H, 0.05, 200);
camera.up.set(0, 0, 1);            // world: x=forward, y=left, z=up
camera.position.set(-4.5, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(W(), H);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxDistance = 30;

scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, 2, 6);
scene.add(sun);

const grid = new THREE.GridHelper(12, 24, 0x30363d, 0x1b2128);
grid.rotation.x = Math.PI / 2;     // put grid in XY plane (z-up)
scene.add(grid);

// ------------------------------------------------------------ voxel cloud
const MAX = 9000;
const voxels = new THREE.InstancedMesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshLambertMaterial(),
  MAX
);
voxels.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
voxels.count = 0;
scene.add(voxels);

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _c = new THREE.Color();
const _target = new THREE.Vector3();

// ------------------------------------------------------------ robot marker
const robot = new THREE.Group();
const body = new THREE.Mesh(
  new THREE.BoxGeometry(0.55, 0.28, 0.2),
  new THREE.MeshLambertMaterial({ color: 0x3fb950 })
);
body.position.z = 0.05;
const nose = new THREE.Mesh(
  new THREE.ConeGeometry(0.09, 0.22, 10),
  new THREE.MeshLambertMaterial({ color: 0xe6edf3 })
);
nose.rotation.z = -Math.PI / 2;
nose.position.set(0.38, 0, 0.05);
robot.add(body, nose);
scene.add(robot);

// ------------------------------------------------------------------ update
function update(pos, meta, rpos, yaw) {
  const res = (meta && meta.resolution) || 0.05;
  const org = (meta && meta.origin) || [0, 0, 0];
  const n = Math.min((pos.length / 3) | 0, MAX);
  const size = res * 0.92;
  _s.set(size, size, size);
  _q.identity();

  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = org[2] + pos[i * 3 + 2] * res;
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
  }
  const zr = Math.max(zmax - zmin, 0.001);

  for (let i = 0; i < n; i++) {
    const wx = org[0] + pos[i * 3] * res;
    const wy = org[1] + pos[i * 3 + 1] * res;
    const wz = org[2] + pos[i * 3 + 2] * res;
    _p.set(wx, wy, wz);
    _m.compose(_p, _q, _s);
    voxels.setMatrixAt(i, _m);
    // app-like palette: green floor -> yellow -> red/magenta walls
    const h = (wz - zmin) / zr;
    _c.setHSL(Math.max(0, 0.38 - 0.46 * h), 0.95, 0.38 + 0.18 * h);
    voxels.setColorAt(i, _c);
  }
  voxels.count = n;
  voxels.instanceMatrix.needsUpdate = true;
  if (voxels.instanceColor) voxels.instanceColor.needsUpdate = true;

  if (rpos && rpos.length >= 2) {
    robot.position.set(rpos[0], rpos[1], (rpos[2] || 0.3) - 0.15);
    if (typeof yaw === "number") robot.rotation.z = yaw;
    _target.set(rpos[0], rpos[1], 0.2);
    controls.target.lerp(_target, 0.15);
  }
}
window.lidar3d = { update };

window.addEventListener("resize", () => {
  camera.aspect = W() / H;
  camera.updateProjectionMatrix();
  renderer.setSize(W(), H);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
