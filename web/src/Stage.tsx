import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

type Side = "left" | "right";
export type StageControls = { pose: (vertices: Float32Array[]) => void; reset: () => void };
type Rig = Record<"vertices" | "faces", Record<Side, string>>;

function decode(value: string) {
  return Uint8Array.from(atob(value), (c) => c.charCodeAt(0)).buffer;
}

export default function Stage({ onError, onReady }: {
  onError: (error: string) => void;
  onReady: (controls: StageControls) => void;
}) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = host.current!;
    const abort = new AbortController();
    const styles = getComputedStyle(element);
    const color = (token: string) => styles.getPropertyValue(`--color-v2-${token}`).trim();
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(color("primary-black"));
    const camera = new THREE.PerspectiveCamera(38, 1, 0.02, 30);
    camera.position.set(1.5, 0.8, 4.1);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.1, 0.85);
    controls.enableDamping = true;
    controls.minDistance = 1;
    controls.maxDistance = 7;
    controls.maxPolarAngle = Math.PI * 0.85;

    scene.add(new THREE.AmbientLight(color("primary-neutral-500"), 0.25));
    const light = new THREE.PointLight(color("primary-neutral-100"), 10, 10, 2);
    light.position.set(0, 0, 2.5);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    light.shadow.camera.near = 0.05;
    light.shadow.camera.far = 5;
    light.shadow.bias = -0.00005;
    light.shadow.normalBias = 0.001;
    scene.add(light);
    const flashlight = new THREE.Group();
    flashlight.position.copy(light.position);
    scene.add(flashlight);
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    loader.load("/flashlight.glb", ({ scene: model }) => {
      if (abort.signal.aborted) return;
      model.scale.setScalar(2);
      model.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.castShadow = false;
        for (const material of [object.material].flat() as THREE.MeshStandardMaterial[]) {
          material.emissive.set("#ffffff");
          material.emissiveMap = material.map;
          material.emissiveIntensity = material.name.includes("black") || object.name.startsWith("LED_emitter") ? 2.5 : 0.3;
        }
      });
      flashlight.add(model);
    }, undefined, () => {
      if (!abort.signal.aborted) onError("Could not load the flashlight model");
    });
    const beam = new THREE.Mesh(
      new THREE.ConeGeometry(0.7, light.position.z, 48, 1, true).rotateX(Math.PI / 2),
      new THREE.ShaderMaterial({
        uniforms: { glow: { value: new THREE.Color(color("primary-neutral-100")) } },
        vertexShader: "varying vec2 vUv; varying float facing; void main() { vUv = uv; vec4 view = modelViewMatrix * vec4(position, 1.0); facing = abs(dot(normalize(normalMatrix * normal), normalize(-view.xyz))); gl_Position = projectionMatrix * view; }",
        fragmentShader: "uniform vec3 glow; varying vec2 vUv; varying float facing; void main() { float fade = smoothstep(0.0, 0.25, vUv.y); gl_FragColor = vec4(glow, 0.1 * fade * facing); }",
        transparent: true, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthTest: true, depthWrite: false,
      }),
    );
    beam.position.copy(light.position).multiplyScalar(0.5);
    scene.add(beam);

    const wall = new THREE.Mesh(
      new THREE.PlaneGeometry(2.4, 1.9),
      new THREE.MeshStandardMaterial({ color: color("primary-neutral-200"), roughness: 1, side: THREE.DoubleSide }),
    );
    wall.receiveShadow = true;
    scene.add(wall);
    const aperture = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.4, -0.3, 0.002),
        new THREE.Vector3(0.4, -0.3, 0.002),
        new THREE.Vector3(0.4, 0.3, 0.002),
        new THREE.Vector3(-0.4, 0.3, 0.002),
      ]),
      new THREE.LineBasicMaterial({ color: color("primary-green-200"), transparent: true, opacity: 0.25 }),
    );
    scene.add(aperture);

    const hands: THREE.Mesh[] = [];
    let transition: { start: number; from: Float32Array[]; to: Float32Array[] } | null = null;

    function pose(to: Float32Array[], immediate = false) {
      transition = {
        start: performance.now(),
        from: hands.map((hand) => (hand.geometry.getAttribute("position").array as Float32Array).slice()),
        to,
      };
      if (immediate) update(1);
    }

    function update(amount: number) {
      if (!transition) return;
      const { from, to } = transition;
      hands.forEach((hand, side) => {
        const position = hand.geometry.getAttribute("position");
        for (let i = 0; i < position.array.length; i++) {
          position.array[i] = from[side][i] + (to[side][i] - from[side][i]) * amount;
        }
        position.needsUpdate = true;
        hand.geometry.computeVertexNormals();
        hand.geometry.computeBoundingSphere();
      });
      renderer.shadowMap.needsUpdate = true;
      if (amount === 1) transition = null;
    }

    fetch("/rig.json", { signal: abort.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Rig request failed (${response.status})`);
        return response.json() as Promise<Rig>;
      })
      .then((rig) => {
        if (abort.signal.aborted) return;
        const skin = new THREE.Color().setRGB(222 / 255, 168 / 255, 128 / 255, THREE.SRGBColorSpace);
        for (const side of ["left", "right"] as const) {
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(decode(rig.vertices[side])), 3));
          geometry.setIndex(new THREE.BufferAttribute(new Uint16Array(decode(rig.faces[side])), 1));
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: skin, roughness: 0.75 }));
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          scene.add(mesh);
          hands.push(mesh);
        }
        renderer.shadowMap.needsUpdate = true;
        const rest = (["left", "right"] as const).map(
          (side) => new Float32Array(decode(rig.vertices[side])),
        );
        onReady({ pose, reset: () => { pose(rest, true); } });
      })
      .catch((error: Error) => {
        if (!abort.signal.aborted) onError(error.message);
      });

    const resize = new ResizeObserver(() => {
      camera.aspect = element.clientWidth / element.clientHeight;
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(
        Math.tan(THREE.MathUtils.degToRad(38) / 2) * Math.max(1, 1.5 / camera.aspect),
      ));
      camera.updateProjectionMatrix();
      renderer.setSize(element.clientWidth, element.clientHeight);
    });
    resize.observe(element);
    renderer.setAnimationLoop((time) => {
      if (transition) {
        update(Math.max(0, Math.min(1, (time - transition.start) / 500)));
      }
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      abort.abort();
      resize.disconnect();
      controls.dispose();
      renderer.setAnimationLoop(null);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      light.shadow.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [onError, onReady]);

  return <div className="stage" ref={host} aria-label="Two articulated hands between a light and a shadow wall" />;
}
