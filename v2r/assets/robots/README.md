# Robot models (MuJoCo Menagerie)

The MJCF files (`*/scene.xml`, `*/g1.xml`, `*/panda.xml`, `*/go2.xml`, ...)
are committed; the binary mesh assets (`*/assets/`, ~100 MB) are NOT — fetch
them from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie
cd mujoco_menagerie
git sparse-checkout set unitree_g1 franka_emika_panda unitree_go2
cp -r unitree_g1/assets      ../assets/robots/g1/assets
cp -r franka_emika_panda/assets ../assets/robots/franka/assets
cp -r unitree_go2/assets     ../assets/robots/go2/assets
```

Without the meshes, `physics_validate` automatically falls back from the
MuJoCo replay (joint limits from the model, self-collision, ground
penetration, foot slide) to numpy kinematic checks (limits from
`config/robots.yaml`, velocity/acceleration) and marks the contact-based
checks as skipped in `physics_report.json`.

Licenses: Unitree G1/Go2 (BSD-3), Franka Emika Panda (Apache-2.0 /
BSD-3 per Menagerie); see each robot directory's LICENSE file in Menagerie.
