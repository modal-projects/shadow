# shadow

play charades with Kimi K3.1 Pro

```bash
uv run --group bench modal run -e demos replays.py --run <run_name>
uv run --group bench modal run -e demos replays.py --run <run_name> --build
cd web && npm ci && npm run build
uv run modal serve -e demos app.py
```

`replays.py --run <run>` records the benchmark into the `shadowbench-runs` volume; `--build` exports quantized `.bin` replays plus `index.json`/`rig.json` into the `shadow-replays` volume, which the app mounts and serves. Guesses are judged by the model endpoint, five wrong guesses per round.

## ShadowBench

```bash
uv run --group bench modal run -e demos replays.py --run <run_name>
```

Runs the published [ShadowBench](https://huggingface.co/datasets/perpetual-research/shadowbench) agent and reviewed targets, pinned in `vendor/shadowbench/UPSTREAM.json`, against the model endpoint: 100 tool calls, 32 looks, penalized book peeks, medium reasoning effort. `--task bursill_bunny` runs one plate. Each plate saves the upstream trajectory and result plus `timing.json` under `<run_name>/` in the `shadowbench-runs` volume.

## Credits

- [flashlight](https://www.innerscene.com/tools/library/3d-parts/led-flashlight-torch-aluminium-tail-switch-b30191c2)
