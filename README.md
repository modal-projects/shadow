# shadow

play charades with Deepseek v4.1 Flash 1017

https://github.com/user-attachments/assets/3b3fa6d5-24b2-4db3-be85-e9e1a13226cf

```bash
# demo
uv run --group bench modal run -e demos replays.py --run <run_name>
uv run --group bench modal run -e demos replays.py --run <run_name> --build
cd web && npm ci && npm run build
uv run modal serve -e demos app.py

# benchmark
uv run --group bench modal run -e demos replays.py --run <run_name>
```

<details>
  <summary>benchmarks results</summary>
  <img width="1258" height="872" alt="scores" src="https://github.com/user-attachments/assets/1601fac8-3274-4c3d-8955-46ba122168a2" />
  <img width="1123" height="872" alt="score_vs_cost" src="https://github.com/user-attachments/assets/a5b0b9c9-27b9-483d-b44b-c1223211be4d" />
  <img width="1188" height="872" alt="score_vs_calls" src="https://github.com/user-attachments/assets/25c971d4-a65c-43e8-bb40-4bcf370ffd1b" />
</details>

## Credits

- [flashlight](https://www.innerscene.com/tools/library/3d-parts/led-flashlight-torch-aluminium-tail-switch-b30191c2)
