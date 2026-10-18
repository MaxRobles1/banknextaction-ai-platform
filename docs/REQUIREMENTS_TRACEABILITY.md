# Matriz de trazabilidad de requisitos

**Fuente:** `Proyecto_Final_Modulo1.pdf` (§5.1 núcleo, §5.2 extras, §7 rúbrica, §10 checklist).

**Estados:** `[ ]` pendiente · `[~]` en curso · `[x]` con evidencia verificable · `[!]` bloqueado.

Ningún requisito se marca `[x]` sin evidencia ejecutable o visual.

## Núcleo obligatorio (100 %)

| Requisito | Fuente | Peso | Archivo(s) | Evidencia | Estado |
|---|---|---|---|---|---|
| src-layout (`src/`, `tests/`, `notebooks/`, `configs/`, `reports/`) | §5.1.1 | 10 % | `src/banknextaction/` | árbol del repo | `[~]` |
| Type hints en funciones públicas | §5.1.1 | | `src/` | ruff y revisión | `[ ]` |
| Decorador con propósito | §5.1.1 | | `utils/timing.py` | latencia en logs y API | `[ ]` |
| dataclass/NamedTuple | §5.1.1 | | `config.py` | pruebas | `[ ]` |
| Context manager | §5.1.1 | | `config.py` | pruebas | `[ ]` |
| Configuración en YAML | §5.1.1 | | `configs/config.yaml` | archivo | `[~]` |
| `.env.example` sin secretos; `.env` ignorado | §5.1.1 | | `.env.example`, `.gitignore` | `check-ignore` | `[~]` |
| uv + `pyproject` + lockfile | §5.1.2 | 10 % | `pyproject.toml`, `uv.lock` | `uv sync --locked` | `[~]` |
| ruff configurado | §5.1.2 | | `pyproject.toml` | `ruff check` | `[~]` |
| pre-commit (ruff, trailing, EOF, check-yaml) | §5.1.2 | | `.pre-commit-config.yaml` | `pre-commit run --all-files` | `[~]` |
| logging en `src/` (sin prints) | §5.1.2 | | `src/` | regla ruff T20 | `[ ]` |
| 4 pruebas: carga, transformación, salida del modelo, contrato API | §5.1.2 | | `tests/` | `pytest -q` | `[ ]` |
| Clase abstracta (`BaseModel`) | §5.1.3 | 10 % | `models/base.py` | pruebas | `[ ]` |
| Separación de responsabilidades | §5.1.3 | | `data/`, `features/`, `models/`, `training/`, `evaluation/`, `api/` | árbol | `[ ]` |
| Patrón de diseño documentado | §5.1.3 | | `models/registry.py`, README | README | `[ ]` |
| NumPy vectorizado y pandas con dtypes y faltantes | §5.1.4 | 10 % | `data/`, `features/` | código y pruebas | `[ ]` |
| 3 gráficas (EDA, curvas, resultados) | §5.1.4 | | `reports/figures/` | PNG en README | `[ ]` |
| Pipeline + ColumnTransformer | §5.1.5 | 10 % | `models/sklearn_model.py` | prueba | `[ ]` |
| CV con `TimeSeriesSplit` | §5.1.5 | | `models/sklearn_model.py` | reporte | `[ ]` |
| GridSearch con ≥2 hiperparámetros | §5.1.5 | | `configs/config.yaml` | README | `[ ]` |
| ≥2 métricas, iguales para ambos modelos | §5.1.5 | | `evaluation/` | tabla | `[ ]` |
| `nn.Module` con `forward` explícito | §5.1.6 | 20 % | `models/torch_model.py` | prueba de salida | `[ ]` |
| `Dataset` propio y `DataLoader` | §5.1.6 | | `training/` | prueba | `[ ]` |
| Tres montones, prueba abierta una vez | §5.1.6 | | `data/split.py` | manifest | `[ ]` |
| Bucle de dos mitades (`train`/`eval`/`no_grad`) | §5.1.6 | | `training/loop.py` | código | `[ ]` |
| Historia por época | §5.1.6 | | `reports/` | JSON y figura | `[ ]` |
| Curvas con diagnóstico | §5.1.6 | | README | figura y texto | `[ ]` |
| Paro temprano **con restauración** | §5.1.6 | | `training/loop.py` | log de la época restaurada | `[ ]` |
| Checkpoint completo (pesos, optimizador, época, métrica) | §5.1.6 | | `models/mlp_checkpoint.pt` | prueba de recarga | `[ ]` |
| Métricas con piso | §5.1.6 | | `evaluation/` | tabla | `[ ]` |
| Umbral elegido en validación | §5.1.6 | | `evaluation/threshold.py` | recibo | `[ ]` |
| Tabla comparativa con conclusión | §5.1.6 | | README | tabla | `[ ]` |
| `main` protegida y trabajo por PR | §5.1.7 | 10 % | GitHub | captura GH013 | `[ ]` |
| Conventional commits | §5.1.7 | | historial | `git log` | `[ ]` |
| PRs con descripción | §5.1.7 | | GitHub | PRs | `[ ]` |
| Hooks activos | §5.1.7 | | `.git/hooks` | commits rechazados | `[~]` |
| `.gitignore` útil; modelo versionado | §5.1.7 | | `.gitignore` | `git ls-files` | `[~]` |
| Dockerfile (slim, lockfile, modelo, `EXPOSE`, `CMD`) | §5.1.8 | 15 % | `Dockerfile` | `docker build` | `[ ]` |
| `.dockerignore` | §5.1.8 | | `.dockerignore` | prueba de contexto | `[ ]` |
| `POST /predict` con Pydantic | §5.1.8 | | `api/` | prueba de contrato | `[ ]` |
| `GET /health` (modelo y fecha) | §5.1.8 | | `api/app.py` | curl | `[ ]` |
| Modelo cargado al arrancar | §5.1.8 | | `api/app.py` | logs | `[ ]` |
| Accesible desde otra máquina | §8.3 | | — | captura del celular | `[ ]` |
| README completo y reproducible | §5.1.9 | 5 % | `README.md` | clon limpio | `[ ]` |

## Extras (P3)

| Extra | Puntos | Estado |
|---|---|---|
| GitHub Actions (ruff + pytest en cada PR) | +1 | `[~]` `ci.yml` creado, sin ejecutar aún |
| Pandera | +2 | `[ ]` |
| MLflow | +2 | `[ ]` |
| DVC con remoto funcional | +2 | `[ ]` |
| Interfaz web | +1 | `[ ]` (P2) |

## Seguridad

| Verificación | Cómo | Resultado | Fecha |
|---|---|---|---|
| `.gitignore` creado antes de `git init` y de `git add` | orden de comandos E1 | `[x]` | 17-oct |
| Raíz del repo correcta | `git rev-parse --show-toplevel` | `[ ]` se verifica de nuevo en `…/ProyectoFinal/banknextaction-ai-platform` (D-013) | |
| Baseline de secretos | `detect-secrets scan` | `[x]` 0 hallazgos | 17-oct |
| Reglas de exclusión probadas | `git check-ignore` en 17 rutas (10 excluidas, 7 permitidas) | `[x]` | 17-oct |
| `detect-private-key` rechaza llave falsa | `pre-commit run --files` con archivo ficticio, borrado | `[x]` Failed como se esperaba | 17-oct |
| `detect-secrets` rechaza clave AWS falsa | mismo archivo ficticio | `[x]` Failed como se esperaba | 17-oct |
| Conventional Commits en `commit-msg` | hook instalado; "arreglos" falla, "chore: …" pasa | `[x]` a nivel de hook | 17-oct |
| Control de tamaño (general, 500 KB) | archivo ficticio de 600 KB | `[x]` Failed "exceeds 500 KB" | 17-oct |
| Lista permitida en `models/` (D-009) | `models/zz_unauthorized.bin` | `[x]` Failed; `.gitkeep` aceptado | 17-oct |
| Staging limpio | 31 archivos, ninguno académico, crudo ni credencial; 0 hallazgos de detect-secrets | `[x]` | 17-oct |
| Auditoría previa al primer push | se repite justo antes del push | `[ ]` | |

## Datos
_Se completa en E2: fuente, licencia, SHA-256, esquema, cortes y commit de código._
