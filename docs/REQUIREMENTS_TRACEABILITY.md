# Matriz de trazabilidad de requisitos

**Fuente:** `Proyecto_Final_Modulo1.pdf` (§5.1 núcleo, §5.2 extras, §7 rúbrica, §10 checklist).

**Estados:**
- `[x]` con evidencia verificable;
- `[~]` implementado y verificado en local, pendiente de publicación o de una acción en GitHub;
- `[ ]` pendiente.

**Última actualización:** 2026-10-18 (tarde).

## Núcleo obligatorio (100 %)

| Requisito | Fuente | Peso | Archivo(s) | Evidencia | Estado |
|---|---|---|---|---|---|
| src-layout (`src/`, `tests/`, `notebooks/`, `configs/`, `reports/`, `models/`) | §5.1.1 | 10 % | `src/banknextaction/` | árbol del repo; paquete instalable con hatchling | `[x]` |
| Type hints en funciones públicas | §5.1.1 | | `src/` | firmas con tipos de parámetros y retorno; ruff | `[x]` |
| Decorador con propósito | §5.1.1 | | `utils/timing.py` (`@timed`) | mide la latencia del entrenamiento y de la inferencia; `inference_time_ms` real en `/predict` | `[x]` |
| dataclass / NamedTuple | §5.1.1 | | `config.py`, `data/split.py`, `evaluation/metrics.py`, `training/loop.py` | configuración congelada, `SplitBoundaries`, `MetricReport`, `History`, `EarlyStopping` | `[x]` |
| Context manager | §5.1.1 | | `utils/io.py` (`atomic_write`) | `test_atomic_write_keeps_previous_file_when_writing_fails` | `[x]` |
| Configuración separada (YAML / variables de entorno) | §5.1.1 | | `configs/config.yaml`, `BNA_CONFIG_PATH` | `test_split_config_rejects_fractions…`; falla al cargar si algo no cuadra | `[x]` |
| `.env.example` sin secretos; `.env` ignorado | §5.1.1 | | `.env.example`, `.gitignore` | `git check-ignore`; solo nombres de variables | `[x]` |
| uv + `pyproject` + lockfile | §5.1.2 | 10 % | `pyproject.toml`, `uv.lock` | `uv sync --locked` en un clon limpio | `[x]` |
| ruff configurado en `pyproject` | §5.1.2 | | `[tool.ruff]` (incluye NPY y T20) | `ruff check .` y `ruff format --check .` en verde | `[x]` |
| pre-commit (ruff lint/format, trailing, EOF, check-yaml) y más | §5.1.2 | | `.pre-commit-config.yaml` | `pre-commit run --all-files` en verde; pruebas negativas (D-009 a D-011) | `[x]` |
| logging en `src/` (sin `print`) | §5.1.2 | | `utils/logging.py`, `get_logger(__name__)` | regla ruff T20; logs sin datos del cliente | `[x]` |
| ≥4 pruebas: carga, transformación, salida del modelo, contrato de la API | §5.1.2 | | `tests/test_data.py`, `test_features_and_baseline.py`, `test_torch_model.py`, `test_api.py` | **33 pruebas** en verde (local y en un clon limpio) | `[x]` |
| Clase abstracta | §5.1.3 | 10 % | `models/base.py` (`BaseModel`) | implementada por `LogisticBaseline` y `MlpModel` | `[x]` |
| Separación de responsabilidades | §5.1.3 | | `data/`, `features/`, `models/`, `training/`, `evaluation/`, `api/` | la carga no entrena; el entrenamiento no conoce la API | `[x]` |
| Patrón de diseño documentado | §5.1.3 | | `models/registry.py`; README "Arquitectura y decisiones" | Strategy + Factory (registro), en dos renglones | `[x]` |
| NumPy vectorizado; pandas con dtypes y faltantes tratados | §5.1.4 | 10 % | `features/preprocess.py`, `evaluation/metrics.py`, `data/load.py` | F1 por umbral con matriz booleana; dtypes explícitos; `unknown` y `pdays=-1` tratados a propósito; `.loc[...].copy()` | `[x]` |
| 3 gráficas (EDA, curvas, resultados) | §5.1.4 | | `reports/figures/01_eda.png`, `02_learning_curves.png`, `03_test_results.png` | en el README, con título, ejes y leyendas | `[x]` |
| Pipeline + ColumnTransformer (sin escalar por fuera) | §5.1.5 | 10 % | `models/sklearn_model.py` | `test_preprocessor_…fit_on_train_only` | `[x]` |
| Validación cruzada temporal (`TimeSeriesSplit`) | §5.1.5 | | `models/sklearn_model.py` | `TimeSeriesSplit(5)` solo en train | `[x]` |
| GridSearch con ≥2 hiperparámetros, documentado | §5.1.5 | | `configs/config.yaml`; README | `C` × `class_weight` (8 combinaciones); ganó `C=0.01`, `None` | `[x]` |
| ≥2 métricas, iguales para ambos modelos | §5.1.5 | | `evaluation/metrics.py` | ROC-AUC, PR-AUC, F1, precisión, recall y accuracy | `[x]` |
| `nn.Module` con `forward` explícito | §5.1.6 | 20 % | `models/torch_model.py` (`PropensityMLP`) | `test_mlp_forward_returns_one_logit_per_row` | `[x]` |
| `Dataset` propio y `DataLoader` | §5.1.6 | | `TabularDataset` | `test_dataset_returns_float32_pairs` | `[x]` |
| Tres montones por corte; prueba abierta una vez | §5.1.6 | | `data/split.py`, `scripts/evaluate.py` | cortes en el manifiesto; `model_selection.json` escrito antes de predecir sobre la prueba | `[x]` |
| Bucle de dos mitades (`train` / `eval` + `no_grad`) | §5.1.6 | | `training/loop.py` | `train_one_epoch` y `evaluate_epoch` | `[x]` |
| Historia por época (pérdidas y métrica) | §5.1.6 | | `reports/mlp_validation.json` | 97 épocas con train_loss, val_loss, val PR-AUC y val ROC-AUC | `[x]` |
| Curvas con diagnóstico escrito | §5.1.6 | | README; D-015 | se bajó la tasa (validación "brincaba"); al final, brecha estable sin tijeras | `[x]` |
| Paro temprano **con restauración** | §5.1.6 | | `training/loop.py` | paro en la época 97 y restauración de la 87; `test_early_stopping_restores_best_weights` | `[x]` |
| Checkpoint completo y cargable sin reentrenar | §5.1.6 | | `models/mlp_checkpoint.pt` | `test_checkpoint_is_complete_and_resumable` | `[x]` |
| Métricas con su piso | §5.1.6 | | `reports/model_comparison.md` | piso de accuracy y de PR-AUC por partición | `[x]` |
| Umbral elegido en validación y aplicado igual en prueba | §5.1.6 | | `evaluation/metrics.py`, `models/modelo.json` | máximo F1 en val (0.555 en el MLP); la API usa el mismo umbral | `[x]` |
| Tabla comparativa con conclusión | §5.1.6 | | README, `reports/model_comparison.md` | MLP servido; conclusión en dos renglones | `[x]` |
| `main` protegida, trabajo por PR | §5.1.7 | 10 % | GitHub; `reports/flujo/` | ruleset activo (API) y push directo rechazado con GH013 | `[x]` |
| Conventional commits | §5.1.7 | | historial de `main` | un commit por PR con título convencional (#2 a #14) | `[x]` |
| PRs con descripción | §5.1.7 | | GitHub | PRs #2–#14 con qué cambia, por qué, cómo se verificó y `Closes #n` | `[x]` |
| Hooks corriendo | §5.1.7 | | `.git/hooks` | todos los commits pasaron hooks; ninguno con `--no-verify` | `[x]` |
| `.gitignore` útil; modelo serializado versionado | §5.1.7 | | `.gitignore`, `models/` | datos crudos, `.env` y `.venv` fuera; 5 artefactos de menos de 100 MB dentro | `[x]` |
| Dockerfile (slim, lockfile, modelo adentro, `EXPOSE`, `CMD`) | §5.1.8 | 15 % | `Dockerfile` | `python:3.11-slim`, `uv sync --locked`, sin caché, usuario sin privilegios | `[x]` |
| `.dockerignore` | §5.1.8 | | `.dockerignore` | `/app` contiene solo código, configuración y modelos | `[x]` |
| Build con un comando y sin pasos manuales | §8.1 | | — | `docker build -t banknextaction .` desde un clon limpio | `[x]` |
| `POST /predict` con Pydantic y probabilidad | §5.1.8 | | `api/` | 200; 422 por rango, categoría, campo extra o campo faltante | `[x]` |
| `GET /health` con modelo y fecha | §5.1.8 | | `api/service.py` | `mlp`, v0.1.0, `artifact_created_at`, MD5 igual al del recibo | `[x]` |
| Modelo cargado al arrancar (una vez) | §5.1.8 | | `api/app.py` (`lifespan`) | "model loaded once" ×1 en los logs del contenedor | `[x]` |
| Accesible desde otra máquina | §8.3 | | — | `http://<IP-local>:8000/health` → 200 desde la misma Mac por IP de red | `[~]` falta probar desde el celular |
| README completo y reproducible | §5.1.9 | 5 % | `README.md` | comandos ejecutados tal cual en un clon limpio, con métricas idénticas | `[x]` |

## Extras (§5.2, hasta +10)

| Extra | Puntos | Estado |
|---|---|---|
| GitHub Actions (ruff + pytest en cada PR) | +1 | `[x]` check `calidad` en verde en todos los PRs y requerido en `main` |
| Pandera | +2 | `[ ]` diferido (P3) |
| MLflow | +2 | `[ ]` diferido (P3) |
| DVC con remoto funcional | +2 | `[ ]` diferido (P3) |
| Interfaz web | +1 | `[ ]` diferido (P2) |

## Seguridad

| Verificación | Cómo | Resultado | Fecha |
|---|---|---|---|
| `.gitignore` antes de `git init` y de `git add` | orden de comandos | `[x]` | 17-oct |
| Raíz del repo correcta | `git rev-parse --show-toplevel` | `[x]` `…/ProyectoFinal/banknextaction-ai-platform`, sin historial heredado | 17-oct |
| Reglas de exclusión | `git check-ignore` | `[x]` incluye `.claude/` | 17-oct |
| Llave privada y secretos falsos rechazados | pruebas controladas | `[x]` | 17-oct |
| detect-secrets acotado a sumas de verificación | `--exclude-lines` solo para claves `*sha256`/`*md5` hex | `[x]` un JSON con credenciales falsas sigue fallando | 17-oct |
| Control de tamaño y lista permitida en `models/` | pruebas controladas | `[x]` | 17-oct |
| Autoría | `git log --format='%ae %ce'` | `[x]` 10 de 10 commits con el noreply de `MaxRobles1`; 0 trailers de IA | 18-oct |
| Integridad de artefactos | `verify_receipt` y `/health` | `[x]` SHA-256 y MD5 coinciden; la API no arranca si no coinciden | 18-oct |
| Auditoría antes del primer push | historial completo (578 blobs) | `[x]` 0 secretos, 0 rutas prohibidas, 0 correos personales | 18-oct |

## Datos

| Elemento | Evidencia |
|---|---|
| Fuente y licencia | UCI Bank Marketing, CC BY 4.0, DOI 10.24432/C5K306 (`data/manifests/bank_full.json`) |
| Integridad | SHA-256 `d1513ec6…`, verificado en cada corrida; la descarga falla si no coincide |
| Esquema | 45,211 × 17, dtypes registrados, sin nulos |
| Cortes | [0, 31647) / [31647, 38429) / [38429, 45211); sin solape; orden idéntico al CSV original |
| Reproducibilidad | `data_code_sha256` y `data_config_sha256`; la prueba falla si el manifiesto queda desactualizado |
| Cambio de distribución | D-014: prevalencia de 5.8 % / 11.5 % / 39.3 % |
