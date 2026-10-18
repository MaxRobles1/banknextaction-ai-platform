# Estado del proyecto — BankNextAction AI

**Repositorio:** `MaxRobles1/banknextaction-ai-platform` · carpeta local `~/Desktop/ProyectoFinal/banknextaction-ai-platform/`
**Última actualización:** 2026-10-17 (sábado), noche
**Etapa actual:** E1 — Bootstrap seguro (repositorio nuevo, D-013)
**Entrega:** el repositorio se evalúa al cierre del **domingo 18-oct-2026**. Demo: lunes 19-oct-2026.

**Estados:** ✅ completada (verificada) · 🔄 en progreso · ⏳ pendiente · ⚠️ bloqueada · ❌ falló · ➖ diferida
**Prioridades:** P1 núcleo ITAM → P2 web mínima → P3 extras → P4 producto.

## Checklist maestro

| Etapa | Entregable | Estado | Evidencia o pendiente |
|---|---|---|---|
| E0 | Verificación del entorno | ✅ COMPLETADA | Git 2.50.1, Docker 29.7.2, uv 0.13.1 |
| E1 | Bootstrap seguro, Git y pre-commit | 🔄 EN PROGRESO | Repo nuevo con historial limpio; falta autorizar el primer commit y el push |
| E1b | GitHub, protección de main y CI | ⏳ PENDIENTE | Repo nuevo, PR de CI, check `calidad`, ruleset |
| E2 | Dataset, trazabilidad, EDA y particiones | ✅ COMPLETADA | Commit `29d6058` (local); SHA-256, manifiesto, splits y 8 verificaciones |
| E3 | Pipeline y modelo scikit-learn | ✅ COMPLETADA | Commit local en `feat/baseline-sklearn`; GridSearchCV + TimeSeriesSplit; 16 pruebas |
| E4 | Modelo PyTorch y entrenamiento | ✅ COMPLETADA | Commit local en `feat/torch-mlp`; paro en la época 97, se restaura la 87; checkpoint reanudable; 21 pruebas |
| E5 | Evaluación, métricas y selección | ✅ COMPLETADA | Se sirve el MLP (elegido con validación); prueba abierta una vez; recibo verificado; 24 pruebas |
| E6 | FastAPI y pruebas | ⏳ PENDIENTE | |
| E7 | Docker reproducible | ⏳ PENDIENTE | |
| E8 | README, documentación y trazabilidad | ⏳ PENDIENTE | |
| E9 | Aplicación web mínima | ⏳ PENDIENTE | P2 |
| E10 | Extras: Pandera, MLflow, DVC y otros | ⏳ PENDIENTE | P3 |
| E11 | Auditoría final, presentación y cierre | ⏳ PENDIENTE | |

---

## E0 · Verificación del entorno — ✅
- [x] Git 2.50.1 con identidad configurada; Docker 29.7.2 con el daemon activo (verificado con `docker run hello-world`).
- [x] uv 0.13.1 en `~/.local/bin`.

## E1 · Bootstrap seguro — 🔄

**Conservado del proyecto anterior (D-013):**
- [x] Respaldo verificado fuera de cualquier repo: `_respaldos/banknextaction-ai_2026-10-17/`, con bundle de Git (8 refs), árbol de trabajo y `SHA256SUMS` OK.
- [x] 31 archivos transferidos selectivamente: estructura src-layout, `pyproject.toml`, `uv.lock`, hooks, `.gitignore`, configuración YAML, CI, plantilla de PR, documentos y prueba de humo. Sin `.git`, `.venv`, `.env`, datos ni material académico.
- [x] Decisiones D-001 a D-012 conservadas en `docs/DECISIONS.md`.
- [x] No existía trabajo de E2: `data/`, `models/`, `checkpoints/` y `notebooks/` solo tenían `.gitkeep`.

**Repetido en el repo nuevo (verificado antes del primer commit):**
- [ ] `git init -b main` en `banknextaction-ai-platform/`, sin historial heredado.
- [ ] Identidad local: author y committer con la dirección noreply de `MaxRobles1`.
- [ ] Atribución automática de Claude Code desactivada en `.claude/settings.local.json` (privado, ignorado por Git).
- [ ] `uv sync --locked`, ruff, pytest y `pre-commit run --all-files` en verde.
- [ ] Hooks instalados (`pre-commit` y `commit-msg`) y pruebas negativas (mensaje y secreto falsos).
- [ ] Auditoría de staging: sin secretos, datos, material académico ni `.claude/`.
- [ ] ⚠️ **Autorización del autor** para el primer commit.
- [ ] ⚠️ **Autorización del autor** para el primer push público.

## E1b · GitHub y flujo — ⏳
- [ ] Repo público `MaxRobles1/banknextaction-ai-platform`, creado vacío en la web.
- [ ] Issue de CI, PR `ci/flujo-de-trabajo` con `Closes #n`, check `calidad` en verde y squash merge.
- [ ] Solo squash merge y borrado automático de ramas.
- [ ] Ruleset en `main`: PR obligatorio, check `calidad`, bloqueo de force push y de borrado.
- [ ] Push directo a `main` rechazado (GH013), con 2 capturas en `reports/flujo/`.
- [ ] Contributors: solo `MaxRobles1`.
- [ ] Repo anterior `MaxRobles1/banknextaction-ai` eliminado manualmente por el autor, después de verificar el nuevo.

## E2 · Datos y trazabilidad — ✅ (commit `29d6058` en `feat/data-pipeline`; PR pendiente de push)
- [x] `scripts/prepare_data.py`: descarga de UCI, extracción del zip anidado y verificación SHA-256 `d1513ec6…` (falla si no coincide). *Verificado:* descarga real y segunda corrida "already present and verified".
- [x] `data/manifests/bank_full.json`:
  - licencia CC BY 4.0, DOI 10.24432/C5K306, cita, fecha de obtención y esquema;
  - 45,211 filas y 17 columnas;
  - variables incluidas y excluidas, criterios de limpieza, cortes, semilla, SHA-256 de la configuración y huella `data_code_sha256`, que evita la dependencia circular con el id de commit (D-014).
- [x] Splits cronológicos [0, 31647) / [31647, 38429) / [38429, 45211), sin barajar. *Verificado:* sin solape y cubren las 45,211 filas; orden y valores idénticos al CSV original (comparación fila por fila); 2 saltos de año por el mes, compatible con el orden documentado.
- [x] Prevalencia y piso por conjunto, medidos:
  - train 5.81 % (piso 0.9419);
  - val 11.52 % (piso 0.8848);
  - test 39.34 % (piso 0.6066).
- [x] Figura 1 `reports/figures/01_eda.png`: tasa a lo largo del tiempo con los cortes, y tasa por `poutcome`.
- [x] Cuaderno de exploración `notebooks/01_eda.ipynb`: usa las funciones de `src/` y ejecuta sin errores.
- [x] Código del paso de datos:
  - `config.py`: dataclasses que validan al cargar;
  - `utils/`: logging, decorador `@timed`, hashes y el context manager `atomic_write`;
  - `data/`: `fetch`, `load` (dtypes explícitos), `split` y `manifest`.
- [x] 7 pruebas nuevas en `tests/test_data.py`, con datos sintéticos (no requieren el dataset real): 8 de 8 en verde. ruff limpio.
- [x] Commit `29d6058 feat(data): add verified download, manifest and chronological split`.
- [ ] PR (requiere push autorizado).
- [x] Prueba de manifiesto desactualizado: 9 de 9 en verde.
- **Hallazgo documentado (D-014):** fuerte cambio de prevalencia en el tiempo. El piso se reporta por partición. ROC-AUC no depende directamente de la prevalencia, pero puede cambiar si cambian las distribuciones de puntuaciones por clase.

## E3 · Baseline scikit-learn — ✅ (commit local en `feat/baseline-sklearn`)
- [x] `features/preprocess.py`, un `ColumnTransformer` que:
  - escala `age`;
  - aplica log con signo y escala a `balance` y `previous` (NumPy vectorizado);
  - convierte `pdays` en indicador de contacto previo más días transcurridos;
  - aplica one-hot a 7 categóricas, ignorando categorías nuevas.
- [x] `models/base.py`: `BaseModel` (ABC) con `from_config`, `fit`, `predict_proba`, `save`, `load` y `predict`.
- [x] `models/registry.py`: patrón **Strategy + Factory** (`@register_model` y `create_model`).
- [x] `models/sklearn_model.py`: `Pipeline(ColumnTransformer + LogisticRegression)`, con `GridSearchCV` sobre `C` × `class_weight` (8 combinaciones) y `TimeSeriesSplit(5)` **solo en train**; la preparación se reajusta en cada fold.
- [x] `evaluation/metrics.py`:
  - ROC-AUC, PR-AUC, F1, precisión, recall y accuracy, con pisos;
  - umbral de máximo F1 en validación, vectorizado, con desempate por el umbral mayor.
- [x] `scripts/train.py`: solo usa train y val; la prueba queda sellada.
- [x] Resultado medido (validación; no es el resultado final):
  - **Mejores hiperparámetros:** `C=0.01`, sin `class_weight`, con PR-AUC de CV de 0.0865 ± 0.032.
  - **Umbral:** 0.060.
  - **Métricas:** ROC-AUC 0.656, PR-AUC 0.233 (piso 0.115), F1 0.276, precisión 0.191, recall 0.496, accuracy 0.700 (piso 0.885; el umbral optimiza F1, no accuracy).
- [x] 7 pruebas nuevas:
  - preprocesamiento sin NaN, ajustado solo con train y tolerante a categorías nuevas;
  - salida `(n,)` en [0, 1] y recarga idéntica;
  - desempate del umbral;
  - F1 vectorizado igual al de `evaluate`.
  - En total, 16 de 16 en verde.
- Artefactos (`models/*.joblib`) y reportes JSON se versionan en E5, junto con el recibo final.

## E4 · Modelo PyTorch — ✅ (commit local en `feat/torch-mlp`)
- [x] `TabularDataset` (`__len__`, `__getitem__`, float32), `DataLoader` con semilla y `PropensityMLP(nn.Module)` con `forward` explícito.
- [x] `training/loop.py`:
  - dos mitades por época: `train()`, y luego `eval()` dentro de `no_grad()`;
  - `History` por época con pérdidas de train y val, val PR-AUC y val ROC-AUC;
  - `EarlyStopping` con paciencia y **restauración** de los mejores pesos;
  - checkpoint completo (pesos, optimizador, época, mejor métrica, arquitectura e historia) escrito en la **mejor** época.
- [x] El preprocesador se ajusta solo con train y se guarda junto a los pesos (`torch_preprocessor.joblib`, `mlp_state.pt`, `mlp_checkpoint.pt`).
- [x] Experimento con la tasa (D-015): con 1e-3 la validación brincaba; 1e-4 es 9 veces más estable. Diagnóstico: brecha estable, sin tijeras.
- [x] Resultado en validación: paro en la época 97, **restaurada la 87**; PR-AUC 0.2577, ROC-AUC 0.6781, F1 0.3115 con umbral 0.555.
- [x] Figura 2: `reports/figures/02_learning_curves.png`.
- [x] 5 pruebas nuevas:
  - forma de la salida `(n,)`;
  - Dataset float32;
  - restauración de los mejores pesos;
  - guardar y recargar con predicción idéntica;
  - checkpoint completo que se reanuda con el optimizador.
  - En total, 21 de 21 en verde.

## E5 · Evaluación y selección — ✅ (commit local en `feat/evaluation`)
- [x] `evaluation/selection.py`: criterios en orden (F1 val → PR-AUC val → latencia p50 → tamaño), con latencia de una fila **medida** (200 repeticiones y 20 de calentamiento).
- [x] `scripts/evaluate.py`:
  - escribe `reports/model_selection.json` **antes** de abrir la prueba;
  - evalúa la prueba una vez con el umbral de validación de cada modelo.
- [x] Tabla comparativa con pisos: `reports/model_comparison.md` (D-016). **Se sirve el MLP.**
  - Prueba: PR-AUC 0.509 (piso 0.393), ROC-AUC 0.636, F1 0.561.
  - Accuracy 0.589, bajo su piso de 0.607, documentado.
- [x] Figura 3: `reports/figures/03_test_results.png`, con curvas PR en prueba contra el piso y matriz de confusión del MLP en su umbral.
- [x] Recibo `models/modelo.json`:
  - modelo servido, umbral, regla y motivo;
  - por artefacto: SHA-256, MD5, tamaño y fecha;
  - métricas de val y test, latencias, esquema de entrada (orden, rangos y categorías), hash del dataset y del código de datos, y versiones.
  - `verify_receipt` no encuentra discrepancias.
- [x] detect-secrets acotado a las sumas de verificación JSON, con prueba negativa que sigue rechazando credenciales.
- [x] 3 pruebas nuevas (orden de selección y desempates; recibo que detecta un artefacto alterado). En total, 24 de 24 en verde.

## E6–E11 — ⏳
Los criterios de aceptación están en el plan aprobado.

---

## Problemas y bloqueos
- El primer commit y el primer push del repo nuevo esperan la autorización del autor.
- Riesgo de calendario: queda el domingo 18 para E2–E8. El orden de recorte ya acordado es E10 → E9.

## Decisiones pendientes
- Ninguna técnica.

## Próximos tres pasos
1. Primer commit del repo nuevo (autorización).
2. Crear el repo vacío en GitHub, hacer el push (autorización), PR de CI y ruleset.
3. E2: descarga verificada, manifiesto, EDA y splits.

## Riesgos abiertos
- Posible cambio de prevalencia de `y` en el tiempo, que afectaría piso y umbral. Se verifica en E2.
- Red del campus el día 19: probar acceso desde el celular en E7.
- detect-secrets es heurístico: un resultado limpio no garantiza ausencia total de secretos.

## Diferido para después del ITAM (P4)
- Batch Scoring.
- Campaign Planner (Top-k%, `precision@k`, `recall@k` y `lift@k`).
- Executive Dashboard.
- Model Governance.
- Despliegue público.
- A/B y uplift modeling.
