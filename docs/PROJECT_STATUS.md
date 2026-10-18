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
| E2 | Dataset, trazabilidad, EDA y particiones | ⏳ PENDIENTE | |
| E3 | Pipeline y modelo scikit-learn | ⏳ PENDIENTE | |
| E4 | Modelo PyTorch y entrenamiento | ⏳ PENDIENTE | |
| E5 | Evaluación, métricas y selección | ⏳ PENDIENTE | |
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

## E2 · Datos y trazabilidad — 🔄 (implementado y verificado en local; falta su commit/PR)
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
- [ ] Commit y PR `feat/data-pipeline` (requieren autorización).
- [x] Prueba de manifiesto desactualizado: 9 de 9 en verde.
- **Hallazgo documentado (D-014):** fuerte cambio de prevalencia en el tiempo. El piso se reporta por partición. ROC-AUC no depende directamente de la prevalencia, pero puede cambiar si cambian las distribuciones de puntuaciones por clase.

## E3–E11 — ⏳
Los criterios de aceptación de cada etapa están en el plan aprobado y se desglosarán aquí al iniciarla.

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
