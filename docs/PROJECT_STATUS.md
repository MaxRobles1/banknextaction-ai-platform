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

## E2 · Datos y trazabilidad — ⏳
- [ ] `scripts/fetch_data.py` con verificación de SHA-256.
- [ ] `data/manifests/bank_full.json` con fuente, licencia, esquema, hash, cortes y commit.
- [ ] EDA y figura 1.
- [ ] Splits cronológicos con los índices registrados.
- [ ] Prueba de carga de datos.

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
