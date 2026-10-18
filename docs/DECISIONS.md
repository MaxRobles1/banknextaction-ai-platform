# Registro de decisiones (ADR ligero)

Formato: **ID · fecha · decisión** — contexto, alternativas, motivo, responsable.

---

### D-001 · 2026-10-17 · Proyecto: propensión sobre UCI Bank Marketing
- **Contexto:** el PDF permite problema libre con etiqueta (§3). Se busca una pieza de portafolio de banca comercial.
- **Decisión:** modelo de propensión supervisada sobre `bank-full.csv` (`y` = contrató depósito a plazo).
- **Alternativas:** mantenimiento predictivo (AI4I), demanda (Bike Sharing), churn.
- **Motivo:** tabular, etiquetado, columnas mixtas y orden cronológico, lo que obliga a `TimeSeriesSplit` (§5.1.5).
- **Límite explícito:** es propensión, **no** inferencia causal ni "next best action" probada.
- **Responsable:** Maximino J. R.

### D-002 · 2026-10-17 · Solo variables precontacto
- **Decisión:** se usan `age, job, marital, education, default, balance, housing, loan, pdays, previous, poutcome`.
- **Excluidas:**
  - `duration`: fuga directa; la propia UCI lo advierte.
  - `campaign`, `contact`, `day`, `month`: se conocen durante la campaña, no antes.
- **Motivo:** el modelo debe usar solo información disponible en el instante de decisión.

### D-003 · 2026-10-17 · Partición cronológica 70/15/15
- **Decisión:** cortes por orden de filas (la fuente está ordenada por fecha). La prueba queda sellada hasta la evaluación final. `TimeSeriesSplit` solo dentro de entrenamiento.
- **Alternativa descartada:** partición aleatoria, porque le hace al modelo una pregunta distinta a la de producción (PDF §5.1.5).

### D-004 · 2026-10-17 · Umbral: máximo F1 en validación
- **Decisión:** un umbral por modelo, el que maximiza F1 en validación, sobre una rejilla de 0.005 a 0.995.
- **Desempate determinista:** entre umbrales con el mismo F1, se toma el **mayor** (más precisión).
- **Aplicación:** el umbral se guarda en el recibo y se aplica sin cambios en prueba y en la API.
- **Motivo:** no requiere costos de negocio inventados.
- **Separación:** la priorización comercial **Top-k%** (Campaign Planner) es una funcionalidad independiente que ordena por probabilidad y no modifica el umbral.

### D-005 · 2026-10-17 · Selección del modelo servido
- **Criterios, en orden:** (1) F1 en validación, (2) PR-AUC en validación, (3) latencia medida, (4) tamaño de artefactos.
- **Restricción:** nunca se usa el conjunto de prueba para seleccionar.

### D-006 · 2026-10-17 · Herramientas
- **uv** con `uv.lock` (instalado en `~/.local/bin`).
- Python 3.11, igual al de la imagen `python:3.11-slim`.
- Torch de CPU en Linux vía índice `pytorch-cpu`.
- ruff, pytest, pre-commit, detect-secrets y nbstripout como dependencias de desarrollo; la versión de ruff es la misma en el proyecto y en el hook.

### D-007 · 2026-10-17 · Seguridad preventiva
- **`.gitignore`** antes de cualquier `git add`.
- **Hooks:** `detect-private-key`, `detect-secrets` (con baseline) y control de tamaño en dos instancias (general de 500 KB; `models/` hasta 100 MB).
- **Auditoría** antes del primer push, que requiere aprobación del autor.

### D-009 · 2026-10-17 · Lista explícita de artefactos permitidos en `models/`
- **Decisión:** un hook local `models-allowlist` (`language: fail`) rechaza todo archivo en `models/` que no sea uno de estos:
  - `baseline_pipeline.joblib`
  - `torch_preprocessor.joblib`
  - `mlp_state.pt`
  - `mlp_checkpoint.pt`
  - `modelo.json`
  - `.gitkeep`
- **Motivo:** el control por tamaño (máximo 100 MB) no basta para garantizar "solo artefactos autorizados". La lista se amplía únicamente mediante una decisión nueva en este archivo.
- **Responsable:** Maximino J. R. (autorizado el 17-oct).

### D-010 · 2026-10-17 · `check-added-large-files` con `--enforce-all`
- **Decisión:** las dos instancias del control de tamaño revisan todo archivo que reciben, no solo los recién agregados.
- **Motivo:**
  - Sin la opción, el hook ignora archivos que no están en staging, así que la prueba controlada no demostraba nada.
  - En un commit normal el hook recibe exactamente los archivos del staging, así que el comportamiento no cambia.
- **Hook de ruff:** se actualizó de `ruff` (alias antiguo) a `ruff-check`, en v0.17.1, la misma versión del proyecto.

### D-011 · 2026-10-17 · Convenciones de ingeniería verificadas en los materiales del curso
Revisadas en modo lectura: PDF del proyecto, apuntes y cuadernos de las Clases 2, 3 y 11 (y Tarea 11). No se copió material al repositorio.

**Documentadas en los materiales (se aplican):**

| Convención | Fuente | Aplicación en el repo |
|---|---|---|
| Formato `tipo(alcance opcional): descripción`; tipos `feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert`; `!` = cambio que rompe | Apuntes Clase 11 §3.1; PDF §5.1.7 (subconjunto) | hooks `conventional-pre-commit` y `commit-msg-class-rule` |
| Regla exacta: tipo en minúsculas, alcance `[a-z0-9-]` en minúsculas, `: ` y descripción no vacía. El hook estándar acepta `Fix:`; la regla del curso no | Apuntes Clase 11 §3.2 (C3A) | hook local `commit-msg-class-rule` (pygrep). Probado: rechaza `Fix:`, `feat:sin-espacio`, `arreglos` y `wip` |
| Un hook local se salta con `--no-verify`; la barrera real está en el servidor (PR, check y ruleset) | Apuntes Clase 11 §3.3 y §4 | E1b: ruleset en `main` con el check `calidad` |
| Una rama por issue con nombre `tipo/descripción` (`feat/esquema`); un PR por rama; plantilla llena con `Closes #n` | Apuntes Clase 11 §4.2 y §4.5; Tarea 11 parte 3 | Desde E1b |
| **Squash and merge**: un commit por PR, con el título del PR como mensaje. El título debe seguir la convención | Apuntes Clase 11 §3.4; Tarea 11 parte 2 | Configuración de GitHub (E1b) |
| Revisión en proyecto individual: leer el propio diff en *Files changed*; el check revisa | Apuntes Clase 11 §4.5 | Ritual previo a cada merge |
| `.gitignore` y hooks entran por la rama `ci/flujo-de-trabajo` junto con CI y plantilla | Tarea 11 partes 1–2 | `.github/` entra por ese PR; ver la excepción abajo |
| `logging.getLogger(__name__)` en cada módulo; nunca `print` ni `logging.info()` directo; nunca llaves, tokens ni datos personales en logs | Apuntes Clase 2 §4.1–4.4 | Regla ruff `T20` y revisión |
| ruff con los grupos E/F, I, UP, B, SIM y **NPY** | Apuntes Clase 2 §5.2 | `NPY` agregado al `select` |
| Lockfile versionado; uv o Poetry | Apuntes Clase 2 §3; PDF §5.1.2 | `uv.lock` |
| Configuración que **falla al arrancar** si algo no cuadra | Apuntes Clase 2 §2 (usa pydantic-settings) | El PDF §5.1.1 pide YAML + dataclasses; se validará al cargar (E2) |
| pytest con fixtures (`conftest.py`), `parametrize` y `tmp_path`; primero la prueba que reproduce el error | Apuntes Clase 2 §6 | Pruebas desde E2 |
| Responsabilidad única, abierto/cerrado e inversión de dependencias; contrato con clase abstracta; composición sobre herencia; registro + fábrica desde la configuración | Apuntes Clase 3 §0.2, §2–§3 | `BaseModel` (ABC), `registry.py` (registro + fábrica = Strategy/Factory), funciones que reciben el contrato |

**No especificadas en los materiales (se adopta práctica estándar):**
- **Idioma de los mensajes:** los ejemplos de clase están en español, pero no hay una regla. Se usa **inglés en modo imperativo** (`add`, `fix`), coherente con el código en inglés y con el primer commit autorizado.
- **Alcances (scopes) del proyecto:** no están definidos en clase. Se proponen `data`, `features`, `models`, `training`, `eval`, `api`, `docker`, `web`, `docs` y `ci`.
- **Tamaño de commit:** la clase no lo fija y aclara que con squash los commits de la rama "pueden ser desordenados". Se adopta la práctica de commits atómicos: un cambio coherente que pasa los hooks y las pruebas por sí solo.
- **Docstrings:** no hay un estilo prescrito. Se usa una línea de resumen en las funciones y clases públicas.

**Contradicciones o tensiones detectadas:**
- **PDF §5.1.7 ("nada entra por commit directo") frente a la creación del repo:** el primer commit `chore: bootstrap project` va directo a `main`, porque un PR necesita una rama base existente. Se documenta como **excepción técnica única**; después se activa el ruleset.
- **Tarea 11 (hooks y `.gitignore` por PR) frente a la política de seguridad del autor (D-007):** `.gitignore` y hooks entran en el bootstrap porque deben existir **antes** del primer commit. CI y plantilla (`.github/`) sí entran por el PR `ci/flujo-de-trabajo`.

### D-012 · 2026-10-17 · Privacidad del autor: correo noreply de GitHub
- **Contexto:** la auditoría previa al push encontró el correo personal en los metadatos de *author* y *committer*. La cuenta tiene activados "Keep my email addresses private" y "Block command line pushes that expose my email".
- **Decisión (opción A, elegida por el autor):**
  - `git config --local user.email` con la dirección noreply de la cuenta `MaxRobles1`. La configuración global no cambia.
  - Se reescribieron los 2 commits **no publicados** con `commit --amend --reset-author`; `ci/flujo-de-trabajo` se reaplicó con `rebase --onto` sobre el nuevo `main`.
- **Verificación:**
  - Árboles idénticos antes y después (`cb1886b`, `f360b80`).
  - Mensajes idénticos.
  - 4 de 4 correos de author y committer son noreply.
  - Las ramas publicables no alcanzan los commits viejos.
  - El ID de usuario (API pública) coincide con el de la dirección noreply.
- **Respaldo:** `refs/backup/*`, solo local y nunca publicado. Se elimina, con autorización, después de verificar el push.

### D-013 · 2026-10-17 · Repositorio definitivo con autoría exclusiva
- **Contexto:** el primer repositorio (`MaxRobles1/banknextaction-ai`) publicó 2 commits con el trailer automático `Co-Authored-By` de Claude Code, y GitHub mostraba a `claude` como contributor.
- **Decisión del autor:** nuevo repositorio `MaxRobles1/banknextaction-ai-platform`, en la carpeta independiente `banknextaction-ai-platform/`, con historial nuevo desde `git init -b main`. `MaxRobles1` es el único author y committer.
- **Atribución de IA:**
  - Claude Code se reconoce como herramienta de asistencia solo en la documentación (README, sección "Herramientas utilizadas").
  - Los commits y PR no llevan trailers ni líneas de atribución.
  - La atribución automática se desactivó en `.claude/settings.local.json`, una configuración local e ignorada por Git; la configuración global no se tocó.
- **Conservación:**
  - Respaldo verificado del repo anterior (bundle con todas las refs y árbol de trabajo, con SHA-256) en `_respaldos/`, fuera de cualquier repo.
  - Transferencia selectiva de 31 archivos.
  - No se perdió funcionalidad: no había trabajo de E2.
- **Repo anterior:** lo eliminará manualmente el autor, solo después de verificar el nuevo. Pueden persistir copias o cachés externos que no controlamos.
- **Efecto sobre D-012:** la reescritura de autor que describe D-012 aplicó al repo anterior. Aquí la identidad noreply se configura desde el inicio.

### D-014 · 2026-10-17 · Cambio de distribución en el tiempo (limitación y riesgo de producción)
- **Evidencia medida:** el archivo está ordenado por fecha (mayo 2008 – noviembre 2010; se verificaron 2 saltos de año por el mes). La prevalencia de `y = yes` es:

  | Partición | Filas | Positivos | Negativos | Prevalencia |
  |---|---|---|---|---|
  | train | [0, 31647) | 1,840 | 29,807 | 5.81 % |
  | val | [31647, 38429) | 781 | 6,001 | 11.52 % |
  | test | [38429, 45211) | 2,668 | 4,114 | 39.34 % |

- **Decisión:** mantener la partición cronológica (PDF §5.1.5). Se reportan prevalencia y piso **por partición**, y la prueba se abre una sola vez al final.
- **Implicaciones, sin anticipar resultados:**
  - El umbral elegido en validación se aplica sin cambios a un periodo con otra prevalencia. Precisión, recall y F1 en prueba pueden moverse en cualquier dirección; se reportará lo que se mida.
  - ROC-AUC no depende directamente de la prevalencia, porque mide el ordenamiento entre clases. Aun así puede cambiar entre periodos si cambian las distribuciones de puntuaciones de cada clase, por ejemplo si cambia el perfil de los clientes contactados.
  - PR-AUC sí depende de la prevalencia; su piso es la prevalencia de cada partición.
- **Riesgo de producción:** un modelo de propensión entrenado en un periodo puede descalibrarse cuando cambian la estrategia de campaña o el contexto económico. Mitigación posterior (roadmap): monitoreo de drift, recalibración periódica y variables de contexto (por ejemplo, las macroeconómicas de `bank-additional-full.csv`).
- **Falsos positivos de detect-secrets:** las sumas SHA-256 públicas, que son cadenas hexadecimales de alta entropía, se marcaron sin desactivar el hook:
  - en el YAML, con `# pragma: allowlist secret (public checksum)` en la línea;
  - en el manifiesto JSON, que no admite comentarios, como entradas auditadas (`is_secret: false`) en `.secrets.baseline`.
  - Cualquier otro hallazgo sigue bloqueando el commit.
- **Trazabilidad del manifiesto:** en lugar de un id de commit, que sería circular, el manifiesto guarda `data_code_sha256` (código del paso de datos) y `data_config_sha256` (secciones `data`, `features` y `split`). La prueba `test_committed_manifest_matches_current_data_code_and_config` falla si el manifiesto queda desactualizado.

### D-015 · 2026-10-17 · MLP: tasa de aprendizaje y diagnóstico de curvas (Clase 9 §4.2)
- **Arquitectura:**
  - capas `Linear(31→64) → ReLU → Dropout(0.2) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)`;
  - pérdida `BCEWithLogitsLoss` con `pos_weight` = negativos / positivos de **train**;
  - optimizador Adam con `weight_decay` 1e-4 y lotes de 256;
  - paro temprano sobre **val PR-AUC** con paciencia 10, restauración de los mejores pesos y checkpoint escrito en la mejor época.
- **Primera corrida (lr 1e-3):** la pérdida de validación **brincaba** (salto medio entre épocas de 0.058, rango 1.74–1.92) mientras la de entrenamiento bajaba suave. Es la cuarta forma de la Clase 9, "ninguna baja, o brincan", cuyo remedio es bajar la tasa 10 veces antes de tocar la arquitectura.
- **Experimento, decidido solo con validación:**

  | lr | mejor/paro | salto medio val_loss | val PR-AUC | val ROC-AUC | val F1 |
  |---|---|---|---|---|---|
  | 1e-3 | 13/23 | 0.058 | 0.2564 | 0.6761 | 0.3145 |
  | 1e-4 | 87/97 | 0.006 | 0.2577 | 0.6781 | 0.3115 |

- **Decisión:** `lr = 1e-4` (validación 9 veces más estable e igual o mejor PR-AUC; la diferencia de F1 es ruido) y `max_epochs = 200`, para que el paro temprano, no el tope, termine el entrenamiento.
- **Diagnóstico final:** las dos pérdidas bajan juntas y luego se aplanan, con una brecha **estable** que no crece. No hay tijeras, es decir, no memoriza.
  - La brecha de nivel se explica en buena parte por la ponderación: `pos_weight` se calcula con la prevalencia de train (5.8 %) y validación tiene el doble de positivos, así que su pérdida ponderada es mayor por construcción.
  - Según la tabla de la Clase 9, mejorar ahora sería "por el lado de los datos": más señal disponible antes del contacto. No compensa añadir más red.
- **Nota:** por `pos_weight`, las probabilidades del MLP no están calibradas (están desplazadas hacia arriba). Por eso su umbral óptimo de F1 ronda 0.55 y el del baseline 0.06. El umbral se elige y se guarda por modelo, y la API usa el del modelo servido.

### D-008 · 2026-10-17 · Roles de Git, DVC y MLflow
- **Git:** código, configuración, manifiestos, recibo y modelos de menos de 100 MB, incluido el **checkpoint completo** de PyTorch.
- **DVC (P3):** datos crudos con un remoto funcional aprobado.
- **MLflow (P3):** registro de experimentos.
- **Regla:** sin duplicar artefactos entre Git y DVC, y ninguno se declara implementado sin verificarlo.
