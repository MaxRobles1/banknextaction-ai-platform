// BankNextAction AI — scoring UI.
// The form is built from the API's own contract (/openapi.json, schema ClientFeatures):
// field types, ranges and allowed categories come from the backend, so the UI cannot drift
// from what /predict accepts. Only human-readable labels live here. No inference happens in
// the browser: every number shown comes from the /predict response.
"use strict";

const SCHEMA_NAME = "ClientFeatures";

const GROUPS = [
  { title: "Perfil", fields: ["age", "job", "marital", "education"] },
  { title: "Situación financiera", fields: ["balance", "default", "housing", "loan"] },
  { title: "Historial de campañas", fields: ["previous", "pdays", "poutcome"] },
];

const LABELS = {
  age: { label: "Edad", hint: "Años" },
  job: { label: "Ocupación" },
  marital: { label: "Estado civil" },
  education: { label: "Nivel educativo" },
  balance: { label: "Saldo promedio anual", hint: "Euros; puede ser negativo" },
  default: { label: "¿Crédito en incumplimiento?" },
  housing: { label: "¿Crédito hipotecario?" },
  loan: { label: "¿Préstamo personal?" },
  previous: { label: "Contactos en campañas anteriores", hint: "0 si nunca fue contactado" },
  pdays: { label: "Días desde el último contacto previo", hint: "-1 = nunca contactado" },
  poutcome: { label: "Resultado de la campaña anterior" },
};

const OPTION_LABELS = {
  job: {
    "admin.": "Administrativo", "blue-collar": "Operativo", entrepreneur: "Emprendedor",
    housemaid: "Trabajo doméstico", management: "Gerencial", retired: "Jubilado",
    "self-employed": "Independiente", services: "Servicios", student: "Estudiante",
    technician: "Técnico", unemployed: "Desempleado", unknown: "Sin dato",
  },
  marital: { divorced: "Divorciado(a)", married: "Casado(a)", single: "Soltero(a)" },
  education: { primary: "Primaria", secondary: "Secundaria", tertiary: "Superior", unknown: "Sin dato" },
  yesno: { no: "No", yes: "Sí" },
  poutcome: { failure: "No contrató", other: "Otro", success: "Contrató", unknown: "Sin campaña previa" },
};

// Synthetic profiles (no real people). The first is the API's documented example.
const EXAMPLES = [
  { age: 41, job: "management", marital: "married", education: "tertiary", default: "no",
    balance: 1500, housing: "yes", loan: "no", pdays: -1, previous: 0, poutcome: "unknown" },
  { age: 35, job: "student", marital: "single", education: "tertiary", default: "no",
    balance: 3000, housing: "no", loan: "no", pdays: 90, previous: 2, poutcome: "success" },
];

const $ = (id) => document.getElementById(id);
let schema = null;
let exampleIndex = 0;

function optionLabel(field, value) {
  const table = OPTION_LABELS[field] || (["default", "housing", "loan"].includes(field) ? OPTION_LABELS.yesno : {});
  return table[value] || value;
}

function buildField(name, spec) {
  const meta = LABELS[name] || { label: name };
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.dataset.field = name;

  const label = document.createElement("label");
  label.htmlFor = `f-${name}`;
  label.textContent = meta.label;
  wrapper.append(label);

  let control;
  if (Array.isArray(spec.enum)) {
    control = document.createElement("select");
    // defaultSelected = true so the native form reset returns to this placeholder.
    const placeholder = new Option("Selecciona…", "", true, true);
    placeholder.disabled = true;
    control.append(placeholder);
    for (const value of spec.enum) control.append(new Option(optionLabel(name, value), value));
  } else {
    control = document.createElement("input");
    control.type = "number";
    control.step = "1";
    control.inputMode = "numeric";
    if (spec.minimum !== undefined) control.min = String(spec.minimum);
    if (spec.maximum !== undefined) control.max = String(spec.maximum);
  }
  control.id = `f-${name}`;
  control.name = name;
  control.required = true;
  control.setAttribute("aria-describedby", `e-${name}`);
  wrapper.append(control);

  const hintText = [meta.hint, rangeText(spec)].filter(Boolean).join(" · ");
  if (hintText) {
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = hintText;
    wrapper.append(hint);
  }
  const error = document.createElement("span");
  error.className = "error";
  error.id = `e-${name}`;
  wrapper.append(error);
  return wrapper;
}

function rangeText(spec) {
  if (spec.minimum === undefined || spec.maximum === undefined) return "";
  const fmt = (n) => Number(n).toLocaleString("es-MX");
  return `Rango ${fmt(spec.minimum)} a ${fmt(spec.maximum)}`;
}

function renderForm(properties, required) {
  const container = $("form-fields");
  container.replaceChildren();
  const placed = new Set();
  for (const group of GROUPS) {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "group";
    const legend = document.createElement("legend");
    legend.textContent = group.title;
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const name of group.fields) {
      if (!properties[name]) continue;
      grid.append(buildField(name, properties[name]));
      placed.add(name);
    }
    fieldset.append(legend, grid);
    container.append(fieldset);
  }
  // Any field the API adds later still appears, so the contract is always complete.
  const extra = required.filter((name) => !placed.has(name));
  if (extra.length) {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "group";
    const legend = document.createElement("legend");
    legend.textContent = "Otros datos";
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const name of extra) grid.append(buildField(name, properties[name]));
    fieldset.append(legend, grid);
    container.append(fieldset);
  }
}

function setFieldError(name, message) {
  const field = document.querySelector(`.field[data-field="${CSS.escape(name)}"]`);
  if (!field) return false;
  field.classList.toggle("has-error", Boolean(message));
  field.querySelector(".error").textContent = message || "";
  return true;
}

function clearErrors() {
  document.querySelectorAll(".field").forEach((f) => setFieldError(f.dataset.field, ""));
  $("form-error").hidden = true;
}

// Client-side checks mirror the schema; the API remains the source of truth (422).
function collectPayload() {
  const payload = {};
  let valid = true;
  for (const [name, spec] of Object.entries(schema.properties)) {
    const control = $(`f-${name}`);
    const raw = control.value.trim();
    if (raw === "") {
      setFieldError(name, "Campo obligatorio.");
      valid = false;
      continue;
    }
    if (Array.isArray(spec.enum)) {
      payload[name] = raw;
      continue;
    }
    const value = Number(raw);
    if (!Number.isInteger(value)) {
      setFieldError(name, "Debe ser un número entero.");
      valid = false;
    } else if ((spec.minimum !== undefined && value < spec.minimum) || (spec.maximum !== undefined && value > spec.maximum)) {
      setFieldError(name, `Fuera de rango (${rangeText(spec).replace("Rango ", "")}).`);
      valid = false;
    } else {
      payload[name] = value;
    }
  }
  return valid ? payload : null;
}

function showState(state) {
  for (const id of ["state-empty", "state-loading", "state-error", "state-success"]) {
    $(id).hidden = id !== `state-${state}`;
  }
}

function setLoading(isLoading) {
  const button = $("submit-btn");
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  button.querySelector(".btn__label").textContent = isLoading ? "Calculando…" : "Calcular propensión";
}

function renderResult(result) {
  const pct = (x) => `${(x * 100).toFixed(1)} %`;
  const probability = result.subscription_probability;
  const threshold = result.decision_threshold;

  $("score-value").textContent = pct(probability);
  $("meter-fill").style.width = `${Math.min(100, Math.max(0, probability * 100))}%`;
  $("meter-threshold").style.left = `${threshold * 100}%`;
  $("meter-threshold-label").textContent = `Umbral ${pct(threshold)}`;
  $("score-meter").setAttribute(
    "aria-label",
    `Probabilidad ${pct(probability)}; umbral de decisión ${pct(threshold)}`,
  );

  const above = result.predicted_class === 1;
  const decision = $("decision");
  decision.className = `decision ${above ? "decision--above" : "decision--below"}`;
  decision.textContent = above
    ? "Supera el umbral de decisión: candidato a priorizar en la campaña."
    : "No supera el umbral de decisión.";

  $("fact-class").textContent = `${result.predicted_class} (${above ? "contrata" : "no contrata"})`;
  $("fact-threshold").textContent = threshold.toFixed(3);
  $("fact-model").textContent = `${result.model_name} · v${result.model_version}`;
  $("fact-latency").textContent = `${Number(result.inference_time_ms).toFixed(1)} ms`;
  showState("success");
}

function showError(message) {
  $("result-error").textContent = message;
  showState("error");
}

async function handleSubmit(event) {
  event.preventDefault();
  clearErrors();
  const payload = collectPayload();
  if (!payload) {
    $("form-error").textContent = "Revisa los campos marcados en rojo.";
    $("form-error").hidden = false;
    return;
  }
  setLoading(true);
  showState("loading");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.status === 422) {
      const body = await response.json().catch(() => ({}));
      for (const item of body.detail || []) {
        const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null;
        setFieldError(String(field), "Valor no válido para el modelo.");
      }
      $("form-error").textContent = "La API rechazó algunos valores. Revisa los campos marcados.";
      $("form-error").hidden = false;
      showState("empty");
      return;
    }
    if (!response.ok) {
      showError(`El servicio respondió con un error (${response.status}). Intenta de nuevo.`);
      return;
    }
    renderResult(await response.json());
  } catch {
    showError("No fue posible conectar con el servicio. Verifica que la API esté en ejecución.");
  } finally {
    setLoading(false);
  }
}

function fillExample() {
  const example = EXAMPLES[exampleIndex % EXAMPLES.length];
  exampleIndex += 1;
  clearErrors();
  for (const [name, value] of Object.entries(example)) {
    const control = $(`f-${name}`);
    if (control) control.value = String(value);
  }
}

async function loadHealth() {
  const status = $("service-status");
  const text = status.querySelector(".status__text");
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(String(response.status));
    const health = await response.json();
    status.classList.add("status--ok");
    text.textContent = `Servicio activo · modelo ${health.model_name} v${health.model_version}`;
  } catch {
    status.classList.add("status--down");
    text.textContent = "Servicio no disponible";
  }
}

async function init() {
  loadHealth();
  try {
    const response = await fetch("/openapi.json", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(String(response.status));
    const spec = await response.json();
    schema = spec.components.schemas[SCHEMA_NAME];
    renderForm(schema.properties, schema.required || Object.keys(schema.properties));
    $("submit-btn").disabled = false;
  } catch {
    $("form-fields").replaceChildren();
    showError("No fue posible cargar el formulario desde la API. Recarga la página.");
    return;
  }
  $("scoring-form").addEventListener("submit", handleSubmit);
  $("scoring-form").addEventListener("reset", () => {
    clearErrors();
    showState("empty");
  });
  $("load-example").addEventListener("click", fillExample);
}

document.addEventListener("DOMContentLoaded", init);
