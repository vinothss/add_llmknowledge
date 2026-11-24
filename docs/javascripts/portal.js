/* Portal interactivity: quizzes + per-learner progress (stored in this browser only). */
(function () {
  const KEY = "akllm-progress-v1";
  const MODULES = [
    ["m01", "Data curation & synthetic data", "modules/01-data-curation/"],
    ["m02", "Evaluating LLMs", "modules/02-evaluation/"],
    ["m03", "Customising LLMs (CPT, SFT, DPO)", "modules/03-customization/"],
    ["m04", "Optimising for deployment", "modules/04-optimization/"],
    ["m05", "Assessment - Safety alignment", "modules/05-assessment/"],
  ];

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; }
  }
  function save(state) {
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* storage blocked */ }
  }

  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => (k === "text" ? (n.textContent = v) : n.setAttribute(k, v)));
    (children || []).forEach((c) => n.appendChild(c));
    return n;
  }

  /* ---------- Quizzes: <script type="application/json" class="quiz" data-id="m01"> ---------- */
  function renderQuizzes() {
    document.querySelectorAll("script.quiz").forEach((script) => {
      if (script.dataset.rendered) return;
      script.dataset.rendered = "1";
      const id = script.dataset.id;
      const questions = JSON.parse(script.textContent);
      const box = el("div", { class: "quiz-box" });
      const answers = {};

      questions.forEach((q, qi) => {
        const card = el("fieldset", { class: "quiz-q" });
        card.appendChild(el("legend", { text: `${qi + 1}. ${q.q}` }));
        q.options.forEach((opt, oi) => {
          const input = el("input", { type: "radio", name: `${id}-${qi}`, value: String(oi) });
          input.addEventListener("change", () => (answers[qi] = oi));
          card.appendChild(el("label", { class: "quiz-opt" }, [input, document.createTextNode(" " + opt)]));
        });
        card.appendChild(el("div", { class: "quiz-feedback" }));
        box.appendChild(card);
      });

      const result = el("div", { class: "quiz-result" });
      const btn = el("button", { class: "md-button md-button--primary", text: "Check answers" });
      btn.addEventListener("click", () => {
        let score = 0;
        box.querySelectorAll(".quiz-q").forEach((card, qi) => {
          const fb = card.querySelector(".quiz-feedback");
          const q = questions[qi];
          const ok = answers[qi] === q.answer;
          score += ok ? 1 : 0;
          card.classList.toggle("is-correct", ok);
          card.classList.toggle("is-wrong", !ok);
          fb.textContent = answers[qi] === undefined ? "Pick an answer." : (ok ? "Correct. " : "Not quite. ") + (q.why || "");
        });
        result.textContent = `Score: ${score}/${questions.length}` + (score === questions.length ? " - module quiz passed!" : "");
        const state = load();
        state[id] = Object.assign({}, state[id], { quiz: Math.max(score, (state[id] || {}).quiz || 0), quizTotal: questions.length });
        save(state);
        renderTracker();
      });
      box.appendChild(btn);
      box.appendChild(result);
      script.after(box);
    });
  }

  /* ---------- "Mark lab complete" buttons: <div class="lab-complete" data-id="m01"></div> ---------- */
  function renderCompleteButtons() {
    document.querySelectorAll(".lab-complete").forEach((div) => {
      const id = div.dataset.id;
      div.innerHTML = "";
      const done = !!(load()[id] || {}).lab;
      const btn = el("button", { class: "md-button" + (done ? " md-button--primary" : ""), text: done ? "Lab completed" : "Mark lab as complete" });
      btn.addEventListener("click", () => {
        const state = load();
        state[id] = Object.assign({}, state[id], { lab: !done });
        save(state);
        renderCompleteButtons();
        renderTracker();
      });
      div.appendChild(btn);
    });
  }

  /* ---------- Progress tracker on the home page: <div id="progress-tracker"></div> ---------- */
  function renderTracker() {
    const root = document.getElementById("progress-tracker");
    if (!root) return;
    const state = load();
    const base = root.dataset.base || "";
    root.innerHTML = "";
    let done = 0;
    MODULES.forEach(([id, name, href]) => {
      const s = state[id] || {};
      const quizOk = s.quiz && s.quiz === s.quizTotal;
      if (s.lab && quizOk) done++;
      const row = el("a", { class: "tracker-row", href: base + href }, [
        el("span", { class: "tracker-name", text: name }),
        el("span", { class: "tracker-pill" + (s.lab ? " on" : ""), text: s.lab ? "Lab done" : "Lab" }),
        el("span", { class: "tracker-pill" + (quizOk ? " on" : ""), text: s.quiz ? `Quiz ${s.quiz}/${s.quizTotal}` : "Quiz" }),
      ]);
      root.appendChild(row);
    });
    const bar = el("div", { class: "tracker-bar" }, [el("div", { class: "tracker-fill", style: `width:${(done / MODULES.length) * 100}%` })]);
    root.prepend(el("div", { class: "tracker-summary", text: `${done} of ${MODULES.length} modules complete` }));
    root.prepend(bar);
  }

  /* ---------- Open links to other sites in a new tab, so learners keep their place ---------- */
  function externalLinksNewTab() {
    document.querySelectorAll(".md-content a[href^='http']").forEach((a) => {
      if (a.host !== window.location.host) {
        a.target = "_blank";
        a.rel = "noopener";
      }
    });
  }

  function init() {
    externalLinksNewTab();
    renderQuizzes();
    renderCompleteButtons();
    renderTracker();
  }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(init); // Material instant navigation
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
