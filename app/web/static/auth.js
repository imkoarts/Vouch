"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const state = { email: "", account: null, index: 0, answers: {}, other: {} };

const questions = [
  {
    id: "response_instinct", title: "When a post gives you something to react to, what do you do first?", type: "single",
    support: "Choose the move that feels most like you—not the one that sounds most polished.",
    options: [["direct_answer","Say exactly what I think","Lead with the point"],["ask_question","Ask the question it raises","Open the conversation"],["add_context","Add the missing context","Extend the source"],["challenge","Push back on the premise","Test the claim"],["make_a_joke","Find the funny angle","Use humor when it fits"]]
  },
  {
    id: "disagreement_style", title: "How do you naturally disagree?", type: "multi",
    support: "Select every approach you actually use.",
    options: [["state_it_directly","State it directly","No ceremonial softening"],["ask_why","Ask why","Let the weak point surface"],["qualify_it","Add a qualification","Keep the nuance"],["show_evidence","Show the evidence","Anchor the objection"],["keep_it_playful","Keep it playful","Lower the temperature"]]
  },
  {
    id: "reasoning_shape", title: "How do you usually explain an idea?", type: "single",
    support: "This shapes how Vouch builds the argument behind a draft.",
    options: [["conclusion_first","Conclusion first","Then the reason"],["step_by_step","Step by step","Make the path visible"],["analogy","Through an analogy","Connect it to something familiar"],["concrete_example","With a concrete example","Show it in action"],["think_out_loud","Think out loud","Let the idea unfold"]]
  },
  {
    id: "certainty_style", title: "How do you sound when the answer is not obvious?", type: "single",
    support: "There is no ideal level of certainty—only an honest one.",
    options: [["decisive","Decisive","Choose the strongest view"],["calibrated","Calibrated","Say what is known and unknown"],["exploratory","Exploratory","Work through possibilities"],["cautious","Cautious","Avoid overclaiming"],["depends_on_topic","It depends","Match confidence to the topic"]]
  },
  {
    id: "humor_style", title: "What kind of humor sounds like you?", type: "single",
    support: "Humor is optional. Vouch should never force a joke.",
    options: [["none","Usually none","Keep it straight"],["dry","Dry","Understated and brief"],["playful","Playful","Warm, not cutting"],["absurd","Absurd","Follow the strange logic"],["situational","Situational","Only when the source earns it"]]
  },
  {
    id: "sarcasm_boundary", title: "Where is your line with sarcasm?", type: "single",
    support: "This is a boundary, not a request to add sarcasm everywhere.",
    options: [["never","I avoid it","Say the point plainly"],["light_only","Light only","A hint, never the whole post"],["with_familiar_people","With familiar people","Context matters"],["often","Often","It is part of my voice"],["safe_targets_only","Only at safe targets","Ideas and situations, not people"]]
  },
  {
    id: "message_rhythm", title: "What rhythm feels natural when you write?", type: "single",
    support: "Think about your real messages, replies, and posts.",
    options: [["terse","Terse","One clean move"],["balanced","Balanced","Compact but complete"],["conversational","Conversational","Like talking to one person"],["layered","Layered","Build the thought in stages"],["depends_on_context","It depends","Let the subject set the pace"]]
  },
  {
    id: "voice_qualities", title: "How should your writing feel?", type: "multi", max: 3,
    support: "Choose up to three qualities.",
    options: [["calm","Calm","Never rushed"],["confident","Confident","Clear without bluffing"],["friendly","Friendly","Open and human"],["sharp","Sharp","Precise and alert"],["professional","Professional","Controlled and credible"],["energetic","Energetic","Movement without hype"]]
  },
  {
    id: "audience_relationship", title: "Who are you talking with—not at?", type: "single",
    support: "Vouch uses this to choose context, vocabulary, and assumed knowledge.",
    options: [["peers","Peers","Shared context"],["experts","Experts","Skip the basics"],["newcomers","Newcomers","Make it accessible"],["customers","Customers","Useful and concrete"],["broad_audience","A broad audience","Explain only what is needed"]]
  },
  {
    id: "feedback_directness", title: "How should Vouch challenge a weak draft?", type: "single",
    support: "This affects revision guidance—not the factual safety checks.",
    options: [["gentle","Gentle","Encourage, then correct"],["balanced","Balanced","Support plus clear advice"],["direct","Direct","Say what is wrong"],["coach_mode","Coach mode","Detailed, actionable critique"]]
  }
];

function csrfHeader() {
  const item = document.cookie.split("; ").find((value) => value.startsWith("vouch_csrf="));
  return item ? { "X-CSRF-Token": decodeURIComponent(item.split("=").slice(1).join("=")) } : {};
}

async function api(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...csrfHeader(), ...(options.headers || {}) } });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const payload = await response.json(); message = typeof payload.detail === "string" ? payload.detail : message; } catch (_) { /* keep status */ }
    throw new Error(message);
  }
  return response.json();
}

function setAuthMessage(message, ok = false) {
  $("#auth-state").textContent = message;
  $("#auth-state").classList.toggle("ok", ok);
}

function showCodeStep() {
  $("#email-form").hidden = true;
  $("#code-form").hidden = false;
  $("#login-title").textContent = "Check your email";
  $("#login-subtitle").textContent = `We sent a one-time code to ${state.email}.`;
  $("#auth-code").focus();
}

function showEmailStep() {
  $("#code-form").hidden = true;
  $("#email-form").hidden = false;
  $("#login-title").textContent = "Welcome to Vouch";
  $("#login-subtitle").textContent = "Sign in with a one-time code sent to your email.";
  setAuthMessage("");
  $("#auth-email").focus();
}

function beginOnboarding() {
  $("#login-view").hidden = true;
  $("#onboarding-view").hidden = false;
  renderQuestion();
}

function selectedValues(question) { return state.answers[question.id] || []; }

function answerValid(question) {
  const values = selectedValues(question);
  if (!values.length) return false;
  if (values.includes("other")) return (state.other[question.id] || "").trim().length >= 2;
  return true;
}

function renderQuestion() {
  const question = questions[state.index];
  const values = selectedValues(question);
  $("#progress-label").textContent = `Question ${state.index + 1} of 10`;
  $("#progress-fill").style.width = `${(state.index + 1) * 10}%`;
  $("#question-number").textContent = `${String(state.index + 1).padStart(2,"0")} / 10`;
  $("#question-title").textContent = question.title;
  $("#question-support").textContent = question.support;
  $("#selection-count").textContent = question.max ? `${values.filter(value => value !== "other").length + (values.includes("other") ? 1 : 0)} / ${question.max} selected` : "";
  $("#question-back").style.visibility = state.index === 0 ? "hidden" : "visible";
  $("#question-next span").textContent = state.index === 9 ? "Complete setup" : "Next";
  $("#question-next").disabled = !answerValid(question);
  $("#onboarding-error").textContent = "";

  const grid = $("#answer-grid");
  grid.replaceChildren();
  [...question.options, ["other","Other","Write your own answer"]].forEach(([value,label,description], optionIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "answer-card";
    button.dataset.value = value;
    button.setAttribute("aria-pressed", String(values.includes(value)));
    button.innerHTML = `<span class="answer-icon" aria-hidden="true">${optionIndex === question.options.length ? "+" : ["↗","?","≋","◇","∿","~"][optionIndex % 6]}</span><span class="answer-check" aria-hidden="true">✓</span><span class="answer-label">${label}<small>${description}</small></span>`;
    button.addEventListener("click", () => chooseAnswer(question, value));
    grid.append(button);
  });

  const other = $("#other-answer");
  other.hidden = !values.includes("other");
  $("#other-text").value = state.other[question.id] || "";
  if (!other.hidden) requestAnimationFrame(() => $("#other-text").focus());
  $("#question-view").style.animation = "none";
  requestAnimationFrame(() => { $("#question-view").style.animation = ""; });
}

function chooseAnswer(question, value) {
  let values = selectedValues(question);
  if (question.type === "single") {
    values = [value];
  } else if (values.includes(value)) {
    values = values.filter(item => item !== value);
  } else {
    if (question.max && values.length >= question.max) {
      $("#onboarding-error").textContent = `Choose up to ${question.max} answers.`;
      return;
    }
    values = [...values, value];
  }
  if (!values.includes("other")) state.other[question.id] = "";
  state.answers[question.id] = values;
  renderQuestion();
}

function serializedAnswers() {
  return Object.fromEntries(questions.map(question => [question.id, selectedValues(question).map(value => value === "other" ? `other:${state.other[question.id].trim()}` : value)]));
}

async function finishOnboarding() {
  const next = $("#question-next");
  next.disabled = true;
  try {
    await api("/api/voice-profile/onboarding", { method: "PUT", body: JSON.stringify({ answers: serializedAnswers() }) });
    $("#question-view").hidden = true;
    $("#question-navigation").hidden = true;
    $("#success-view").hidden = false;
    setTimeout(() => window.location.replace("/"), 1200);
  } catch (error) {
    $("#onboarding-error").textContent = error.message;
    next.disabled = false;
  }
}

$("#email-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = $("#auth-email").value.trim();
  if (!email || !$("#auth-email").checkValidity()) { setAuthMessage("Enter a valid email address."); return; }
  const button = $("#email-form button"); button.disabled = true; setAuthMessage("Sending your code…", true);
  try {
    await api("/api/auth/otp", { method: "POST", body: JSON.stringify({ email }) });
    state.email = email;
    showCodeStep();
    setAuthMessage("Code sent. It may take a few seconds to arrive.", true);
  } catch (error) { setAuthMessage(error.message); }
  finally { button.disabled = false; }
});

$("#code-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = $("#auth-code").value.trim();
  const button = $("#code-form .primary-auth-button"); button.disabled = true; setAuthMessage("Verifying…", true);
  try {
    await api("/api/auth/verify", { method: "POST", body: JSON.stringify({ email: state.email, token }) });
    state.account = await api("/api/auth/session");
    if (state.account.onboarding_complete) {
      $("#login-title").textContent = `Welcome back, ${state.account.display_name}`;
      $("#login-subtitle").textContent = "Your workspace and voice profile are ready.";
      $("#code-form").hidden = true;
      setAuthMessage("Opening your workspace…", true);
      setTimeout(() => window.location.replace("/"), 800);
    } else beginOnboarding();
  } catch (error) { setAuthMessage(error.message); button.disabled = false; }
});

$("#back-to-email").addEventListener("click", showEmailStep);
$("#other-text").addEventListener("input", (event) => {
  const question = questions[state.index]; state.other[question.id] = event.target.value;
  $("#question-next").disabled = !answerValid(question);
});
$("#question-back").addEventListener("click", () => { if (state.index > 0) { state.index -= 1; renderQuestion(); } });
$("#question-next").addEventListener("click", () => {
  const question = questions[state.index];
  if (!answerValid(question)) return;
  if (state.index === questions.length - 1) { finishOnboarding(); return; }
  state.index += 1; renderQuestion();
});

(async function initialize() {
  try {
    await api("/api/auth/config");
    state.account = await api("/api/auth/session");
    const forceVoice = new URLSearchParams(window.location.search).get("mode") === "voice";
    if (forceVoice || !state.account.onboarding_complete) { beginOnboarding(); return; }
    $("#login-title").textContent = `Welcome back, ${state.account.display_name}`;
    $("#login-subtitle").textContent = "Your workspace and voice profile are ready.";
    $("#email-form").hidden = true;
    setAuthMessage("Opening your workspace…", true);
    setTimeout(() => window.location.replace("/"), 800);
  } catch (_) { showEmailStep(); }
})();
