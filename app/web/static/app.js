"use strict";

const state = {
  page: "chat",
  status: null,
  settings: null,
  settingsDirty: false,
  settingsSection: "general",
  evergreenTopics: [],
  accounts: null,
  accountsDirty: false,
  feed: new Map(),
  feedOrder: [],
  nextCursor: null,
  loadingOlder: false,
  activityStream: null,
  unseenPosts: 0,
  chatIds: new Set(),
  chatLoaded: false,
  drafts: [],
  authConfig: null,
  account: null,
  voiceProfile: null,
  credentials: null,
  draftAction: null,
  publishPreviewDraftId: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(url, options = {}) {
  const csrf = document.cookie.split("; ").find((item) => item.startsWith("vouch_csrf="));
  const csrfHeader = csrf ? { "X-CSRF-Token": decodeURIComponent(csrf.split("=").slice(1).join("=")) } : {};
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...csrfHeader, ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch (_) {
      // Keep the HTTP status message.
    }
    throw new Error(detail);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

async function requireAccount() {
  const config = await api("/api/auth/config");
  state.authConfig = config;
  try {
    state.account = await api("/api/auth/session");
    const voice = await api("/api/voice-profile");
    state.voiceProfile = voice.profile;
    if (!voice.profile?.onboarding_complete) window.location.replace("/login?mode=voice");
    return;
  } catch (_) {
    window.location.replace("/login");
    await new Promise(() => {});
  }
}

async function loadAccountPanel() {
  try {
    state.account = await api("/api/auth/session");
    const voice = await api("/api/voice-profile");
    state.voiceProfile = voice.profile;
    $("#account-email").textContent = state.account.email;
    $("#profile-email").textContent = state.account.email;
    $("#profile-popover-email").textContent = state.account.email;
    $("#profile-initial").textContent = state.account.email.charAt(0).toUpperCase();
    $("#account-auth-mode").textContent = `${state.authConfig.mode} auth · Vouch ${state.authConfig.version} · ${state.authConfig.runtime_revision}`;
    $("#voice-summary").textContent = voice.profile
      ? `${voice.profile.account_type} · ${voice.profile.sample_count} analyzed X posts · ${voice.profile.response_preferences.join(", ") || "default responses"}`
      : "Voice profile is not configured.";
    $("#analyze-voice").disabled = !voice.profile?.x_username;
  } catch (error) {
    toast(`Could not load account: ${error.message}`, true);
  }
}

async function analyzeVoice() {
  const button = $("#analyze-voice");
  button.disabled = true;
  button.textContent = "Analyzing…";
  try {
    const result = await api("/api/voice-profile/analyze", { method: "POST" });
    state.voiceProfile = result.profile;
    await loadAccountPanel();
    toast(`Voice updated from ${result.evidence.sample_count} X posts and replies`);
  } catch (error) {
    toast(`Voice analysis failed: ${error.message}`, true);
  } finally {
    button.textContent = "Re-analyze X voice";
    button.disabled = !state.voiceProfile?.x_username;
  }
}

async function logoutAccount() {
  try {
    await api("/api/auth/logout", { method: "POST" });
    window.location.reload();
  } catch (error) {
    toast(`Logout failed: ${error.message}`, true);
  }
}

function toast(message, error = false) {
  const element = document.createElement("div");
  element.className = `toast${error ? " is-error" : ""}`;
  element.textContent = message;
  $("#toast-region").append(element);
  window.setTimeout(() => element.remove(), 4400);
}

function setPage(page) {
  if (page === "settings") {
    const dialog = $("#settings-dialog");
    $("#page-settings").classList.add("is-visible");
    if (!dialog.open) dialog.showModal();
    $$(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.page === "settings"));
    if (!state.settings) loadSettings();
    loadAccountPanel();
    loadCredentials();
    return;
  }
  state.page = page;
  const titles = {
    chat: ["WORKSPACE", "Command chat"],
    settings: ["PREFERENCES", "Settings"],
    accounts: ["X SOURCES", "Selected accounts"],
  };
  $$(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.page === page));
  $$('[data-page-panel]').forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.pagePanel === page));
  $("#page-eyebrow").textContent = titles[page][0];
  $("#page-title").textContent = titles[page][1];
  if (page === "accounts" && !state.accounts) loadSelectedAccounts();
}

function closeSettings() {
  const dialog = $("#settings-dialog");
  if (dialog.open) dialog.close();
  $$(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.page === state.page));
}

function toggleProfileMenu(force) {
  const popover = $("#profile-popover");
  const open = typeof force === "boolean" ? force : popover.hidden;
  popover.hidden = !open;
  $("#profile-menu-button").setAttribute("aria-expanded", String(open));
}

function setSettingsSection(section) {
  state.settingsSection = section;
  const metadata = {
    general: ["General", "Core generation behavior and presentation."],
    ai: ["AI & Telegram", "Review notifications and integration behavior."],
    sources: ["X sources", "Home timeline, manual post links, and evergreen themes."],
    discovery: ["Discovery", "Schedule, limits, trends, and bounded candidate selection."],
    publishing: ["Publishing", "Manual publication controls and safety boundaries."],
    advanced: ["Advanced", "Conservative request pacing for external APIs."],
  };
  $$(".settings-nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.settingsSection === section));
  $$('[data-settings-panel]').forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.settingsPanel === section));
  $("#settings-section-title").textContent = metadata[section][0];
  $("#settings-section-description").textContent = metadata[section][1];
}

function setConnection(ok, label) {
  void ok;
  $("#progress-label").textContent = label;
}

async function loadStatus() {
  try {
    const status = await api("/api/dashboard/status");
    state.status = status;
    $("#version-label").textContent = `Version ${status.version}`;
    $("#post-count").textContent = String(status.total_posts);
    const sources = [];
    if (status.home_timeline) sources.push("Home timeline");
    if (status.selected_accounts) {
      sources.push(`${status.selected_accounts} selected ${status.selected_accounts === 1 ? "account" : "accounts"}`);
    }
    if (status.trends) sources.push("Trends");
    const sourceLabel = sources.length ? sources.join(" + ") : "no readable sources";
    const scheduleLabel = status.startup_discovery
      ? `startup scan, then every ${status.discovery_interval_minutes}m`
      : "manual runs only";
    setConnection(
      true,
      status.automatic_discovery
        ? `Discovery enabled · ${sourceLabel} · ${scheduleLabel}`
        : "Discovery is paused",
    );
  } catch (_) {
    setConnection(false, "Connection lost");
  }
}

function formatDate(value) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function compactNumber(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "0";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(number);
}

function authorInitial(author) {
  const source = (author && (author.name || author.username)) || "X";
  return source.trim().charAt(0).toUpperCase() || "X";
}

function makeAvatar(author) {
  if (author && author.avatar_url) {
    const image = document.createElement("img");
    image.className = "tweet-avatar";
    image.src = author.avatar_url;
    image.alt = "";
    image.loading = "lazy";
    image.referrerPolicy = "no-referrer";
    image.addEventListener("error", () => {
      const fallback = document.createElement("div");
      fallback.className = "tweet-avatar";
      fallback.textContent = authorInitial(author);
      image.replaceWith(fallback);
    });
    return image;
  }
  const fallback = document.createElement("div");
  fallback.className = "tweet-avatar";
  fallback.textContent = authorInitial(author);
  return fallback;
}

function makeMedia(post) {
  const items = (post.media || []).filter((item) => item.url || item.preview_image_url || item.video_url);
  if (!items.length) return null;
  const grid = document.createElement("div");
  grid.className = `tweet-media${items.length === 1 ? " one" : ""}`;
  for (const item of items.slice(0, 4)) {
    let element;
    if (item.video_url) {
      element = document.createElement("video");
      element.src = item.video_url;
      element.poster = item.preview_image_url || item.url || "";
      element.controls = true;
      element.preload = "metadata";
    } else {
      element = document.createElement("img");
      element.src = item.url || item.preview_image_url;
      element.loading = "lazy";
      element.referrerPolicy = "no-referrer";
    }
    element.alt = item.alt_text || "Source post media";
    element.addEventListener("error", () => element.remove());
    grid.append(element);
  }
  return grid;
}

function makeMetric(icon, value, title) {
  const item = document.createElement("span");
  item.className = "metric";
  item.title = title;
  const iconNode = document.createElement("span");
  iconNode.textContent = icon;
  const valueNode = document.createElement("span");
  valueNode.textContent = compactNumber(value);
  item.append(iconNode, valueNode);
  return item;
}

function makeTweetCard(post) {
  const card = document.createElement("article");
  card.className = "tweet-card";
  card.dataset.postId = post.id;

  const head = document.createElement("div");
  head.className = "tweet-head";
  head.append(makeAvatar(post.author || {}));

  const identity = document.createElement("div");
  identity.className = "tweet-identity";
  const nameRow = document.createElement("div");
  nameRow.className = "tweet-name-row";
  const name = document.createElement("span");
  name.className = "tweet-name";
  name.textContent = (post.author && (post.author.name || post.author.username)) || "Unknown author";
  nameRow.append(name);
  if (post.author && post.author.verified) {
    const verified = document.createElement("span");
    verified.className = "verified";
    verified.textContent = "●";
    verified.title = "Verified";
    nameRow.append(verified);
  }
  const handle = document.createElement("div");
  handle.className = "tweet-handle";
  handle.textContent = post.author && post.author.username ? `@${post.author.username}` : "Source post";
  identity.append(nameRow, handle);
  head.append(identity);

  if (post.url) {
    const open = document.createElement("button");
    open.className = "tweet-open";
    open.type = "button";
    open.textContent = "↗";
    open.title = "Open original post";
    open.addEventListener("click", () => window.open(post.url, "_blank", "noopener,noreferrer"));
    head.append(open);
  }
  card.append(head);

  const text = document.createElement("p");
  text.className = "tweet-text";
  text.textContent = post.text || "";
  card.append(text);

  const media = makeMedia(post);
  if (media) card.append(media);

  const meta = document.createElement("div");
  meta.className = "tweet-meta";
  const time = document.createElement("span");
  time.textContent = formatDate(post.published_at || post.fetched_at);
  meta.append(time);
  for (const label of (post.source_labels || []).slice(0, 4)) {
    const badge = document.createElement("span");
    badge.className = "tweet-label";
    badge.textContent = label;
    meta.append(badge);
  }
  card.append(meta);

  const metrics = post.public_metrics || {};
  const metricsRow = document.createElement("div");
  metricsRow.className = "tweet-metrics";
  metricsRow.append(
    makeMetric("↩", metrics.reply_count, "Replies"),
    makeMetric("⇄", metrics.retweet_count || metrics.repost_count, "Reposts"),
    makeMetric("♡", metrics.like_count, "Likes"),
    makeMetric("◉", metrics.impression_count || metrics.view_count, "Views"),
  );
  card.append(metricsRow);
  return card;
}

function renderFeed({ preserveScroll = false } = {}) {
  const scroll = $("#activity-scroll");
  const previousHeight = scroll.scrollHeight;
  const previousTop = scroll.scrollTop;
  const feed = $("#tweet-feed");
  const fragment = document.createDocumentFragment();
  for (const id of state.feedOrder) {
    const post = state.feed.get(id);
    if (post) fragment.append(makeTweetCard(post));
  }
  feed.replaceChildren(fragment);
  $("#activity-empty").hidden = state.feedOrder.length > 0;
  $("#load-more-button").hidden = !state.nextCursor;
  $("#feed-end").hidden = state.feedOrder.length === 0 || Boolean(state.nextCursor);
  if (preserveScroll) {
    const delta = scroll.scrollHeight - previousHeight;
    scroll.scrollTop = previousTop + delta;
  }
}

function feedTimestamp(post) {
  const value = post.updated_at || post.fetched_at || post.published_at || "";
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sortFeedNewestFirst() {
  state.feedOrder.sort((leftId, rightId) => {
    const left = state.feed.get(leftId);
    const right = state.feed.get(rightId);
    const byTime = feedTimestamp(right || {}) - feedTimestamp(left || {});
    return byTime || String(rightId).localeCompare(String(leftId));
  });
}

function mergePosts(items, { prepend = false, preserveScroll = false } = {}) {
  let added = 0;
  for (const post of items) {
    const exists = state.feed.has(post.id);
    state.feed.set(post.id, post);
    if (!exists) {
      added += 1;
      if (prepend) state.feedOrder.unshift(post.id);
      else state.feedOrder.push(post.id);
    }
  }
  const seen = new Set();
  state.feedOrder = state.feedOrder.filter((id) => !seen.has(id) && seen.add(id));
  sortFeedNewestFirst();
  renderFeed({ preserveScroll: preserveScroll && added > 0 });
  return added;
}

async function loadInitialFeed() {
  try {
    const payload = await api("/api/activity/feed?limit=30");
    state.nextCursor = payload.next_cursor;
    mergePosts(payload.items);
    connectActivityStream();
  } catch (error) {
    toast(`Could not load Activity: ${error.message}`, true);
  }
}

async function loadOlderPosts() {
  if (!state.nextCursor || state.loadingOlder) return;
  state.loadingOlder = true;
  const button = $("#load-more-button");
  button.disabled = true;
  button.textContent = "Loading...";
  try {
    const query = new URLSearchParams({
      limit: "30",
      before: state.nextCursor.before,
      before_id: state.nextCursor.before_id,
    });
    const payload = await api(`/api/activity/feed?${query}`);
    state.nextCursor = payload.next_cursor;
    mergePosts(payload.items);
  } catch (error) {
    toast(`Could not load older posts: ${error.message}`, true);
  } finally {
    state.loadingOlder = false;
    button.disabled = false;
    button.textContent = "Load older posts";
  }
}

function latestFeedCursor() {
  let latest = null;
  for (const id of state.feedOrder) {
    const post = state.feed.get(id);
    if (!post || !post.updated_at) continue;
    if (!latest || post.updated_at > latest.updated_at || (post.updated_at === latest.updated_at && post.id > latest.id)) latest = post;
  }
  return latest;
}

function connectActivityStream() {
  if (state.activityStream) state.activityStream.close();
  const cursor = latestFeedCursor();
  const query = new URLSearchParams();
  if (cursor) {
    query.set("after", cursor.updated_at);
    query.set("after_id", cursor.id);
  }
  const source = new EventSource(`/api/activity/stream${query.toString() ? `?${query}` : ""}`);
  state.activityStream = source;
  source.addEventListener("source_post", (event) => {
    try {
      const post = JSON.parse(event.data);
      const scroll = $("#activity-scroll");
      const awayFromTop = scroll.scrollTop > 90;
      const added = mergePosts([post], { prepend: true, preserveScroll: awayFromTop });
      if (added && awayFromTop) {
        state.unseenPosts += added;
        const button = $("#new-posts-button");
        button.hidden = false;
        button.textContent = `Show ${state.unseenPosts} new ${state.unseenPosts === 1 ? "post" : "posts"}`;
      }
      loadStatus();
    } catch (_) {
      // Ignore malformed stream events; reconnect or polling will recover.
    }
  });
  source.onopen = () => setConnection(true, state.status && state.status.automatic_discovery ? "Discovery is enabled" : "Live feed connected");
  source.onerror = () => setConnection(false, "Reconnecting live feed...");
}

function makeMessage(message) {
  const row = document.createElement("div");
  row.className = `message${message.role === "user" ? " is-user" : ""}`;
  row.dataset.messageId = message.id;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = message.role === "user" ? "Y" : "X";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const text = document.createElement("p");
  const value = String(message.text || "");
  const urlPattern = /https?:\/\/[^\s<>()]+/g;
  let cursor = 0;
  for (const match of value.matchAll(urlPattern)) {
    const candidate = match[0].replace(/[.,!?;:]+$/, "");
    text.append(document.createTextNode(value.slice(cursor, match.index)));
    try {
      const parsed = new URL(candidate);
      if (parsed.protocol === "https:" || parsed.protocol === "http:") {
        const link = document.createElement("a");
        link.href = parsed.href;
        link.textContent = candidate;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        text.append(link);
      } else {
        text.append(document.createTextNode(candidate));
      }
    } catch (_) {
      text.append(document.createTextNode(candidate));
    }
    cursor = Number(match.index) + candidate.length;
  }
  text.append(document.createTextNode(value.slice(cursor)));
  const time = document.createElement("div");
  time.className = "message-time";
  time.textContent = formatDate(message.created_at);
  bubble.append(text, time);
  row.append(avatar, bubble);
  return row;
}

function appendChatMessages(messages, { scroll = true } = {}) {
  const container = $("#chat-messages");
  let added = 0;
  for (const message of messages) {
    if (!message.id || state.chatIds.has(message.id)) continue;
    state.chatIds.add(message.id);
    container.append(makeMessage(message));
    added += 1;
  }
  if (added) container.querySelector(".message-welcome")?.remove();
  if (added && scroll) container.scrollTop = container.scrollHeight;
}

function renderWelcome() {
  if (state.chatIds.size || $("#chat-messages .message-welcome")) return;
  const welcome = makeMessage({
    id: "welcome",
    role: "assistant",
    text: "The dashboard is ready. Use /new with a direct X post link, or type /help for all commands.",
    created_at: new Date().toISOString(),
  });
  welcome.classList.add("message-welcome");
  $("#chat-messages").append(welcome);
}

async function loadChatHistory() {
  try {
    const payload = await api("/api/chat/history?limit=150");
    appendChatMessages(payload.items, { scroll: !state.chatLoaded });
    state.chatLoaded = true;
    if (!payload.items.length) renderWelcome();
  } catch (error) {
    if (!state.chatLoaded) toast(`Could not load chat history: ${error.message}`, true);
  }
}

function renderRecentDrafts() {
  const panel = $("#draft-review");
  const list = $("#draft-review-list");
  const items = state.drafts || [];
  panel.hidden = items.length === 0;
  list.replaceChildren();
  if (!items.length) return;
  $("#draft-review-count").textContent = `${items.length} saved`;
  for (const draft of items) {
    const article = document.createElement("article");
    article.className = "draft-review-card";
    const meta = document.createElement("div");
    meta.className = "draft-review-meta";
    const status = document.createElement("span");
    status.textContent = String(draft.status || "needs_review").replaceAll("_", " ");
    const id = document.createElement("code");
    id.textContent = String(draft.id || "").slice(0, 8);
    meta.append(status, id);
    const text = document.createElement("p");
    text.textContent = draft.text || "";
    const footer = document.createElement("small");
    const blocked = (draft.blocking_flags || []).length;
    footer.textContent = blocked
      ? `${blocked} blocking flag${blocked === 1 ? "" : "s"} · review required`
      : `${draft.fact_check_status || "not_required"} · approval required`;
    const actions = document.createElement("div");
    actions.className = "draft-review-actions";
    const addAction = (label, action, enabled = true, className = "") => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.disabled = !enabled;
      if (className) button.className = className;
      button.addEventListener("click", () => openDraftAction(action, draft));
      actions.append(button);
    };
    if (draft.can_edit) addAction("Edit", "edit");
    if (draft.can_rewrite) addAction("Rewrite", "rewrite");
    if (draft.can_approve) {
      addAction("Approve", "approve", blocked === 0, "is-primary");
      if (blocked) actions.lastElementChild.title = "Resolve blocking flags before approval";
    }
    if (draft.can_publish) addAction("Publish", "publish", true, "is-primary");
    if (draft.can_reject) addAction("Reject", "reject", true, "is-danger");
    article.append(meta, text, footer, actions);
    list.append(article);
  }
}

function closeDraftAction() {
  const dialog = $("#draft-action-dialog");
  if (dialog.open) dialog.close();
  state.draftAction = null;
}

async function openDraftAction(mode, draft) {
  if (mode === "publish" && state.publishPreviewDraftId === draft.id) return;
  const dialog = $("#draft-action-dialog");
  const textWrap = $("#draft-action-text-wrap");
  const text = $("#draft-action-text");
  const factsWrap = $("#draft-facts-wrap");
  const publishWrap = $("#draft-publish-confirmation");
  const submit = $("#draft-action-submit");
  state.draftAction = { mode, draft, preview: null };
  textWrap.hidden = true;
  factsWrap.hidden = true;
  publishWrap.hidden = true;
  text.value = "";
  $("#draft-facts-confirmed").checked = false;
  $("#draft-action-eyebrow").textContent = `DRAFT ${String(draft.id).slice(0, 8)}`;

  if (mode === "edit") {
    $("#draft-action-title").textContent = "Edit draft";
    $("#draft-action-description").textContent = "Saving creates a new version and resets any previous approval.";
    $("#draft-action-text-label").textContent = "Draft text";
    text.value = draft.text || "";
    textWrap.hidden = false;
    submit.textContent = "Save changes";
  } else if (mode === "rewrite") {
    $("#draft-action-title").textContent = "Rewrite draft";
    $("#draft-action-description").textContent = "Add focused guidance, or leave it blank for a fresh version. This makes one explicit AI generation request.";
    $("#draft-action-text-label").textContent = "Rewrite guidance (optional)";
    text.placeholder = "Make it sharper, less formal, keep the core observation…";
    textWrap.hidden = false;
    submit.textContent = "Generate rewrite";
  } else if (mode === "approve") {
    $("#draft-action-title").textContent = "Approve this version?";
    $("#draft-action-description").textContent = "Approval is bound to the current text and media hash. Any later edit will revoke it.";
    factsWrap.hidden = draft.fact_check_status !== "required";
    submit.textContent = "Approve draft";
  } else if (mode === "reject") {
    $("#draft-action-title").textContent = "Reject this draft?";
    $("#draft-action-description").textContent = "The draft will leave the review queue and move to local quarantine. Nothing will be deleted from X.";
    submit.textContent = "Reject draft";
  } else if (mode === "publish") {
    $("#draft-action-title").textContent = "Prepare manual publication";
    $("#draft-action-description").textContent = "Vouch is checking the approved hash, facts, media and configured X account. No post is sent yet.";
    submit.textContent = "Checking…";
    submit.disabled = true;
  }
  if (!dialog.open) dialog.showModal();

  if (mode === "publish") {
    state.publishPreviewDraftId = draft.id;
    try {
      const preview = await api(`/api/drafts/${encodeURIComponent(draft.id)}/publish-preview`, { method: "POST" });
      if (!state.draftAction || state.draftAction.draft.id !== draft.id) return;
      state.draftAction.preview = preview;
      $("#draft-publish-summary").textContent = (preview.parts || []).join("\n\n");
      publishWrap.hidden = false;
      $("#draft-action-description").textContent = `Ready to ${preview.action} as @${preview.account_username || "configured account"}. This is the final write confirmation.`;
      submit.textContent = "Publish to X";
      submit.disabled = false;
    } catch (error) {
      closeDraftAction();
      toast(`Publication check failed: ${error.message}`, true);
      await loadChatHistory();
    } finally {
      if (state.publishPreviewDraftId === draft.id) state.publishPreviewDraftId = null;
    }
  } else {
    submit.disabled = false;
    window.setTimeout(() => {
      if (!textWrap.hidden) text.focus();
    }, 0);
  }
}

async function submitDraftAction(event) {
  event.preventDefault();
  const action = state.draftAction;
  if (!action) return;
  const { mode, draft } = action;
  const submit = $("#draft-action-submit");
  submit.disabled = true;
  try {
    let payload;
    if (mode === "edit") {
      const text = $("#draft-action-text").value.trim();
      if (!text) throw new Error("Draft text cannot be empty");
      payload = await api(`/api/drafts/${encodeURIComponent(draft.id)}`, {
        method: "PUT",
        body: JSON.stringify({ text }),
      });
    } else if (mode === "rewrite") {
      payload = await api(`/api/drafts/${encodeURIComponent(draft.id)}/rewrite`, {
        method: "POST",
        body: JSON.stringify({ feedback: $("#draft-action-text").value.trim() }),
      });
    } else if (mode === "approve") {
      if (draft.fact_check_status === "required" && !$("#draft-facts-confirmed").checked) {
        throw new Error("Confirm that you reviewed the factual claims first");
      }
      payload = await api(`/api/drafts/${encodeURIComponent(draft.id)}/approve`, {
        method: "POST",
        body: JSON.stringify({ facts_confirmed: $("#draft-facts-confirmed").checked }),
      });
    } else if (mode === "reject") {
      payload = await api(`/api/drafts/${encodeURIComponent(draft.id)}/reject`, { method: "POST" });
    } else if (mode === "publish") {
      payload = await api(`/api/drafts/${encodeURIComponent(draft.id)}/publish`, {
        method: "POST",
        body: JSON.stringify({ confirmation_phrase: action.preview?.confirmation_phrase || "" }),
      });
    }
    closeDraftAction();
    toast(payload?.message || (mode === "publish" ? "Published to X" : "Draft updated"));
    await Promise.all([loadRecentDrafts(), loadChatHistory()]);
  } catch (error) {
    toast(`${mode === "publish" ? "Publication" : "Draft action"} failed: ${error.message}`, true);
  } finally {
    submit.disabled = false;
  }
}

async function loadRecentDrafts() {
  try {
    const payload = await api("/api/drafts/recent?limit=3");
    state.drafts = payload.items || [];
    renderRecentDrafts();
  } catch (error) {
    toast(`Could not load drafts: ${error.message}`, true);
  }
}

async function sendCommand(text) {
  const input = $("#chat-input");
  const send = $("#chat-send");
  send.disabled = true;
  input.disabled = true;
  try {
    const payload = await api("/api/chat/command", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    $("#chat-messages .message-welcome")?.remove();
    appendChatMessages(payload.messages || []);
    if (payload.action === "settings") setPage("settings");
    if (payload.action === "accounts") setPage("accounts");
    if (payload.action === "activity") $("#activity-scroll").scrollTo({ top: 0, behavior: "smooth" });
    await loadStatus();
  } catch (error) {
    toast(`Command failed: ${error.message}`, true);
  } finally {
    send.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function renderEvergreenTopics() {
  const list = $("#evergreen-topic-list");
  list.replaceChildren();
  if (!state.evergreenTopics.length) {
    const empty = document.createElement("p");
    empty.className = "topic-empty";
    empty.textContent = "No evergreen topics configured.";
    list.append(empty);
    return;
  }
  state.evergreenTopics.forEach((topic, index) => {
    const item = document.createElement("div");
    item.className = "topic-item";
    const text = document.createElement("span");
    text.textContent = topic;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove topic";
    remove.addEventListener("click", () => {
      state.evergreenTopics.splice(index, 1);
      markSettingsDirty();
      renderEvergreenTopics();
    });
    item.append(text, remove);
    list.append(item);
  });
}

function addEvergreenTopic() {
  const input = $("#evergreen-topic-input");
  const value = input.value.trim().replace(/\s+/g, " ");
  if (!value) return;
  if (state.evergreenTopics.some((topic) => topic.toLowerCase() === value.toLowerCase())) {
    toast("That evergreen topic already exists", true);
    return;
  }
  state.evergreenTopics.push(value);
  input.value = "";
  markSettingsDirty();
  renderEvergreenTopics();
}

function populateSettings(settings) {
  state.settings = settings;
  state.evergreenTopics = [...(settings.evergreen_topics || [])];
  $("#generation-provider").value = settings.generation_provider;
  $("#post-length-mode").value = settings.post_length_mode;
  $("#humanizer-enabled").checked = settings.humanizer_enabled;
  $("#images-enabled").checked = settings.images_enabled;
  $("#automatic-discovery").checked = settings.automatic_discovery_enabled;
  $("#interval-preset").value = settings.interval_preset;
  $("#custom-interval").value = settings.custom_interval_minutes;
  $("#max-runs").value = settings.max_runs_per_utc_day;
  $("#lookback-hours").value = settings.lookback_hours;
  $("#trends-woeid").value = settings.trends_woeid;
  $("#max-trends").value = settings.max_trends;
  $("#max-total-posts").value = settings.max_total_posts;
  $("#final-candidates").value = settings.final_candidates;
  $("#generation-candidates").value = settings.generation_candidates_per_run;
  $("#notify-no-candidate").checked = settings.notify_when_no_candidate;
  $("#home-enabled").checked = settings.home_timeline_enabled;
  $("#home-max-posts").value = settings.home_max_posts;
  $("#home-exclude-replies").checked = settings.home_exclude_replies;
  $("#home-exclude-retweets").checked = settings.home_exclude_retweets;
  $("#manual-sources-enabled").checked = settings.manual_sources_enabled;
  $("#evergreen-enabled").checked = settings.evergreen_enabled;
  $("#telegram-enabled").checked = settings.telegram_enabled;
  $("#telegram-autostart").checked = settings.telegram_autostart;
  $("#telegram-notify").checked = settings.telegram_notify_on_new_draft;
  $("#manual-publish-enabled").checked = settings.manual_x_publish_enabled;
  $("#enterprise-quotes-enabled").checked = settings.enterprise_quote_posts_enabled;
  $("#x-request-delay").value = settings.x_request_delay_seconds;
  $("#llm-minimum-interval").value = settings.llm_minimum_interval_seconds;
  renderEvergreenTopics();
  updateCustomIntervalVisibility();
  state.settingsDirty = false;
  const badge = $("#settings-state");
  badge.textContent = "Synced";
  badge.className = "save-state is-saved";
}

function collectSettings() {
  return {
    generation_provider: $("#generation-provider").value,
    post_length_mode: $("#post-length-mode").value,
    humanizer_enabled: $("#humanizer-enabled").checked,
    images_enabled: $("#images-enabled").checked,
    automatic_discovery_enabled: $("#automatic-discovery").checked,
    interval_preset: $("#interval-preset").value,
    custom_interval_minutes: Number($("#custom-interval").value),
    max_runs_per_utc_day: Number($("#max-runs").value),
    lookback_hours: Number($("#lookback-hours").value),
    trends_woeid: Number($("#trends-woeid").value),
    max_trends: Number($("#max-trends").value),
    max_total_posts: Number($("#max-total-posts").value),
    final_candidates: Number($("#final-candidates").value),
    generation_candidates_per_run: Number($("#generation-candidates").value),
    notify_when_no_candidate: $("#notify-no-candidate").checked,
    home_timeline_enabled: $("#home-enabled").checked,
    home_max_posts: Number($("#home-max-posts").value),
    home_exclude_replies: $("#home-exclude-replies").checked,
    home_exclude_retweets: $("#home-exclude-retweets").checked,
    manual_sources_enabled: $("#manual-sources-enabled").checked,
    evergreen_enabled: $("#evergreen-enabled").checked,
    evergreen_topics: state.evergreenTopics,
    telegram_enabled: $("#telegram-enabled").checked,
    telegram_autostart: $("#telegram-autostart").checked,
    telegram_notify_on_new_draft: $("#telegram-notify").checked,
    manual_x_publish_enabled: $("#manual-publish-enabled").checked,
    enterprise_quote_posts_enabled: $("#enterprise-quotes-enabled").checked,
    x_request_delay_seconds: Number($("#x-request-delay").value),
    llm_minimum_interval_seconds: Number($("#llm-minimum-interval").value),
  };
}

function updateCustomIntervalVisibility() {
  $("#custom-interval-row").hidden = $("#interval-preset").value !== "custom";
}

function markSettingsDirty() {
  state.settingsDirty = true;
  const badge = $("#settings-state");
  badge.textContent = "Unsaved changes";
  badge.className = "save-state";
}

async function loadSettings({ silent = false } = {}) {
  if (state.settingsDirty && silent) return;
  try {
    const settings = await api("/api/dashboard/settings");
    populateSettings(settings);
  } catch (error) {
    const badge = $("#settings-state");
    badge.textContent = "Load failed";
    badge.className = "save-state is-error";
    if (!silent) toast(`Could not load settings: ${error.message}`, true);
  }
}

const credentialFields = {
  openai_api_key: "#credential-openai-api-key",
  xai_api_key: "#credential-xai-api-key",
  x_bearer_token: "#credential-x-bearer-token",
  x_consumer_key: "#credential-x-consumer-key",
  x_consumer_secret: "#credential-x-consumer-secret",
  x_access_token: "#credential-x-access-token",
  x_access_token_secret: "#credential-x-access-token-secret",
  x_client_id: "#credential-x-client-id",
  x_client_secret: "#credential-x-client-secret",
  x_user_id: "#credential-x-user-id",
  telegram_bot_token: "#credential-telegram-token",
  heygen_api_key: "#credential-heygen-api-key",
};

async function loadCredentials() {
  try {
    const payload = await api("/api/dashboard/credentials");
    state.credentials = payload;
    for (const [field, configured] of Object.entries(payload.configured || {})) {
      $$(`[data-credential-status="${field}"]`).forEach((label) => {
        label.textContent = configured ? "Configured" : "Not configured";
        label.classList.toggle("is-configured", configured);
      });
    }
  } catch (error) {
    $("#credential-state").textContent = "Status unavailable";
    toast(`Could not load credential status: ${error.message}`, true);
  }
}

async function saveCredentials() {
  const button = $("#save-credentials");
  const values = {};
  for (const [field, selector] of Object.entries(credentialFields)) {
    const value = $(selector).value.trim();
    if (value) values[field] = value;
  }
  button.disabled = true;
  try {
    await api("/api/dashboard/credentials", {
      method: "PUT",
      body: JSON.stringify({ values, clear: [] }),
    });
    for (const selector of Object.values(credentialFields)) {
      if ($(selector).type !== "number") $(selector).value = "";
    }
    $("#credential-state").textContent = "Saved · restart required";
    $("#credential-state").className = "save-state is-saved";
    toast("Credentials saved locally. Restart Vouch to apply them.");
    await loadCredentials();
  } catch (error) {
    $("#credential-state").textContent = "Save failed";
    $("#credential-state").className = "save-state is-error";
    toast(`Could not save credentials: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const button = $("#save-settings");
  button.disabled = true;
  try {
    const payload = await api("/api/dashboard/settings", {
      method: "PUT",
      body: JSON.stringify(collectSettings()),
    });
    state.settingsDirty = false;
    const badge = $("#settings-state");
    badge.textContent = "Saved · restart needed";
    badge.className = "save-state is-saved";
    toast(payload.message);
    await loadStatus();
  } catch (error) {
    const badge = $("#settings-state");
    badge.textContent = "Save failed";
    badge.className = "save-state is-error";
    toast(`Could not save settings: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

function accountCard(account) {
  const card = document.createElement("article");
  card.className = `account-card${account.enabled ? " is-enabled" : " is-paused"}`;
  const avatar = document.createElement("div");
  avatar.className = "account-avatar";
  avatar.textContent = account.username.charAt(0).toUpperCase();
  const identity = document.createElement("div");
  identity.className = "account-identity";
  const name = document.createElement("strong");
  name.textContent = `@${account.username}`;
  const status = document.createElement("span");
  status.textContent = account.enabled ? "Tracking enabled" : "Tracking paused";
  identity.append(name, status);
  const controls = document.createElement("div");
  controls.className = "account-controls";
  const switchLabel = document.createElement("label");
  switchLabel.className = "switch account-switch";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = account.enabled;
  input.setAttribute("aria-label", `Track @${account.username}`);
  const slider = document.createElement("span");
  switchLabel.append(input, slider);
  input.addEventListener("change", async () => {
    input.disabled = true;
    try {
      const payload = await api(`/api/selected-accounts/${encodeURIComponent(account.username)}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled: input.checked }),
      });
      state.accounts = payload;
      renderSelectedAccounts();
      toast(input.checked ? `@${account.username} is now tracked` : `@${account.username} is paused`);
      await loadStatus();
    } catch (error) {
      input.checked = !input.checked;
      toast(`Could not update @${account.username}: ${error.message}`, true);
    } finally {
      input.disabled = false;
    }
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "delete-account-button";
  remove.textContent = "Delete";
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Remove @${account.username} from Selected accounts?`)) return;
    remove.disabled = true;
    try {
      const payload = await api(`/api/selected-accounts/${encodeURIComponent(account.username)}`, { method: "DELETE" });
      state.accounts = payload;
      renderSelectedAccounts();
      toast(`@${account.username} was removed`);
      await loadStatus();
    } catch (error) {
      toast(`Could not delete @${account.username}: ${error.message}`, true);
      remove.disabled = false;
    }
  });
  controls.append(switchLabel, remove);
  card.append(avatar, identity, controls);
  return card;
}

function renderSelectedAccounts() {
  if (!state.accounts) return;
  $("#selected-accounts-enabled").checked = state.accounts.enabled;
  $("#selected-accounts-max-posts").value = state.accounts.max_posts_per_account;
  const list = $("#accounts-list");
  list.replaceChildren();
  for (const account of state.accounts.items) list.append(accountCard(account));
  $("#accounts-empty").hidden = state.accounts.items.length > 0;
  state.accountsDirty = false;
  const badge = $("#accounts-state");
  badge.textContent = `${state.accounts.items.filter((item) => item.enabled).length} active`;
  badge.className = "save-state is-saved";
}

async function loadSelectedAccounts() {
  try {
    state.accounts = await api("/api/selected-accounts");
    renderSelectedAccounts();
  } catch (error) {
    const badge = $("#accounts-state");
    badge.textContent = "Load failed";
    badge.className = "save-state is-error";
    toast(`Could not load Selected accounts: ${error.message}`, true);
  }
}

function markAccountsDirty() {
  state.accountsDirty = true;
  const badge = $("#accounts-state");
  badge.textContent = "Unsaved changes";
  badge.className = "save-state";
}

async function saveSelectedAccountSettings() {
  const button = $("#save-account-settings");
  button.disabled = true;
  try {
    state.accounts = await api("/api/selected-accounts", {
      method: "PUT",
      body: JSON.stringify({
        enabled: $("#selected-accounts-enabled").checked,
        max_posts_per_account: Number($("#selected-accounts-max-posts").value),
      }),
    });
    renderSelectedAccounts();
    toast("Selected-account source settings saved");
    await loadStatus();
  } catch (error) {
    toast(`Could not save Selected accounts: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

async function addSelectedAccount(event) {
  event.preventDefault();
  const dialog = $("#account-dialog");
  const input = $("#account-username");
  const username = input.value.trim();
  if (!username) {
    input.focus();
    return;
  }
  const button = $("#add-account-submit");
  button.disabled = true;
  try {
    state.accounts = await api("/api/selected-accounts", {
      method: "POST",
      body: JSON.stringify({ username }),
    });
    input.value = "";
    dialog.close();
    renderSelectedAccounts();
    toast("Account added and tracking enabled");
    await loadStatus();
  } catch (error) {
    toast(`Could not add account: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

function autoSizeComposer() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
}

function bindEvents() {
  $("#draft-action-form").addEventListener("submit", submitDraftAction);
  $("#draft-action-close").addEventListener("click", closeDraftAction);
  $("#draft-action-cancel").addEventListener("click", closeDraftAction);
  $("#draft-action-dialog").addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDraftAction();
  });
  $("#analyze-voice").addEventListener("click", analyzeVoice);
  $("#logout-account").addEventListener("click", logoutAccount);
  $("#close-settings").addEventListener("click", closeSettings);
  $("#settings-dialog").addEventListener("cancel", (event) => {
    event.preventDefault();
    closeSettings();
  });
  $("#profile-menu-button").addEventListener("click", () => toggleProfileMenu());
  $$('[data-profile-action]').forEach((button) => button.addEventListener("click", () => {
    toggleProfileMenu(false);
    if (button.dataset.profileAction === "settings") setPage("settings");
    if (button.dataset.profileAction === "logout") logoutAccount();
    if (button.dataset.profileAction === "voice") {
      window.location.href = "/login?mode=voice";
    }
  }));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".profile-menu-wrap")) toggleProfileMenu(false);
  });
  $$(".nav-button").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.page)));
  $$(".settings-nav-button").forEach((button) => button.addEventListener("click", () => setSettingsSection(button.dataset.settingsSection)));
  $("#refresh-button").addEventListener("click", async () => {
    await Promise.all([loadStatus(), loadChatHistory(), loadRecentDrafts(), loadSettings({ silent: true }), loadSelectedAccounts()]);
    toast("Dashboard refreshed");
  });
  $("#chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    autoSizeComposer();
    await sendCommand(text);
  });
  $("#chat-input").addEventListener("input", autoSizeComposer);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $$('[data-command]').forEach((button) => button.addEventListener("click", () => {
    const command = button.dataset.command;
    if (command === "/new ") {
      const input = $("#chat-input");
      input.value = command;
      input.focus();
      autoSizeComposer();
      return;
    }
    sendCommand(command);
  }));
  $("#settings-form").addEventListener("change", markSettingsDirty);
  $("#settings-form").addEventListener("submit", saveSettings);
  $("#save-credentials").addEventListener("click", saveCredentials);
  $("#interval-preset").addEventListener("change", updateCustomIntervalVisibility);
  $("#add-evergreen-topic").addEventListener("click", addEvergreenTopic);
  $("#evergreen-topic-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addEvergreenTopic();
    }
  });
  $("#selected-accounts-enabled").addEventListener("change", markAccountsDirty);
  $("#selected-accounts-max-posts").addEventListener("input", markAccountsDirty);
  $("#save-account-settings").addEventListener("click", saveSelectedAccountSettings);
  $("#open-add-account").addEventListener("click", () => {
    $("#account-dialog").showModal();
    window.setTimeout(() => $("#account-username").focus(), 0);
  });
  $("#account-form").addEventListener("submit", (event) => {
    const submitter = event.submitter;
    if (submitter && submitter.value === "cancel") return;
    addSelectedAccount(event);
  });
  $("#load-more-button").addEventListener("click", loadOlderPosts);
  $("#activity-scroll").addEventListener("scroll", () => {
    const scroll = $("#activity-scroll");
    if (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 220) loadOlderPosts();
  });
  $("#new-posts-button").addEventListener("click", () => {
    state.unseenPosts = 0;
    $("#new-posts-button").hidden = true;
    $("#activity-scroll").scrollTo({ top: 0, behavior: "smooth" });
  });
  window.addEventListener("beforeunload", (event) => {
    if (state.settingsDirty || state.accountsDirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
}

async function start() {
  bindEvents();
  await requireAccount();
  const query = new URLSearchParams(window.location.search);
  $("#desktop-pill").hidden = query.get("desktop") !== "1";
  setSettingsSection("general");
  await Promise.all([loadStatus(), loadInitialFeed(), loadChatHistory(), loadRecentDrafts()]);
  await loadAccountPanel();
  window.setInterval(loadStatus, 10000);
  window.setInterval(loadChatHistory, 3000);
  window.setInterval(loadRecentDrafts, 3000);
  window.setInterval(() => loadSettings({ silent: true }), 12000);
}

window.addEventListener("DOMContentLoaded", start);
