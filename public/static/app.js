const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send");
const historyList = document.getElementById("history");
let conversationId = null;
let isSubmitting = false;

async function readApiResponse(response) {
  const body = await response.text();
  let data;
  try {
    data = body ? JSON.parse(body) : {};
  } catch {
    throw new Error(`The assistant returned an invalid response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).filter(Boolean).join(" ")
      : typeof data.detail === "string" ? data.detail : "";
    throw new Error(detail || `The assistant request failed (HTTP ${response.status}).`);
  }
  return data;
}

function isChatResponse(data) {
  return data &&
    typeof data.conversation_id === "string" &&
    typeof data.answer === "string" &&
    typeof data.is_evidence_sufficient === "boolean" &&
    Array.isArray(data.citations) &&
    data.retrieval_trace &&
    typeof data.retrieval_trace === "object";
}

function setSubmitting(submitting) {
  isSubmitting = submitting;
  sendButton.disabled = submitting;
  questionInput.disabled = submitting;
  document.querySelectorAll(".example").forEach((button) => {
    button.disabled = submitting;
  });
}

function appendQuestion(text) {
  const node = document.createElement("div");
  node.className = "message question-message";
  node.textContent = text;
  messages.appendChild(node);
}

function appendInlineMarkdown(parent, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    const element = token.startsWith("**")
      ? document.createElement("strong")
      : token.startsWith("`")
        ? document.createElement("code")
        : document.createElement("em");
    element.textContent = token.startsWith("**")
      ? token.slice(2, -2)
      : token.slice(1, -1);
    parent.appendChild(element);
    cursor = match.index + token.length;
  }
  parent.appendChild(document.createTextNode(text.slice(cursor)));
}

function splitMarkdownTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function isMarkdownTableDivider(line) {
  const cells = splitMarkdownTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function appendMarkdownTable(parent, lines) {
  const rows = [splitMarkdownTableRow(lines[0]), ...lines.slice(1).map(splitMarkdownTableRow)];
  const columnCount = Math.max(...rows.map((row) => row.length));
  const wrapper = document.createElement("div");
  wrapper.className = "answer-table-wrap";
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", "Answer comparison table");
  const table = document.createElement("table");
  table.className = "answer-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  rows[0].forEach((value) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    appendInlineMarkdown(cell, value);
    headerRow.appendChild(cell);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.slice(1).forEach((row) => {
    const tr = document.createElement("tr");
    for (let index = 0; index < columnCount; index += 1) {
      const cell = document.createElement("td");
      appendInlineMarkdown(cell, row[index] || "");
      tr.appendChild(cell);
    }
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrapper.appendChild(table);
  parent.appendChild(wrapper);
}

function renderMarkdown(parent, markdown) {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index += 1;
      continue;
    }

    if (
      line.includes("|") &&
      index + 1 < lines.length &&
      isMarkdownTableDivider(lines[index + 1])
    ) {
      const tableLines = [line];
      index += 2;
      while (index < lines.length && lines[index].trim().includes("|")) {
        tableLines.push(lines[index].trim());
        index += 1;
      }
      appendMarkdownTable(parent, tableLines);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(`h${Math.min(heading[1].length + 1, 5)}`);
      appendInlineMarkdown(element, heading[2]);
      parent.appendChild(element);
      index += 1;
      continue;
    }

    const unorderedItem = line.match(/^[-*+]\s+(.+)$/);
    const orderedItem = line.match(/^\d+[.)]\s+(.+)$/);
    if (unorderedItem || orderedItem) {
      const list = document.createElement(unorderedItem ? "ul" : "ol");
      list.className = "answer-list";
      while (index < lines.length) {
        const current = lines[index].trim();
        const item = current.match(unorderedItem ? /^[-*+]\s+(.+)$/ : /^\d+[.)]\s+(.+)$/);
        if (!item) break;
        const li = document.createElement("li");
        appendInlineMarkdown(li, item[1]);
        list.appendChild(li);
        index += 1;
      }
      parent.appendChild(list);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4})\s+/.test(lines[index].trim()) &&
      !/^[-*+]\s+/.test(lines[index].trim()) &&
      !/^\d+[.)]\s+/.test(lines[index].trim()) &&
      !(lines[index].includes("|") && index + 1 < lines.length && isMarkdownTableDivider(lines[index + 1]))
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join(" "));
    parent.appendChild(paragraph);
  }
}

function appendAnswer(data) {
  const card = document.createElement("article");
  card.className = `message answer-card${data.is_evidence_sufficient ? "" : " insufficient"}`;
  const label = document.createElement("p");
  label.className = "answer-label";
  label.textContent = data.is_evidence_sufficient ? "Document-grounded answer" : "Insufficient evidence";
  const answer = document.createElement("div");
  answer.className = "answer-text";
  renderMarkdown(answer, data.answer);
  card.append(label, answer);

  if (data.citations?.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.textContent = "Sources";
    data.citations.forEach((citation) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const title = document.createElement("strong");
      title.textContent = `${citation.source}${citation.page_number ? ` · page ${citation.page_number}` : ""}`;
      const excerpt = document.createElement("p");
      excerpt.className = "source-excerpt";
      excerpt.textContent = citation.excerpt;
      item.append(title, excerpt);
      sources.appendChild(item);
    });
    card.appendChild(sources);
  }

  if (data.retrieval_trace) {
    const trace = data.retrieval_trace;
    const details = document.createElement("details");
    details.className = "retrieval-details";
    const summary = document.createElement("summary");
    summary.textContent =
      `Retrieval details · dense ${trace.dense_count ?? 0} · BM25 ${trace.lexical_count ?? 0}`;
    const list = document.createElement("ul");
    (trace.selected_chunks || []).forEach((chunk) => {
      const item = document.createElement("li");
      item.textContent = `${chunk.source || "Source"}${chunk.page_number ? ` · page ${chunk.page_number}` : ""} · ${chunk.chunk_id}`;
      list.appendChild(item);
    });
    details.append(summary, list);
    card.appendChild(details);
  }
  messages.appendChild(card);
  messages.scrollTop = messages.scrollHeight;
}

async function submitQuestion(text) {
  const query = text.trim();
  if (isSubmitting) return;
  if (!query) {
    questionInput.setCustomValidity("Please enter a question.");
    questionInput.reportValidity();
    return;
  }
  questionInput.setCustomValidity("");
  document.getElementById("welcome")?.remove();
  appendQuestion(query);
  questionInput.value = "";
  setSubmitting(true);
  const loading = document.createElement("article");
  loading.className = "message answer-card";
  loading.textContent = "Searching the restaurant documents…";
  messages.appendChild(loading);
  messages.scrollTop = messages.scrollHeight;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query, conversation_id: conversationId }),
    });
    const data = await readApiResponse(response);
    if (!isChatResponse(data)) {
      throw new Error("The assistant returned an invalid chat response.");
    }
    conversationId = data.conversation_id;
    loading.remove();
    appendAnswer(data);
    loadHistory();
  } catch (error) {
    loading.classList.add("error-text");
    loading.textContent = error instanceof TypeError
      ? `Could not reach the assistant at ${window.location.origin}. Check that the backend is running, then retry.`
      : error.message || "The assistant could not process that question.";
  } finally {
    setSubmitting(false);
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => submitQuestion(button.textContent));
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    document.getElementById("status").textContent =
      `${data.documents} documents · ${data.pages} pages · ${data.chunks} chunks`;
    const model = document.getElementById("model-status");
    model.textContent = data.gemini_configured
      ? "Gemini generation · hybrid retrieval"
      : "Offline evidence mode · hybrid retrieval";
  } catch {
    document.getElementById("status").textContent = "Index status unavailable.";
  }
}

async function loadHistory() {
  const response = await fetch("/api/chat/conversations");
  if (!response.ok) return;
  const items = await response.json();
  historyList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No conversations yet.";
    historyList.appendChild(empty);
    return;
  }
  items.slice().reverse().forEach((item) => {
    const button = document.createElement("button");
    button.className = "history-item";
    button.type = "button";
    button.textContent = item.title;
    button.addEventListener("click", async () => {
      const history = await fetch(`/api/chat/conversations/${encodeURIComponent(item.conversation_id)}`);
      if (!history.ok) return;
      conversationId = item.conversation_id;
      messages.replaceChildren();
      (await history.json()).forEach((turn) => {
        if (turn.role === "user") appendQuestion(turn.content);
        else if (turn.role === "assistant" && turn.response) appendAnswer(turn.response);
        else {
          const article = document.createElement("article");
          article.className = "message answer-card";
          article.textContent = turn.content;
          messages.appendChild(article);
        }
      });
    });
    historyList.appendChild(button);
  });
}

document.getElementById("reindex").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Re-indexing…";
  try {
    const response = await fetch("/api/admin/reindex", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Re-index failed.");
    button.textContent = `Done · ${data.documents_processed} scanned`;
    await loadHealth();
  } catch (error) {
    button.textContent = error.message;
  } finally {
    window.setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
    }, 2500);
  }
});

loadHealth();
loadHistory();
