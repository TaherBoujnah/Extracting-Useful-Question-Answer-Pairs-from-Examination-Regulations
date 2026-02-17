const chat = document.getElementById("chat");
const msg = document.getElementById("msg");
const send = document.getElementById("send");
const track = document.getElementById("track");

function add(role, text) {
  const div = document.createElement("div");
  div.className = role;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function ask() {
  const q = msg.value.trim();
  if (!q) return;

  add("user", q);
  msg.value = "";

  const res = await fetch("http://127.0.0.1:8000/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({track: track.value, message: q})
  });

  const data = await res.json();

  if (data.type === "answer") {
    add("bot", data.answer + `\n\n(Score: ${data.score.toFixed(2)})`);
  } else if (data.type === "redirect") {
    add("bot",
      "Ich finde keine passende FAQ-Antwort in meiner Wissensbasis.\n" +
      "Bitte sende eine E-Mail an:\n" +
      `- Bachelor Informatik: ${data.emails.bsc_informatik}\n` +
      `- Master Informatik: ${data.emails.msc_informatik}\n` +
      `- Master AI & Data Science: ${data.emails.msc_ai_ds}`
    );
  } else {
    add("bot", "Fehler: " + JSON.stringify(data));
  }
}

send.addEventListener("click", ask);
msg.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
