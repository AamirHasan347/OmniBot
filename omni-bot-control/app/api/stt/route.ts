import { NextRequest, NextResponse } from "next/server";

// Same cache pattern as omni-brain so a dead key never blocks STT twice
const exhaustedKeys = new Map<string, number>();

function isKeyExhausted(key: string): boolean {
  const until = exhaustedKeys.get(key);
  if (!until) return false;
  if (Date.now() > until) {
    exhaustedKeys.delete(key);
    return false;
  }
  return true;
}

// Helper to prevent hanging requests
async function fetchWithTimeout(url: string, options: RequestInit, timeoutMs = 6000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    const lang = String(formData.get("lang") || "en");

    if (!file || file.size < 900) {
      return NextResponse.json(
        { text: "", error: "empty_audio" },
        { status: 200 },
      );
    }

    const keyPool: string[] = [];
    for (const k of [
      process.env.GROQ_API_KEY_1,
      process.env.GROQ_API_KEY_2,
      process.env.GROQ_API_KEY_3,
      process.env.GROQ_API_KEY,
    ]) {
      const t = (k || "").trim();
      if (t && !keyPool.includes(t)) keyPool.push(t);
    }

    if (keyPool.length === 0) {
      console.warn("[STT] No Groq API keys found in .env.local");
      return NextResponse.json(
        { text: "", error: "no_groq_key" },
        { status: 200 },
      );
    }

    let lastError = "";

    for (const apiKey of keyPool) {
      if (isKeyExhausted(apiKey)) continue;

      try {
        const groqForm = new FormData();
        groqForm.append("file", file, file.name || "speech.webm");
        groqForm.append("model", "whisper-large-v3-turbo");
        if (lang.startsWith("hi")) groqForm.append("language", "hi");

        const res = await fetchWithTimeout(
          "https://api.groq.com/openai/v1/audio/transcriptions",
          {
            method: "POST",
            headers: { Authorization: `Bearer ${apiKey}` },
            body: groqForm,
          },
          6000 // 6 second timeout per key
        );

        if (res.ok) {
          const data = await res.json();
          return NextResponse.json(
            { text: data?.text || "" },
            { status: 200 },
          );
        }

        lastError = `HTTP ${res.status}`;
        if (res.status === 429) exhaustedKeys.set(apiKey, Date.now() + 60_000);
        console.warn(`[STT] Groq ${res.status}:`, lastError);
      } catch (e: any) {
        lastError = e?.name === 'AbortError' ? 'Timeout' : (e?.message || "network_error");
        console.warn("[STT] key attempt failed:", lastError);
      }
    }

    return NextResponse.json(
      { text: "", error: `stt_failed: ${lastError}` },
      { status: 200 },
    );
  } catch (err: any) {
    console.error("[STT] outer error:", err);
    return NextResponse.json(
      { text: "", error: "stt_exception" },
      { status: 200 },
    );
  }
}