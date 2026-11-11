// src/app/api/stt/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const audioFile = formData.get("file") as Blob | null;
    const lang = (formData.get("lang") as string) || "en";

    if (!audioFile) {
      return NextResponse.json({ error: "No audio file" }, { status: 400 });
    }

    const groqKey = (process.env.GROQ_API_KEY || "").trim();
    if (!groqKey) {
      return NextResponse.json(
        { error: "Missing GROQ_API_KEY in .env.local" },
        { status: 500 }
      );
    }

    const upstreamForm = new FormData();
    upstreamForm.append("file", audioFile, "speech.webm");
    upstreamForm.append("model", "whisper-large-v3-turbo");
    upstreamForm.append("temperature", "0");
    upstreamForm.append("response_format", "json");
    if (lang === "hi-IN") {
      upstreamForm.append("language", "hi");
    }

    const res = await fetch(
      "https://api.groq.com/openai/v1/audio/transcriptions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${groqKey}`,
        },
        body: upstreamForm,
      }
    );

    if (!res.ok) {
      const errText = await res.text();
      console.warn("[STT] Groq Whisper error:", errText);
      return NextResponse.json({ text: "" }, { status: 200 });
    }

    const data = await res.json();
    const rawText = (data.text || "").trim();

    // Filter out common Whisper silence hallucinations ("Thank you.", "You", ".")
    const hallucinations = [
      "thank you.",
      "thank you",
      "thanks for watching.",
      "you",
      ".",
      "bye.",
      "subtitles by",
    ];
    if (
      rawText.length < 2 ||
      hallucinations.includes(rawText.toLowerCase())
    ) {
      return NextResponse.json({ text: "" });
    }

    return NextResponse.json({ text: rawText });
  } catch (err: any) {
    console.error("[STT] Exception:", err?.message || err);
    return NextResponse.json({ text: "" }, { status: 200 });
  }
}