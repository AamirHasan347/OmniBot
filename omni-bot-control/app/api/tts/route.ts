// src/app/api/tts/route.ts
import { NextRequest, NextResponse } from "next/server";

// Default Voice ID for ElevenLabs "Donovan" (or auto-discovered from user's VoiceLab)
let cachedDonovanVoiceId: string | null = null;

async function resolveDonovanVoiceId(apiKey: string): Promise<string> {
  if (cachedDonovanVoiceId) return cachedDonovanVoiceId;

  try {
    const res = await fetch("https://api.elevenlabs.io/v1/voices", {
      headers: { "xi-api-key": apiKey },
    });
    if (res.ok) {
      const data = await res.json();
      const voices: Array<{ voice_id: string; name: string }> = data.voices || [];

      // 1. Look for "Donovan" in the user's added voices
      const donovan = voices.find((v) =>
        v.name.toLowerCase().includes("donovan")
      );
      if (donovan) {
        cachedDonovanVoiceId = donovan.voice_id;
        return donovan.voice_id;
      }

      // 2. Otherwise pick Brian / Adam / Daniel (Deep resonant pre-made male voices)
      const deepMale = voices.find((v) =>
        ["brian", "adam", "daniel", "callum", "marcus"].some((m) =>
          v.name.toLowerCase().includes(m)
        )
      );
      if (deepMale) {
        cachedDonovanVoiceId = deepMale.voice_id;
        return deepMale.voice_id;
      }
    }
  } catch {
    // Fallback to standard deep male ID if lookup fails
  }

  // Fallback to "Brian" (deep, resonant narrator built into every ElevenLabs account)
  return "nPczCjzI2devNBz1zQrb";
}

export async function POST(req: NextRequest) {
  try {
    const { text, customElevenKey, customVoiceId } = await req.json();
    const cleanText = (text || "").trim();
    if (!cleanText) {
      return NextResponse.json({ error: "Empty text" }, { status: 400 });
    }

    const elevenKey = (
      customElevenKey ||
      process.env.ELEVENLABS_API_KEY ||
      ""
    ).trim();

    // 1. Try ElevenLabs Turbo v2.5 ("Donovan" Voice) if API key is present
    if (elevenKey) {
      const voiceId =
        (customVoiceId || "").trim() ||
        process.env.ELEVENLABS_VOICE_ID ||
        (await resolveDonovanVoiceId(elevenKey));

      const elevenUrl = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream?optimize_streaming_latency=3`;
      const elevenRes = await fetch(elevenUrl, {
        method: "POST",
        headers: {
          "xi-api-key": elevenKey,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text: cleanText,
          model_id: "eleven_turbo_v2_5",
          voice_settings: {
            stability: 0.55,
            similarity_boost: 0.8,
            style: 0.35,
            use_speaker_boost: true,
          },
        }),
      });

      if (elevenRes.ok && elevenRes.body) {
        return new NextResponse(elevenRes.body, {
          status: 200,
          headers: {
            "Content-Type": "audio/mpeg",
            "X-Voice-Engine": "ElevenLabs-Donovan",
          },
        });
      } else {
        const errText = await elevenRes.text();
        console.warn("[TTS] ElevenLabs warning, falling back:", errText);
      }
    }

    // 2. Free Zero-Key Neural Male Voice Stream (StreamElements / AWS Polly Matthew/Brian)
    // Gives a natural, consistent male voice even without an ElevenLabs key!
    const encoded = encodeURIComponent(cleanText.slice(0, 280));
    const freeTtsUrl = `https://api.streamelements.com/kappa/v2/speech?voice=Brian&text=${encoded}`;
    const freeRes = await fetch(freeTtsUrl);

    if (freeRes.ok && freeRes.body) {
      return new NextResponse(freeRes.body, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "X-Voice-Engine": "Free-Neural-Male",
        },
      });
    }

    return NextResponse.json(
      { error: "Use browser TTS fallback" },
      { status: 503 }
    );
  } catch (err) {
    console.error("[TTS] Error:", err);
    return NextResponse.json(
      { error: "TTS failure" },
      { status: 500 }
    );
  }
}