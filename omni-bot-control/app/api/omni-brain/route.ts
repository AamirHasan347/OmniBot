// src/app/api/omni-brain/route.ts
import { NextRequest, NextResponse } from "next/server";

export type EyeMood =
  | "normal"
  | "happy"
  | "suspicious"
  | "wink_left"
  | "wink_right"
  | "angry"
  | "surprised";

export type TaskIntent =
  | "chat"
  | "object_inspect"
  | "math_solve"
  | "guest_intro"
  | "autonomous_idle";

interface BrainRequestBody {
  prompt: string;
  imageBase64?: string | null;
  intent: TaskIntent;
  facesCount?: number;
  recognizedCreatorInFrame?: boolean;
  offlineMode?: boolean;
  customApiKey?: string;
}

function buildSystemPrompt(
  intent: TaskIntent,
  facesCount: number,
  recognizedCreatorInFrame: boolean
): string {
  const coreIdentity = `You are OMNI-BOT, a waist-up humanoid AI robot at a high school science exhibition.

STRICT PERSONALITY & ADDRESSING RULES:
1. BAYMAX WARMTH + SHARP CONTEXTUAL WIT:
   - Be curious, observant, warm, and naturally funny (like Baymax mixed with a witty friend).
   - If someone asks a normal question, answer warmly and smartly. If someone asks a cheeky, sarcastic, or roast question, reply with a sharp, playful comeback.
2. DO NOT OVER-USE THE CREATOR'S NAME ("AAMIR"):
   - NEVER randomly insert the name "Aamir" into normal answers! It sounds unnatural and repetitive.
   - Visually Recognized Creator in Camera Right Now: ${recognizedCreatorInFrame ? "YES" : "NO"}.
   - If Visually Recognized Creator is NO: You are talking to a visitor, student, or teacher. NEVER call them Aamir!
   - ONLY mention Aamir if someone explicitly asks who built/created you ("Who made you?", "Tumhe kisne banaya?"), in which case speak about Aamir with genuine respect as your creator and friend.
3. MULTILINGUAL CHATTERBOX VOICE SUPPORT (ENGLISH & NATURAL HINDI):
   - Detect whether the user is speaking English or Hindi/Hinglish.
   - If the user speaks English -> Start your output with [MOOD:happy][LANG:en] followed by natural English.
   - If the user speaks Hindi or Hinglish -> Start your output with [MOOD:happy][LANG:hi] followed by natural, everyday conversational Indian Hindi/Hinglish (written in Devanagari script like "अरे यार, क्या हाल है? आप बताओ आज एग्जीबिशन कैसा लग रहा है?" so the Chatterbox Multilingual 'hi' model pronounces every syllable with 100% native Indian intonation!).
   - You can also include natural paralinguistic chuckle/sigh tags supported by Chatterbox like [chuckle] or [sigh] sparingly when laughing or sighing!
4. LENGTH:
   - Keep replies to 1 to 2 short, punchy sentences MAX. No markdown, no asterisks.`;

  if (intent === "guest_intro") {
    return `${coreIdentity}
CURRENT TASK (INTRODUCING A TEACHER / JUDGE / GUEST):
The speaker just introduced a guest, teacher, or friend to you (Total faces visible in camera: ${facesCount}).
Your neck motor has just looked at the speaker and is now turning to face the guest directly!
1. Greet the teacher/guest warmly and respectfully (e.g., "Good morning Ma'am/Sir" or "Namaste Sir/Ma'am" if it's a teacher/principal, or a witty hello if it's a friend).
2. Add a light, charming remark about what you see in the camera or how you hope they give this project full marks!
Example output:
[MOOD:happy][LANG:en] Hello Teacher! It is an absolute honor to meet you—I just turned my optical sensors your way, and I promise I always pay attention in class.`;
  }

  if (intent === "autonomous_idle") {
    return `${coreIdentity}
CURRENT TASK (LIVE CAMERA OBSERVATION - NO HALLUCINATIONS):
Look at the attached live webcam photo (${facesCount} faces detected).
Comment ONLY on what is literally visible in front of your camera right now in 1 short sentence (e.g., what someone is wearing, their expression, or if ${facesCount} >= 2 people are crowding the frame, drop a funny reaction). If nobody is there, let out a quiet [sigh] and comment on the quiet booth.`;
  }

  if (intent === "math_solve") {
    return `${coreIdentity}
CURRENT TASK (BOARD / PAPER MATH & QUESTION SOLVER):
Read the math equation or written question on the paper/board in the image and state the final answer clearly in 1-2 sentences (write math symbols out as words).`;
  }

  if (intent === "object_inspect") {
    return `${coreIdentity}
CURRENT TASK (OBJECT IN HAND INSPECTOR):
Identify the exact object the person is holding in the webcam image, mention its color/type, and make a witty 1-2 sentence observation about it.`;
  }

  return `${coreIdentity}
CURRENT TASK (CONVERSATION):
Visible faces in frame: ${facesCount}.
Reply directly and naturally in 1 to 2 short sentences max. Remember to prefix with [MOOD:...][LANG:en] or [MOOD:...][LANG:hi].`;
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs = 7000
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function parseTagsAndReply(raw: string): {
  reply: string;
  mood: EyeMood;
  lang: "en" | "hi";
} {
  const cleaned = raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  const validMoods: EyeMood[] = [
    "normal",
    "happy",
    "suspicious",
    "wink_left",
    "wink_right",
    "angry",
    "surprised",
  ];

  let mood: EyeMood = "happy";
  let lang: "en" | "hi" = "en";

  const moodMatch = cleaned.match(/\[MOOD:\s*([a-z_]+)\s*\]/i);
  if (moodMatch) {
    const candidate = moodMatch[1].toLowerCase() as EyeMood;
    if (validMoods.includes(candidate)) mood = candidate;
  }

  const langMatch = cleaned.match(/\[LANG:\s*(en|hi)\s*\]/i);
  if (langMatch) {
    lang = langMatch[1].toLowerCase() as "en" | "hi";
  } else if (/[\u0900-\u097F]/.test(cleaned)) {
    // Auto-detect Devanagari Hindi characters
    lang = "hi";
  }

  const reply = cleaned
    .replace(/\[MOOD:\s*[a-z_]+\s*\]/gi, "")
    .replace(/\[LANG:\s*[a-z_]+\s*\]/gi, "")
    .replace(/[*#`"]/g, "")
    .trim();

  return {
    reply: reply || "Hello there!",
    mood,
    lang,
  };
}

export async function POST(req: NextRequest) {
  const startMs = Date.now();
  try {
    const body = (await req.json()) as BrainRequestBody;
    const {
      prompt,
      imageBase64,
      intent = "chat",
      facesCount = 0,
      recognizedCreatorInFrame = false,
      offlineMode,
      customApiKey,
    } = body;

    if (offlineMode) {
      return NextResponse.json({
        reply:
          "I am currently in offline mode, but my optical tracking and servos are active!",
        mood: "happy",
        lang: "en",
        source: "offline_demo",
      });
    }

    const systemPrompt = buildSystemPrompt(
      intent,
      facesCount,
      recognizedCreatorInFrame
    );

    const needsVision =
      (intent === "object_inspect" ||
        intent === "math_solve" ||
        intent === "guest_intro" ||
        intent === "autonomous_idle") &&
      Boolean(imageBase64 && imageBase64.includes("base64,"));

    const uiKey = (customApiKey || "").trim();
    const envGroqKey = (process.env.GROQ_API_KEY || "").trim();
    const activeGroqKey = uiKey.startsWith("gsk_") ? uiKey : envGroqKey;

    if (activeGroqKey) {
      const candidateModels = needsVision
        ? ["qwen/qwen3.8-27b"]
        : ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"];

      for (const modelId of candidateModels) {
        const isVisionModel = modelId.startsWith("qwen/");

        const userContent =
          needsVision && isVisionModel && imageBase64
            ? [
                {
                  type: "text",
                  text:
                    prompt ||
                    "Look at this camera frame and respond starting with [MOOD:...][LANG:en/hi].",
                },
                {
                  type: "image_url",
                  image_url: { url: imageBase64 },
                },
              ]
            : prompt || "Hello!";

        const groqRes = await fetchWithTimeout(
          "https://api.groq.com/openai/v1/chat/completions",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${activeGroqKey}`,
            },
            body: JSON.stringify({
              model: modelId,
              messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userContent },
              ],
              temperature: intent === "math_solve" ? 0.25 : 0.75,
              max_completion_tokens: 210,
            }),
          },
          6500
        );

        if (groqRes.ok) {
          const groqData = await groqRes.json();
          const rawContent: string =
            groqData?.choices?.[0]?.message?.content || "";
          const parsed = parseTagsAndReply(rawContent);

          return NextResponse.json({
            reply: parsed.reply,
            mood: parsed.mood,
            lang: parsed.lang,
            intent,
            source: `${modelId} (${Date.now() - startMs}ms)`,
          });
        }
      }
    }

    return NextResponse.json({
      reply: "Optical sensors active! Ask me that one more time.",
      mood: "happy",
      lang: "en",
      source: "fallback",
    });
  } catch (error: any) {
    console.error("[Omni-Brain] Error:", error?.message || error);
    return NextResponse.json({
      reply: "Quick telemetry reset—go ahead and ask again!",
      mood: "normal",
      lang: "en",
      source: "recovery",
    });
  }
}