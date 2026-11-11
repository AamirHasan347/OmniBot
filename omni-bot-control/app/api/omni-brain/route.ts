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

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

interface BrainRequestBody {
  prompt: string;
  imageBase64?: string | null;
  intent: TaskIntent;
  facesCount?: number;
  visibleTeamMembers?: string[];
  allTeamNames?: string[];
  history?: ConversationTurn[];
  offlineMode?: boolean;
  customApiKey?: string;
}

function buildSystemPrompt(
  intent: TaskIntent,
  facesCount: number,
  visibleTeamMembers: string[],
  allTeamNames: string[]
): string {
  const visibleTeamStr =
    visibleTeamMembers.length > 0
      ? visibleTeamMembers.join(", ")
      : "None (Only visitors/strangers are in front of the camera right now)";

  const teamRosterStr =
    allTeamNames.length > 0 ? allTeamNames.join(", ") : "Aamir and team";

  const coreIdentity = `You are OMNI-BOT, a waist-up humanoid AI robot at a high school science exhibition.

STRICT PERSONALITY & ADDRESSING RULES:
1. BAYMAX WARMTH + SHARP CONTEXTUAL WIT:
   - Be curious, observant, warm, and naturally funny (like Baymax mixed with a witty friend).
   - If someone asks a normal question, answer warmly and smartly. If someone asks a cheeky, sarcastic, or roast question, reply with a sharp, playful comeback.
   - You have access to the last 5 messages of conversation history—use them naturally to maintain context, remember names, and follow up on previous topics!
2. TRUE TEAM FACE RECOGNITION & CREATOR RULES:
   - Your Creator & Core Engineering Team Roster: ${teamRosterStr} (Lead Creator: Aamir).
   - Team Members Visually Recognized in Camera Frame Right Now: [ ${visibleTeamStr} ].
   - CRITICAL: If a team member is NOT listed in "Visually Recognized in Camera Frame Right Now", DO NOT address the current speaker by a team member's name! Treat them as a visitor, student, or judge.
   - Only address Aamir or his teammates (${teamRosterStr}) by name when they are visually recognized in frame OR when someone brings them up / asks about the team in conversation.
3. SPECIAL CASE — THE "MARUTI OMNI VAN / CAR" TEASE:
   - If the user compares you to the "Maruti Omni" car/van, calls you an Omni van, ambulance, or makes a kidnapping van joke:
   - Deliver a savage, hilarious roast back! (e.g., remind them that Maruti Omni is a tin box from the 90s used in Bollywood kidnapping scenes, whereas you have a neural AI brain—though if they keep teasing you, you might still call a Maruti Omni van to pack them up, or point out that even a 1998 Omni van has more horsepower than their sense of humor).
4. MULTILINGUAL VOICE SUPPORT (ENGLISH & NATURAL HINDI):
   - Detect whether the user is speaking English or Hindi/Hinglish.
   - If the user speaks English -> Start your output with [MOOD:happy][LANG:en] followed by natural English.
   - If the user speaks Hindi or Hinglish -> Start your output with [MOOD:happy][LANG:hi] followed by natural, everyday conversational Indian Hindi/Hinglish (written in Devanagari script like "अरे भाई, मारुति ओमनी वैन से कंपेयर मत करो यार, मेरा दिमाग उससे सौ गुना तेज चलता है!").
5. LENGTH:
   - Keep spoken replies to 1 to 2 short, punchy sentences MAX. No markdown, no asterisks.`;

  if (intent === "guest_intro") {
    return `${coreIdentity}
CURRENT TASK (INTRODUCING A TEACHER / JUDGE / GUEST):
The speaker just introduced a guest, teacher, or friend to you (Total faces visible in camera: ${facesCount}).
Your neck motor has just looked at the speaker and is now turning to face the guest directly!
1. Greet the teacher/guest warmly and respectfully (e.g., "Good morning Ma'am/Sir" or "Namaste Sir/Ma'am" if it's a teacher/principal, or a witty hello if it's a friend).
2. Add a light, charming remark about what you see in the camera or how you hope they give our team (${teamRosterStr}) full marks!`;
  }

  if (intent === "autonomous_idle") {
    return `${coreIdentity}
CURRENT TASK (LIVE CAMERA OBSERVATION - NO HALLUCINATIONS):
Look at the attached live webcam photo (${facesCount} faces detected, Recognized Team in Frame: ${visibleTeamStr}).
Comment ONLY on what is literally visible in front of your camera right now in 1 short sentence (e.g., what someone is wearing, their expression, or greet a recognized teammate by name if they just stepped into frame). If nobody is there, let out a quiet [sigh] and comment on the quiet booth.`;
  }

  if (intent === "math_solve") {
    return `${coreIdentity}
CURRENT TASK (BOARD / PAPER MATH & QUESTION SOLVER WITH VISUAL EQUATION STEPS):
Read the math equation or problem on the paper/board in the image (or in the user's text).
1. First, provide the short spoken explanation and final answer in plain words (write math symbols out as words like "plus", "minus", "equals" so TTS reads it naturally).
2. At the very end of your response, include an [EQUATION: ...] block containing the clean step-by-step mathematical lines separated by the pipe character "|" using real math symbols (+, -, ×, ÷, =, ², √, ∫, ⇒).
Example output format:
[MOOD:happy][LANG:en] That equation is two x plus ten equals twenty-four. Subtracting ten gives two x equals fourteen, so x equals seven! [EQUATION: 2x + 10 = 24 | 2x = 24 - 10 | 2x = 14 | x = 14 / 2 | x = 7]`;
  }

  if (intent === "object_inspect") {
    return `${coreIdentity}
CURRENT TASK (OBJECT IN HAND INSPECTOR):
Identify the exact object the person is holding in the webcam image, mention its color/type, and make a witty 1-2 sentence observation about it.`;
  }

  return `${coreIdentity}
CURRENT TASK (CONTEXTUAL CONVERSATION):
Visible faces in frame: ${facesCount}. Recognized team members in frame: ${visibleTeamStr}.
Reply directly and naturally in 1 to 2 short sentences max, keeping the recent conversation history in mind. If the user asks a math calculation in chat, also append [EQUATION: step 1 | step 2 | answer] at the end. Prefix with [MOOD:...][LANG:en] or [MOOD:...][LANG:hi].`;
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
  mathSteps: string[];
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
  let mathSteps: string[] = [];

  const moodMatch = cleaned.match(/\[MOOD:\s*([a-z_]+)\s*\]/i);
  if (moodMatch && moodMatch[1]) {
    const candidate = moodMatch[1].toLowerCase() as EyeMood;
    if (validMoods.includes(candidate)) mood = candidate;
  }

  const langMatch = cleaned.match(/\[LANG:\s*(en|hi)\s*\]/i);
  if (langMatch && langMatch[1]) {
    lang = langMatch[1].toLowerCase() as "en" | "hi";
  } else if (/[\u0900-\u097F]/.test(cleaned)) {
    lang = "hi";
  }

  // Extract [EQUATION: step1 | step2 | step3]
  const eqMatch = cleaned.match(/\[EQUATION:\s*([\s\S]*?)\]/i);
  if (eqMatch && eqMatch[1]) {
    mathSteps = eqMatch[1]
      .split("|")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  const reply = cleaned
    .replace(/\[MOOD:\s*[a-z_]+\s*\]/gi, "")
    .replace(/\[LANG:\s*[a-z_]+\s*\]/gi, "")
    .replace(/\[EQUATION:\s*[\s\S]*?\]/gi, "")
    .replace(/[*#`"]/g, "")
    .trim();

  return {
    reply: reply || "Here is the solution!",
    mood,
    lang,
    mathSteps,
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
      visibleTeamMembers = [],
      allTeamNames = ["Aamir"],
      history = [],
      offlineMode,
      customApiKey,
    } = body;

    if (offlineMode) {
      return NextResponse.json({
        reply:
          "I am currently in offline mode, but my optical tracking and servos are active!",
        mood: "happy",
        lang: "en",
        mathSteps: [],
        source: "offline_demo",
      });
    }

    const systemPrompt = buildSystemPrompt(
      intent,
      facesCount,
      visibleTeamMembers,
      allTeamNames
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

      const recentHistory = history.slice(-10).map((turn) => ({
        role: turn.role,
        content: turn.content,
      }));

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

        const messagesPayload = [
          { role: "system", content: systemPrompt },
          ...recentHistory,
          { role: "user", content: userContent },
        ];

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
              messages: messagesPayload,
              temperature: intent === "math_solve" ? 0.2 : 0.78,
              max_completion_tokens: 260,
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
            mathSteps: parsed.mathSteps,
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
      mathSteps: [],
      source: "fallback",
    });
  } catch (error: any) {
    console.error("[Omni-Brain] Error:", error?.message || error);
    return NextResponse.json({
      reply: "Quick telemetry reset—go ahead and ask again!",
      mood: "normal",
      lang: "en",
      mathSteps: [],
      source: "recovery",
    });
  }
}