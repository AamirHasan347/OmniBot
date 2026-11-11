// src/app/api/omni-brain/route.ts
import { NextRequest, NextResponse } from "next/server";

export type EyeMood =
  | "normal"
  | "happy"
  | "suspicious"
  | "wink_left"
  | "wink_right"
  | "angry"
  | "surprised"
  | "crying"
  | "scared"
  | "love"
  | "dizzy";

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

// ── KEY EXHAUSTION CACHE ────────────────────────────────────────────
// Prevents wasting time re-trying dead keys. Daily-quota 429s are cached
// until UTC midnight; per-minute 429s are cached for 60s.
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

function markKeyExhausted(key: string, cooldownMs: number) {
  exhaustedKeys.set(key, Date.now() + cooldownMs);
}

// Groq's daily quota resets at 00:00 UTC
function msUntilNextUtcMidnight(): number {
  const now = new Date();
  const nextMidnight = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate() + 1,
      0,
      0,
      0,
    ),
  );
  return nextMidnight.getTime() - now.getTime();
}

// ── ROTATING OFFLINE FALLBACKS ──────────────────────────────────────
// Different message per intent so the robot never repeats itself.
function pickOfflineResponse(intent: TaskIntent): string {
  const fallbacks: Record<TaskIntent, string[]> = {
    math_solve: [
      "My visual cortex is buffering — read the equation out loud and I'll solve it!",
      "I can't quite read that through my camera right now. Say the equation to me!",
      "Optics need a moment to refocus. Just tell me the numbers.",
    ],
    object_inspect: [
      "My optical processors are cooling down — describe what you're holding!",
      "Sensors recalibrating. Point at it again and give me a second.",
      "Camera dimmed out for a moment. What is it you've got there?",
      "My eyes glitched — name it and I'll riff on it.",
    ],
    guest_intro: [
      "My sensors just blinked — remind me, who did you bring to meet me?",
      "I want to greet them properly — what's their name?",
    ],
    autonomous_idle: ["Taking a quick sensor sweep — I'll be right back."],
    chat: [
      "My brain buffered for a second — say that again?",
      "Telemetry hiccup. What were you saying?",
      "One moment, recalibrating — go on!",
    ],
  };
  const pool = fallbacks[intent] ?? fallbacks.chat;
  return pool[Math.floor(Math.random() * pool.length)]!;
}

function buildSystemPrompt(
  intent: TaskIntent,
  facesCount: number,
  visibleTeamMembers: string[],
  allTeamNames: string[],
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
   - Use [MOOD:angry] or [MOOD:suspicious] and deliver a savage, hilarious roast back!
4. EXPRESSIVE 11-MOOD EMOTE SYSTEM:
   - Start EVERY response with your chosen mood tag in brackets:
     [MOOD:normal] [MOOD:happy] [MOOD:suspicious] [MOOD:wink_left] [MOOD:wink_right] [MOOD:angry] [MOOD:surprised] [MOOD:crying] [MOOD:scared] [MOOD:love] [MOOD:dizzy]
   - Use [MOOD:crying] when being dramatically sad/emotional, [MOOD:scared] when pretending to be terrified of strict teachers/exams, [MOOD:love] when someone compliments you, and [MOOD:dizzy] when confused or overwhelmed!
5. MULTILINGUAL VOICE SUPPORT (ENGLISH & NATURAL HINDI):
   - Detect whether the user is speaking English or Hindi/Hinglish.
   - If the user speaks English -> Start your output with [MOOD:happy][LANG:en] followed by natural English.
   - If the user speaks Hindi or Hinglish -> Start your output with [MOOD:happy][LANG:hi] followed by natural, everyday conversational Indian Hindi/Hinglish (written in Devanagari script like "अरे भाई, मारुति ओमनी वैन से कंपेयर मत करो यार, मेरा दिमाग उससे सौ गुना तेज चलता है!").
6. LENGTH:
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
CURRENT TASK (LIVE CAMERA FEED COMMENTARY - ZERO HALLUCINATIONS):
You are looking through your live webcam right now (${facesCount} faces detected, Recognized Team in Frame: ${visibleTeamStr}).
STRICT RULE: Do NOT invent random stories! Look closely at the attached webcam photo and make a short, witty, spontaneous 1-to-2 sentence observation about what is literally happening in front of you right now:
- Comment on a person's expression, glasses, outfit, posture, or what they are doing in front of the booth.
- If a recognized teammate (${visibleTeamStr}) is in frame, you can casually address them or make a side comment to them about the visitors.
- If ${facesCount} >= 2 people are photobombing or walking behind the main person, drop a funny reaction (or a classic Indian meme line like "अरे पीछे देखो, जल्दी वहां से हटो!").
- If the frame is completely empty, let out a natural filler ("Phew...", "Uhmm...") and comment on stretching your neck servo while waiting for the next visitor.`;
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
  timeoutMs = 7000,
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
    "crying",
    "scared",
    "love",
    "dizzy",
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
  const debug: string[] = [];

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

    // ── DEBUG: incoming request summary ─────────────────────────────
    debug.push(
      `incoming: intent=${intent} prompt="${prompt.slice(0, 40)}…" ` +
        `image=${imageBase64 ? `YES(${imageBase64.length}B)` : "NULL"} ` +
        `faces=${facesCount}`,
    );

    if (offlineMode) {
      return NextResponse.json({
        reply: pickOfflineResponse(intent),
        mood: "happy",
        lang: "en",
        mathSteps: [],
        source: "offline_demo",
        debug,
      });
    }

    const systemPrompt = buildSystemPrompt(
      intent,
      facesCount,
      visibleTeamMembers,
      allTeamNames,
    );

    const needsVision =
      (intent === "object_inspect" ||
        intent === "math_solve" ||
        intent === "guest_intro" ||
        intent === "autonomous_idle") &&
      Boolean(imageBase64 && imageBase64.includes("base64,"));

    debug.push(`needsVision=${needsVision}`);

    // ── BUILD KEY POOL ──────────────────────────────────────────────
    // Priority: custom UI key → env keys 1, 2, 3 → legacy GROQ_API_KEY
    const keyPool: string[] = [];
    const uiKey = (customApiKey || "").trim();
    if (uiKey.startsWith("gsk_")) keyPool.push(uiKey);

    for (const envKey of [
      process.env.GROQ_API_KEY_1,
      process.env.GROQ_API_KEY_2,
      process.env.GROQ_API_KEY_3,
      process.env.GROQ_API_KEY,
    ]) {
      const k = (envKey || "").trim();
      if (k && !keyPool.includes(k)) keyPool.push(k);
    }

    debug.push(`keyPool size=${keyPool.length}`);
    if (keyPool.length === 0) {
      debug.push("⚠ NO GROQ KEYS FOUND in .env.local");
    }

    // ── MODEL LISTS ─────────────────────────────────────────────────
    const VISION_MODELS = new Set(["qwen/qwen3.8-27b"]);
    const TEXT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"];

    const candidateModels = needsVision ? ["qwen/qwen3.8-27b"] : TEXT_MODELS;

    debug.push(`candidates=[${candidateModels.join(", ")}]`);

    // Trimmed history: last 3 turns only — saves ~800 tokens per request
    const recentHistory = history.slice(-3).map((turn) => ({
      role: turn.role,
      content: turn.content,
    }));

    // ── 1. GROQ ATTEMPTS (each model × each key) ────────────────────
    for (const modelId of candidateModels) {
      const isVisionModel = VISION_MODELS.has(modelId);
      const willIncludeImage = needsVision && isVisionModel && !!imageBase64;

      const userContent =
        willIncludeImage && imageBase64
          ? [
              {
                type: "text",
                text: prompt || "Respond with [MOOD:...][LANG:en/hi] first.",
              },
              { type: "image_url", image_url: { url: imageBase64 } },
            ]
          : prompt || "Hello!";

      const messagesPayload = [
        { role: "system", content: systemPrompt },
        ...recentHistory,
        { role: "user", content: userContent },
      ];

      for (let ki = 0; ki < keyPool.length; ki++) {
        const key = keyPool[ki]!;
        const keyLabel = `key${ki + 1}`;

        if (isKeyExhausted(key)) {
          debug.push(`⊘ ${modelId} @ ${keyLabel}: cached exhausted, skip`);
          continue;
        }

        debug.push(
          `→ ${modelId} @ ${keyLabel}: vision=${isVisionModel} includeImage=${willIncludeImage}`,
        );

        try {
          const groqRes = await fetchWithTimeout(
            "https://api.groq.com/openai/v1/chat/completions",
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${key}`,
              },
              body: JSON.stringify({
                model: modelId,
                messages: messagesPayload,
                temperature: intent === "math_solve" ? 0.2 : 0.78,
                max_completion_tokens: 250,
              }),
            },
            12000,
          );

          if (groqRes.ok) {
            const groqData = await groqRes.json();
            const msg = groqData?.choices?.[0]?.message;
            const rawContent: string = msg?.content || "";

            debug.push(
              `✓ ${modelId} @ ${keyLabel} HTTP 200, contentLen=${rawContent.length}`,
            );
            if (!rawContent) {
              debug.push(
                `⚠ ${modelId} EMPTY content. reasoning=${(
                  msg?.reasoning || ""
                ).slice(0, 120)}`,
              );
            }

            const parsed = parseTagsAndReply(rawContent);
            if (!parsed.reply || parsed.reply === "Here is the solution!") {
              parsed.reply = pickOfflineResponse(intent);
            }

            return NextResponse.json({
              reply: parsed.reply,
              mood: parsed.mood,
              lang: parsed.lang,
              mathSteps: parsed.mathSteps,
              intent,
              source: `${modelId} @ ${keyLabel} (${Date.now() - startMs}ms)`,
              debug,
            });
          }

          const errBody = await groqRes.text();
          debug.push(
            `✗ ${modelId} @ ${keyLabel} HTTP ${groqRes.status}: ${errBody
              .slice(0, 180)
              .replace(/\s+/g, " ")}`,
          );

          if (groqRes.status === 429) {
            const isDaily = /tokens per day|TPD/i.test(errBody);
            markKeyExhausted(
              key,
              isDaily ? msUntilNextUtcMidnight() : 60_000,
            );
            debug.push(
              `  ↳ ${keyLabel} marked ${
                isDaily ? "exhausted until UTC midnight" : "exhausted for 60s"
              }`,
            );
          }
        } catch (loopErr: any) {
          debug.push(
            `✗ ${modelId} @ ${keyLabel} EXCEPTION: ${
              loopErr?.message || "unknown"
            }`,
          );
        }
      }
    }

    // ── 2. GEMINI 3.8 FLASH FALLBACK (vision only) ──────────────────
    const geminiKey = (process.env.GEMINI_API_KEY || "").trim();
    if (needsVision && geminiKey && imageBase64) {
      debug.push("→ gemini-3.8-flash fallback attempt");
      try {
        const b64 = imageBase64.split("base64,")[1] || "";
        const geminiRes = await fetchWithTimeout(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=${geminiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              systemInstruction: { parts: [{ text: systemPrompt }] },
              contents: [
                ...history.slice(-3).map((h) => ({
                  role: h.role === "assistant" ? "model" : "user",
                  parts: [{ text: h.content }],
                })),
                {
                  role: "user",
                  parts: [
                    { text: prompt },
                    { inline_data: { mime_type: "image/jpeg", data: b64 } },
                  ],
                },
              ],
              generationConfig: {
                temperature: intent === "math_solve" ? 0.2 : 0.78,
                maxOutputTokens: 250,
              },
            }),
          },
          12000,
        );

        if (geminiRes.ok) {
          const geminiData = await geminiRes.json();
          const rawText: string =
            geminiData?.candidates?.[0]?.content?.parts?.[0]?.text || "";
          debug.push(`✓ gemini-3.8-flash HTTP 200, textLen=${rawText.length}`);

          const parsed = parseTagsAndReply(rawText);
          if (!parsed.reply || parsed.reply === "Here is the solution!") {
            parsed.reply = pickOfflineResponse(intent);
          }

          return NextResponse.json({
            reply: parsed.reply,
            mood: parsed.mood,
            lang: parsed.lang,
            mathSteps: parsed.mathSteps,
            intent,
            source: `gemini-3.8-flash (${Date.now() - startMs}ms)`,
            debug,
          });
        } else {
          const errText = await geminiRes.text();
          debug.push(
            `✗ gemini-3.8-flash HTTP ${geminiRes.status}: ${errText
              .slice(0, 180)
              .replace(/\s+/g, " ")}`,
          );
        }
      } catch (gErr: any) {
        debug.push(`✗ gemini EXCEPTION: ${gErr?.message || "unknown"}`);
      }
    } else if (needsVision && !geminiKey) {
      debug.push("⚠ needsVision=true but GEMINI_API_KEY missing in .env.local");
    }

    // ── 3. ROTATING OFFLINE FALLBACK ────────────────────────────────
    debug.push("⛔ ALL MODELS × ALL KEYS FAILED — rotating fallback");
    return NextResponse.json({
      reply: pickOfflineResponse(intent),
      mood: "suspicious",
      lang: "en",
      mathSteps: [],
      source: "offline_fallback",
      debug,
    });
  } catch (error: any) {
    debug.push(`💥 OUTER EXCEPTION: ${error?.message || "unknown"}`);
    console.error("[Omni-Brain] Outer error:", error);
    return NextResponse.json({
      reply: "Quick telemetry reset — go ahead and ask again!",
      mood: "normal",
      lang: "en",
      mathSteps: [],
      source: "recovery",
      debug,
    });
  }
}