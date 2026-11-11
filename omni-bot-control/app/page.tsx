// src/app/page.tsx
"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

type EyeMood =
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

type TaskIntent =
  | "chat"
  | "object_inspect"
  | "math_solve"
  | "guest_intro"
  | "autonomous_idle";

interface ChatLog {
  id: string;
  sender: "USER" | "OMNI-BOT";
  text: string;
  mathSteps?: string[];
  mood?: EyeMood;
  intent?: TaskIntent;
  latencyMs?: number;
  timestamp: string;
}

interface TrackedFace {
  centerX: number;
  centerY: number;
  panAngle: number;
  area: number;
  matchedName: string | null;
}

interface TeamMemberProfile {
  id: string;
  name: string;
  role: string;
  descriptor: number[] | null;
}

const DEFAULT_TEAM_ROSTER: TeamMemberProfile[] = [
  { id: "member-1", name: "Aamir", role: "Lead Creator", descriptor: null },
  { id: "member-2", name: "Teammate 2", role: "Team Member", descriptor: null },
  { id: "member-3", name: "Teammate 3", role: "Team Member", descriptor: null },
  { id: "member-4", name: "Teammate 4", role: "Team Member", descriptor: null },
  { id: "member-5", name: "Teammate 5", role: "Team Member", descriptor: null },
];

function euclideanDistance128(
  descA: number[],
  descB: Float32Array | number[],
): number {
  if (descA.length !== descB.length) return 999;
  let sum = 0;
  for (let i = 0; i < descA.length; i++) {
    const valA = descA[i] ?? 0;
    const valB = descB[i] ?? 0;
    const d = valA - valB;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function parsePhysicalVoiceReflex(transcript: string): {
  targetAngle: number | null;
  mood: EyeMood;
  pureCommandReply?: string;
} {
  const lower = transcript.toLowerCase().trim();

  if (
    lower.includes("look left") ||
    lower.includes("look to your left") ||
    lower.includes("look to ur left") ||
    lower.includes("turn left") ||
    lower.includes("left dekho")
  ) {
    return {
      targetAngle: 145,
      mood: "surprised",
      pureCommandReply:
        lower.split(" ").length <= 6
          ? "Turned to my left! What are we looking at?"
          : undefined,
    };
  }

  if (
    lower.includes("look right") ||
    lower.includes("look to your right") ||
    lower.includes("look to ur right") ||
    lower.includes("turn right") ||
    lower.includes("right dekho")
  ) {
    return {
      targetAngle: 35,
      mood: "surprised",
      pureCommandReply:
        lower.split(" ").length <= 6 ? "Looking to my right now!" : undefined,
    };
  }

  if (
    lower.includes("look back") ||
    lower.includes("peeche dekho") ||
    lower.includes("piche dekho") ||
    lower.includes("turn around")
  ) {
    return {
      targetAngle: 158,
      mood: "dizzy",
      pureCommandReply:
        "My neck servo maxes out at one-sixty degrees—any further and I will unscrew my own head!",
    };
  }

  if (
    lower.includes("look center") ||
    lower.includes("look straight") ||
    lower.includes("saamne dekho") ||
    lower.includes("samne dekho")
  ) {
    return {
      targetAngle: 90,
      mood: "happy",
      pureCommandReply:
        lower.split(" ").length <= 5 ? "Centered and ready!" : undefined,
    };
  }

  return { targetAngle: null, mood: "normal" };
}

function detectUserIntent(text: string): TaskIntent {
  const lower = text.toLowerCase();

  const introKeywords = [
    "meet my",
    "introduce",
    "this is my",
    "say hello to",
    "say hi to",
    "say namaste to",
    "inse milo",
    "ye mere",
    "yeh mere",
    "teacher",
    "sir se milo",
    "mam se milo",
    "ma'am",
    "principal",
  ];
  if (introKeywords.some((k) => lower.includes(k))) {
    return "guest_intro";
  }

  const mathKeywords = [
    "solve",
    "equation",
    "math",
    "calculate",
    "paper",
    "board",
    "written",
    "read this",
    "answer this",
    "formula",
    "problem",
    "integral",
    "derivative",
    "algebra",
    "hisab",
    "sawal",
    "plus",
    "minus",
    "multiply",
    "divide",
  ];
  if (mathKeywords.some((k) => lower.includes(k))) {
    return "math_solve";
  }

  const objectKeywords = [
    "holding",
    "in my hand",
    "what is this",
    "what am i showing",
    "can you see",
    "what do you see",
    "look at this",
    "inspect",
    "object",
    "wearing",
    "haath mein",
    "hath me",
    "ye kya hai",
    "yeh kya hai",
    "pakda hai",
  ];
  if (objectKeywords.some((k) => lower.includes(k))) {
    return "object_inspect";
  }

  return "chat";
}

export default function OmniBotMissionControl() {
  // ---------------------------------------------------------------------------
  // 1. CONNECTION, TELEMETRY & TEAM ROSTER STATE
  // ---------------------------------------------------------------------------
  const [wsUrl, setWsUrl] = useState<string>(
    process.env.NEXT_PUBLIC_ESP32_WS_URL || "ws://192.168.10.44:81",
  );
  const [chatterboxUrl, setChatterboxUrl] = useState<string>(
    "http://127.0.0.1:4123/tts",
  );
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [ttsEngineStatus, setTtsEngineStatus] =
    useState<string>("Neural TTS (4123)");
  const [showDebugDrawer, setShowDebugDrawer] = useState<boolean>(false);
  const [customApiKey, setCustomApiKey] = useState<string>("");

  const [panAngle, setPanAngle] = useState<number>(90);
  const [currentMood, setCurrentMood] = useState<EyeMood>("normal");
  const [vuVolume, setVuVolume] = useState<number>(0);

  const [wsPacketsSent, setWsPacketsSent] = useState<number>(0);
  const [lastWsPayload, setLastWsPayload] = useState<string>(
    '{"pan":90,"mood":"normal","volume":0}',
  );
  const [lastAiLatency, setLastAiLatency] = useState<number>(0);

  const [teamRoster, setTeamRoster] =
    useState<TeamMemberProfile[]>(DEFAULT_TEAM_ROSTER);
  const [faceIdModelReady, setFaceIdModelReady] = useState<boolean>(false);
  const [visibleTeamNames, setVisibleTeamNames] = useState<string[]>([]);
  const [enrollingSlotIdx, setEnrollingSlotIdx] = useState<number | null>(null);

  const [autoTrackFace, setAutoTrackFace] = useState<boolean>(true);
  const [passiveLifeHabits, setPassiveLifeHabits] = useState<boolean>(true);
  const [crowdControlEnabled, setCrowdControlEnabled] = useState<boolean>(true);
  // Enabled by default! Every 28s of silence, Omni-Bot comments on the live camera feed:
  const [spontaneousMonologue, setSpontaneousMonologue] =
    useState<boolean>(true);
  const [invertPanDirection, setInvertPanDirection] = useState<boolean>(false);
  const [offlineMode, setOfflineMode] = useState<boolean>(false);

  // ---------------------------------------------------------------------------
  // 2. A/V DEVICES
  // ---------------------------------------------------------------------------
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [audioInDevices, setAudioInDevices] = useState<MediaDeviceInfo[]>([]);
  const [micLang, setMicLang] = useState<"en-IN" | "hi-IN">("en-IN");
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [selectedMicId, setSelectedMicId] = useState<string>("");

  // ---------------------------------------------------------------------------
  // 3. VISION, GROQ WHISPER VAD MIC & AI PIPELINE STATE
  // ---------------------------------------------------------------------------
  const [visionReady, setVisionReady] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [facesDetected, setFacesDetected] = useState<number>(0);

  const [micActive, setMicActive] = useState<boolean>(false);
  const [micVolumeLevel, setMicVolumeLevel] = useState<number>(0);
  const [isRecordingSpeech, setIsRecordingSpeech] = useState<boolean>(false);
  const [micStatusHint, setMicStatusHint] = useState<string>(
    "Click 'Start Always-On Mic' — Works with Built-in MacBook Mic or Wired Mic!",
  );
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [activeIntentBadge, setActiveIntentBadge] =
    useState<TaskIntent>("chat");
  const [promptInput, setPromptInput] = useState<string>("");

  const [chatLogs, setChatLogs] = useState<ChatLog[]>([
    {
      id: "init-1",
      sender: "OMNI-BOT",
      text: "Systems online! Added Crying, Scared, Love, and Dizzy emotes, Red Angry Eye/Ring Overdrive, and Groq Whisper Hardware Mic!",
      mood: "normal",
      intent: "chat",
      timestamp: "READY",
    },
  ]);

  // ---------------------------------------------------------------------------
  // 4. MUTABLE REFS
  // ---------------------------------------------------------------------------
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const faceDetectorRef = useRef<FaceDetector | null>(null);
  const faceApiRef = useRef<any>(null);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);

  const teamRosterRef = useRef<TeamMemberProfile[]>(DEFAULT_TEAM_ROSTER);
  const recognizedSpatialRef = useRef<Array<{ normX: number; name: string }>>(
    [],
  );
  const visibleTeamNamesRef = useRef<string[]>([]);
  const trackedFacesRef = useRef<TrackedFace[]>([]);
  const chatLogsRef = useRef<ChatLog[]>([]);

  const telemetryRef = useRef<{ pan: number; mood: EyeMood; volume: number }>({
    pan: 90,
    mood: "normal",
    volume: 0,
  });
  const lastSentRef = useRef<{ pan: number; mood: EyeMood; volume: number }>({
    pan: -1,
    mood: "normal",
    volume: -1,
  });

  const autoTrackRef = useRef<boolean>(true);
  const invertPanRef = useRef<boolean>(false);
  const micActiveRef = useRef<boolean>(false);
  const micLangRef = useRef<"en-IN" | "hi-IN">("en-IN");
  const crowdControlEnabledRef = useRef<boolean>(true);
  const isSpeakingRef = useRef<boolean>(false);
  const isThinkingRef = useRef<boolean>(false);
  const manualNeckHoldUntilRef = useRef<number>(0);
  const lastCrowdAlertMsRef = useRef<number>(0);
  const crowdOverloadStartMsRef = useRef<number>(0);

  const facesCountRef = useRef<number>(0);
  const lastInteractionMsRef = useRef<number>(0);
  const speakingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Hardware Audio Stream & VAD Recorder Refs
  const micStreamRef = useRef<MediaStream | null>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const vadAnimFrameRef = useRef<number | null>(null);

  useEffect(() => {
    chatLogsRef.current = chatLogs;
  }, [chatLogs]);

  useEffect(() => {
    teamRosterRef.current = teamRoster;
  }, [teamRoster]);

  useEffect(() => {
    micLangRef.current = micLang;
  }, [micLang]);

  useEffect(() => {
    lastInteractionMsRef.current = Date.now();
    try {
      const savedRoster = localStorage.getItem("omnibot_team_roster_v2");
      if (savedRoster) {
        const parsed = JSON.parse(savedRoster) as TeamMemberProfile[];
        if (Array.isArray(parsed) && parsed.length === 5) {
          setTeamRoster(parsed);
          teamRosterRef.current = parsed;
        }
      }
    } catch {
      // Ignore storage error
    }
  }, []);

  useEffect(() => {
    autoTrackRef.current = autoTrackFace;
  }, [autoTrackFace]);

  useEffect(() => {
    invertPanRef.current = invertPanDirection;
  }, [invertPanDirection]);

  useEffect(() => {
    crowdControlEnabledRef.current = crowdControlEnabled;
  }, [crowdControlEnabled]);

  // ---------------------------------------------------------------------------
  // 5. LOAD 128-D NEURAL FACE RECOGNITION MODELS (@vladmandic/face-api)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let isMounted = true;
    async function initFaceIdModels() {
      try {
        const faceapi = await import("@vladmandic/face-api");
        const MODEL_URL =
          "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);
        if (isMounted) {
          faceApiRef.current = faceapi;
          setFaceIdModelReady(true);
        }
      } catch (err) {
        console.error("Failed to load 128-D Face ID models:", err);
      }
    }
    initFaceIdModels();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      const faceapi = faceApiRef.current;
      const video = videoRef.current;
      if (!faceapi || !video || video.readyState < 2) return;

      const enrolledMembers = teamRosterRef.current.filter(
        (m) => m.descriptor !== null && m.descriptor.length === 128,
      );
      if (enrolledMembers.length === 0) {
        recognizedSpatialRef.current = [];
        visibleTeamNamesRef.current = [];
        setVisibleTeamNames([]);
        return;
      }

      try {
        const detections = await faceapi
          .detectAllFaces(
            video,
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 224,
              scoreThreshold: 0.5,
            }),
          )
          .withFaceLandmarks(true)
          .withFaceDescriptors();

        const width = video.videoWidth || 640;
        const newMatches: Array<{ normX: number; name: string }> = [];
        const matchedNamesSet = new Set<string>();

        for (const det of detections) {
          const box = det.detection.box;
          const normX = (box.x + box.width / 2) / width;

          let bestName: string | null = null;
          let bestDist = 0.52;

          for (const member of enrolledMembers) {
            if (!member.descriptor) continue;
            const dist = euclideanDistance128(
              member.descriptor,
              det.descriptor,
            );
            if (dist < bestDist) {
              bestDist = dist;
              bestName = member.name;
            }
          }

          if (bestName) {
            newMatches.push({ normX, name: bestName });
            matchedNamesSet.add(bestName);
          }
        }

        recognizedSpatialRef.current = newMatches;
        const namesList = Array.from(matchedNamesSet);
        visibleTeamNamesRef.current = namesList;
        setVisibleTeamNames(namesList);
      } catch {
        // Ignore single frame recognition error
      }
    }, 650);

    return () => clearInterval(interval);
  }, []);

  const enrollTeamMemberFace = useCallback(async (slotIndex: number) => {
    const faceapi = faceApiRef.current;
    const video = videoRef.current;
    if (!faceapi || !video || video.readyState < 2) {
      alert(
        "Wait a second for the camera and Face ID neural net to finish loading!",
      );
      return;
    }

    setEnrollingSlotIdx(slotIndex);
    try {
      const detection = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: 0.5,
          }),
        )
        .withFaceLandmarks(true)
        .withFaceDescriptor();

      if (!detection || !detection.descriptor) {
        alert(
          "Could not lock onto a clear face! Look directly at the camera with good lighting and click Enroll again.",
        );
        setEnrollingSlotIdx(null);
        return;
      }

      const rawFloats = detection.descriptor as Float32Array;
      const descriptorArray: number[] = [];
      for (let i = 0; i < rawFloats.length; i++) {
        descriptorArray.push(Number(rawFloats[i] ?? 0));
      }

      setTeamRoster((prev) => {
        const updated: TeamMemberProfile[] = prev.map((item, idx) =>
          idx === slotIndex
            ? {
                id: item.id,
                name: item.name,
                role: item.role,
                descriptor: descriptorArray,
              }
            : item,
        );
        teamRosterRef.current = updated;
        try {
          localStorage.setItem(
            "omnibot_team_roster_v2",
            JSON.stringify(updated),
          );
        } catch {
          // Ignore
        }
        return updated;
      });
    } catch (err) {
      console.error("Enrollment error:", err);
    } finally {
      setEnrollingSlotIdx(null);
    }
  }, []);

  const updateTeamMemberName = useCallback(
    (slotIndex: number, newName: string) => {
      setTeamRoster((prev) => {
        const updated: TeamMemberProfile[] = prev.map((item, idx) =>
          idx === slotIndex
            ? {
                id: item.id,
                name: newName,
                role: item.role,
                descriptor: item.descriptor,
              }
            : item,
        );
        teamRosterRef.current = updated;
        try {
          localStorage.setItem(
            "omnibot_team_roster_v2",
            JSON.stringify(updated),
          );
        } catch {
          // Ignore
        }
        return updated;
      });
    },
    [],
  );

  const clearTeamMemberSlot = useCallback((slotIndex: number) => {
    setTeamRoster((prev) => {
      const updated: TeamMemberProfile[] = prev.map((item, idx) =>
        idx === slotIndex
          ? {
              id: item.id,
              name: item.name,
              role: item.role,
              descriptor: null,
            }
          : item,
      );
      teamRosterRef.current = updated;
      try {
        localStorage.setItem("omnibot_team_roster_v2", JSON.stringify(updated));
      } catch {
        // Ignore
      }
      return updated;
    });
  }, []);

  // ---------------------------------------------------------------------------
  // 6. ZERO-LAG DELTA WEBSOCKET DISPATCHER
  // ---------------------------------------------------------------------------
  const sendNowIfChanged = useCallback((force = false) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > 128 && !force) return;

    const curr = telemetryRef.current;
    const prev = lastSentRef.current;

    const panChanged = Math.abs(curr.pan - prev.pan) >= 1;
    const moodChanged = curr.mood !== prev.mood;
    const volChanged = curr.volume !== prev.volume;

    if (force || panChanged || moodChanged || volChanged) {
      const payloadStr = JSON.stringify(curr);
      ws.send(payloadStr);
      lastSentRef.current = { ...curr };
      setLastWsPayload(payloadStr);
      setWsPacketsSent((c) => c + 1);
    }
  }, []);

  const updateTelemetry = useCallback(
    (
      partial: Partial<{ pan: number; mood: EyeMood; volume: number }>,
      instant = false,
    ) => {
      if (partial.pan !== undefined) {
        const clampedPan = Math.max(20, Math.min(160, Math.round(partial.pan)));
        telemetryRef.current.pan = clampedPan;
        setPanAngle(clampedPan);
      }
      if (partial.mood !== undefined) {
        telemetryRef.current.mood = partial.mood;
        setCurrentMood(partial.mood);
      }
      if (partial.volume !== undefined) {
        const clampedVol = Math.max(
          0,
          Math.min(16, Math.round(partial.volume)),
        );
        telemetryRef.current.volume = clampedVol;
        setVuVolume(clampedVol);
      }
      if (
        instant ||
        partial.mood !== undefined ||
        partial.volume !== undefined
      ) {
        sendNowIfChanged(true);
      }
    },
    [sendNowIfChanged],
  );

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    try {
      const socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setWsConnected(true);
        sendNowIfChanged(true);
      };
      socket.onclose = () => setWsConnected(false);
      socket.onerror = () => setWsConnected(false);
      wsRef.current = socket;
    } catch {
      setWsConnected(false);
    }
  }, [wsUrl, sendNowIfChanged]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  useEffect(() => {
    const interval = setInterval(() => {
      sendNowIfChanged(false);
    }, 66);
    return () => clearInterval(interval);
  }, [sendNowIfChanged]);

  // ---------------------------------------------------------------------------
  // 7. ENUMERATE CAMERA & MICROPHONE DEVICES
  // ---------------------------------------------------------------------------
  const refreshDevices = useCallback(async () => {
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      tempStream.getTracks().forEach((t) => t.stop());

      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      const mics = all.filter((d) => d.kind === "audioinput");

      setVideoDevices(cams);
      setAudioInDevices(mics);

      if (!selectedVideoId && cams.length > 0) {
        const usbCam = cams.find((c) => c.label.toLowerCase().includes("usb"));
        const firstCam = cams[0];
        if (usbCam) {
          setSelectedVideoId(usbCam.deviceId);
        } else if (firstCam) {
          setSelectedVideoId(firstCam.deviceId);
        }
      }
      if (!selectedMicId && mics.length > 0) {
        // Default to Built-in MacBook Mic or USB Mic automatically
        const firstMic = mics[0];
        if (firstMic) {
          setSelectedMicId(firstMic.deviceId);
        }
      }
    } catch (err) {
      console.error("Device enumeration error:", err);
    }
  }, [selectedVideoId, selectedMicId]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  useEffect(() => {
    let activeStream: MediaStream | null = null;
    async function startCamera() {
      if (!selectedVideoId) return;
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedVideoId },
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera start error:", err);
      }
    }
    startCamera();
    return () => activeStream?.getTracks().forEach((t) => t.stop());
  }, [selectedVideoId]);

  // ---------------------------------------------------------------------------
  // 8. LOCAL NEURAL TTS + SPECIAL CROWD CONTROL 2-SECOND PAUSE CHOREOGRAPHY
  // ---------------------------------------------------------------------------
  const playSingleTtsSegment = useCallback(
    async (text: string, mood: EyeMood, lang: "en" | "hi"): Promise<void> => {
      if (typeof window === "undefined") return;

      updateTelemetry({ mood }, true);

      try {
        const res = await fetch(chatterboxUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            language_id: lang,
            exaggeration: 0.65,
            cfg_weight: 0.4,
          }),
        });

        if (res.ok) {
          setTtsEngineStatus(`Neural TTS (${lang.toUpperCase()})`);
          const blob = await res.blob();
          const audioUrl = URL.createObjectURL(blob);

          await new Promise<void>((resolve) => {
            const audio = new Audio(audioUrl);
            activeAudioRef.current = audio;

            const audioCtx = new AudioContext();
            const source = audioCtx.createMediaElementSource(audio);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 128;
            source.connect(analyser);
            analyser.connect(audioCtx.destination);
            const freqData = new Uint8Array(analyser.frequencyBinCount);

            if (speakingTimerRef.current)
              clearInterval(speakingTimerRef.current);
            speakingTimerRef.current = setInterval(() => {
              analyser.getByteFrequencyData(freqData);
              let sum = 0;
              for (let i = 0; i < 24; i++) sum += freqData[i] ?? 0;
              const avg = sum / 24;
              const mappedVu = Math.min(16, Math.max(2, Math.round(avg / 10)));
              updateTelemetry({ volume: mappedVu }, true);
            }, 55);

            const finishSegment = () => {
              if (speakingTimerRef.current) {
                clearInterval(speakingTimerRef.current);
                speakingTimerRef.current = null;
              }
              audioCtx.close().catch(() => {});
              URL.revokeObjectURL(audioUrl);
              updateTelemetry({ volume: 0 }, true);
              resolve();
            };

            audio.onended = finishSegment;
            audio.onerror = finishSegment;
            audio.play().catch(finishSegment);
          });
          return;
        }
      } catch {
        setTtsEngineStatus("Browser Fallback");
      }

      await new Promise<void>((resolve) => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang === "hi" ? "hi-IN" : "en-IN";
        utterance.rate = 1.05;
        utterance.pitch = 0.9;

        let tick = 0;
        if (speakingTimerRef.current) clearInterval(speakingTimerRef.current);
        speakingTimerRef.current = setInterval(() => {
          tick++;
          const wave = Math.abs(Math.sin(tick * 0.45)) * 10 + Math.random() * 5;
          updateTelemetry(
            { volume: Math.min(16, Math.max(3, Math.round(wave))) },
            true,
          );
        }, 60);

        const done = () => {
          if (speakingTimerRef.current) {
            clearInterval(speakingTimerRef.current);
            speakingTimerRef.current = null;
          }
          updateTelemetry({ volume: 0 }, true);
          resolve();
        };

        utterance.onend = done;
        utterance.onerror = done;
        window.speechSynthesis.speak(utterance);
      });
    },
    [chatterboxUrl, updateTelemetry],
  );

  const speakWithChatterbox = useCallback(
    async (text: string, mood: EyeMood, lang: "en" | "hi" = "en") => {
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }

      setIsSpeaking(true);
      isSpeakingRef.current = true;

      await playSingleTtsSegment(text, mood, lang);

      setIsSpeaking(false);
      isSpeakingRef.current = false;
      updateTelemetry({ volume: 0, mood: "normal" }, true);
    },
    [playSingleTtsSegment, updateTelemetry],
  );

  const executeCrowdControlAlert = useCallback(async () => {
    if (isSpeakingRef.current || isThinkingRef.current) return;
    lastCrowdAlertMsRef.current = Date.now();
    lastInteractionMsRef.current = Date.now();

    setIsSpeaking(true);
    isSpeakingRef.current = true;
    manualNeckHoldUntilRef.current = Date.now() + 9000;

    const faces = trackedFacesRef.current;
    const primarySpeakerAngle = faces[0]?.panAngle ?? 90;
    let crowdAngle = primarySpeakerAngle > 90 ? 45 : 135;
    if (faces.length > 1) {
      const sideFaces = faces.slice(1);
      const avgSideAngle =
        sideFaces.reduce((acc, f) => acc + f.panAngle, 0) / sideFaces.length;
      crowdAngle = Math.round(avgSideAngle);
    }

    setChatLogs((prev) => [
      {
        id: `crowd-${Date.now()}`,
        sender: "OMNI-BOT",
        text: "हाँ आओ ना, घुस जाओ मेरे अंदर! ... [2s Red Glare] ... हटो सब सामने से!!!",
        mood: "angry",
        intent: "chat",
        latencyMs: 0,
        timestamp: new Date().toLocaleTimeString(),
      },
      ...prev,
    ]);

    updateTelemetry({ pan: crowdAngle, mood: "angry" }, true);
    await playSingleTtsSegment("हाँ आओ ना, घुस जाओ मेरे अंदर!", "angry", "hi");

    updateTelemetry({ mood: "angry", volume: 0 }, true);
    await new Promise((r) => setTimeout(r, 2000));

    await playSingleTtsSegment("हटो सब सामने से!", "angry", "hi");

    updateTelemetry(
      { pan: primarySpeakerAngle, mood: "normal", volume: 0 },
      true,
    );
    setIsSpeaking(false);
    isSpeakingRef.current = false;
  }, [playSingleTtsSegment, updateTelemetry]);

  // ---------------------------------------------------------------------------
  // 9. MEDIAPIPE 60FPS MULTI-FACE TRACKING + >4 CROWD DETECTOR
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let isMounted = true;
    async function loadMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
        );
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.52,
        });
        if (isMounted) {
          faceDetectorRef.current = detector;
          setVisionReady(true);
        }
      } catch (err) {
        console.error("MediaPipe init error:", err);
      }
    }
    loadMediaPipe();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let animFrameId: number;
    let lastVideoTime = -1;
    let frameCount = 0;
    let lastFpsTimestamp = performance.now();

    const renderVisionLoop = () => {
      const video = videoRef.current;
      const canvas = hudCanvasRef.current;
      const detector = faceDetectorRef.current;

      const now = performance.now();
      frameCount++;
      if (now - lastFpsTimestamp >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastFpsTimestamp = now;
      }

      if (
        video &&
        canvas &&
        detector &&
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTime
      ) {
        lastVideoTime = video.currentTime;
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, width, height);

          const result = detector.detectForVideo(video, now);
          const detections = result.detections || [];
          const count = detections.length;
          setFacesDetected(count);
          facesCountRef.current = count;

          if (count >= 5 && crowdControlEnabledRef.current) {
            if (crowdOverloadStartMsRef.current === 0) {
              crowdOverloadStartMsRef.current = Date.now();
            } else if (
              Date.now() - crowdOverloadStartMsRef.current >= 1200 &&
              Date.now() - lastCrowdAlertMsRef.current >= 45000 &&
              !isSpeakingRef.current &&
              !isThinkingRef.current
            ) {
              crowdOverloadStartMsRef.current = 0;
              executeCrowdControlAlert();
            }
          } else {
            crowdOverloadStartMsRef.current = 0;
          }

          const currentFaces: TrackedFace[] = [];

          if (count > 0) {
            const sorted = [...detections].sort((a, b) => {
              const areaA =
                (a.boundingBox?.width || 0) * (a.boundingBox?.height || 0);
              const areaB =
                (b.boundingBox?.width || 0) * (b.boundingBox?.height || 0);
              return areaB - areaA;
            });

            sorted.forEach((det, idx) => {
              const box = det.boundingBox;
              if (!box) return;

              const faceCenterX = box.originX + box.width / 2;
              const faceCenterY = box.originY + box.height / 2;
              const normalizedX = Math.max(0, Math.min(1, faceCenterX / width));
              const mappedAngle = invertPanRef.current
                ? 20 + normalizedX * 140
                : 160 - normalizedX * 140;

              let matchedName: string | null = null;
              for (const rec of recognizedSpatialRef.current) {
                if (Math.abs(rec.normX - normalizedX) < 0.14) {
                  matchedName = rec.name;
                  break;
                }
              }

              currentFaces.push({
                centerX: faceCenterX,
                centerY: faceCenterY,
                panAngle: mappedAngle,
                area: box.width * box.height,
                matchedName,
              });

              const color = matchedName
                ? "#34d399"
                : idx === 0
                  ? "#38bdf8"
                  : "#fbbf24";
              ctx.strokeStyle = color;
              ctx.lineWidth = idx === 0 ? 2.5 : 2;
              ctx.strokeRect(box.originX, box.originY, box.width, box.height);

              ctx.fillStyle = color;
              ctx.font = "bold 11px monospace";
              const tagLabel = matchedName
                ? `TEAM: ${matchedName.toUpperCase()} • ${Math.round(
                    mappedAngle,
                  )}°`
                : idx === 0
                  ? `SPEAKER #1 • ${Math.round(mappedAngle)}°`
                  : `CROWD #${idx + 1} • ${Math.round(mappedAngle)}°`;

              ctx.fillText(
                tagLabel,
                box.originX,
                Math.max(14, box.originY - 6),
              );

              ctx.beginPath();
              ctx.arc(faceCenterX, faceCenterY, 3.5, 0, Math.PI * 2);
              ctx.fill();
            });

            const firstFace = currentFaces[0];
            if (
              autoTrackRef.current &&
              Date.now() > manualNeckHoldUntilRef.current &&
              firstFace
            ) {
              updateTelemetry({ pan: firstFace.panAngle });
            }
          }

          trackedFacesRef.current = currentFaces;
        }
      }
      animFrameId = requestAnimationFrame(renderVisionLoop);
    };

    animFrameId = requestAnimationFrame(renderVisionLoop);
    return () => cancelAnimationFrame(animFrameId);
  }, [executeCrowdControlAlert, updateTelemetry]);

  // ---------------------------------------------------------------------------
  // 10. PASSIVE "ALIVE" HABITS
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!passiveLifeHabits) return;

    const habitTimer = setInterval(() => {
      if (
        isSpeakingRef.current ||
        isThinkingRef.current ||
        Date.now() < manualNeckHoldUntilRef.current
      )
        return;

      if (facesCountRef.current === 0 && autoTrackRef.current) {
        const gentleAngles = [75, 85, 90, 96, 108, 90];
        const nextPan =
          gentleAngles[Math.floor(Math.random() * gentleAngles.length)] ?? 90;
        updateTelemetry({ pan: nextPan }, true);
      }

      const roll = Math.random();
      if (roll < 0.2) {
        updateTelemetry({ mood: "happy" }, true);
        setTimeout(() => {
          if (!isSpeakingRef.current) updateTelemetry({ mood: "normal" }, true);
        }, 1800);
      } else if (roll < 0.32) {
        updateTelemetry({ mood: "wink_left" }, true);
        setTimeout(() => {
          if (!isSpeakingRef.current) updateTelemetry({ mood: "normal" }, true);
        }, 850);
      }
    }, 6500);

    return () => clearInterval(habitTimer);
  }, [passiveLifeHabits, updateTelemetry]);

  const captureFrameForIntent = useCallback(
    (intent: TaskIntent): string | null => {
      if (intent === "chat" && facesCountRef.current <= 1) return null;

      const video = videoRef.current;
      if (!video || video.readyState < 2) return null;

      const offscreen = document.createElement("canvas");
      const targetW = 480;
      const targetH = 360;
      offscreen.width = targetW;
      offscreen.height = targetH;

      const ctx = offscreen.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, targetW, targetH);
      return offscreen.toDataURL("image/jpeg", 0.6);
    },
    [],
  );

  // ---------------------------------------------------------------------------
  // 11. ASK OMNI-BOT (INCLUDES MATH EQUATIONS + 5-CHAT MEMORY + GUEST INTRO)
  // ---------------------------------------------------------------------------
  const triggerOmniBrain = useCallback(
    async (rawText: string, forcedIntent?: TaskIntent) => {
      const cleanPrompt = rawText.trim();
      if (!cleanPrompt) return;

      lastInteractionMsRef.current = Date.now();

      if (!forcedIntent) {
        const reflex = parsePhysicalVoiceReflex(cleanPrompt);
        if (reflex.targetAngle !== null) {
          manualNeckHoldUntilRef.current = Date.now() + 5500;
          updateTelemetry({ pan: reflex.targetAngle, mood: reflex.mood }, true);

          if (reflex.pureCommandReply) {
            const replyText = reflex.pureCommandReply;
            const nowTime = new Date().toLocaleTimeString();
            setChatLogs((prev) => [
              {
                id: `bot-${Date.now()}`,
                sender: "OMNI-BOT",
                text: replyText,
                mood: reflex.mood,
                intent: "chat",
                latencyMs: 5,
                timestamp: nowTime,
              },
              {
                id: `vis-${Date.now() - 1}`,
                sender: "USER",
                text: cleanPrompt,
                intent: "chat",
                timestamp: nowTime,
              },
              ...prev,
            ]);
            speakWithChatterbox(replyText, reflex.mood, "en");
            return;
          }
        }
      }

      const resolvedIntent: TaskIntent =
        forcedIntent || detectUserIntent(cleanPrompt);
      setActiveIntentBadge(resolvedIntent);
      setIsThinking(true);
      isThinkingRef.current = true;

      let guestTargetAngle: number | null = null;
      if (resolvedIntent === "guest_intro") {
        const faces = trackedFacesRef.current;
        if (faces.length >= 2) {
          const speakerFace =
            faces.find((f) => f.matchedName !== null) ?? faces[0];
          const guestFace = faces.find((f) => f !== speakerFace) ?? faces[1];

          if (speakerFace && guestFace) {
            manualNeckHoldUntilRef.current = Date.now() + 8000;
            updateTelemetry(
              { pan: speakerFace.panAngle, mood: "surprised" },
              true,
            );
            guestTargetAngle = guestFace.panAngle;
          }
        } else if (faces.length === 1 && faces[0]) {
          const currentFacePan = faces[0].panAngle;
          guestTargetAngle = currentFacePan > 90 ? 60 : 120;
        }
      } else {
        updateTelemetry(
          {
            mood: resolvedIntent === "math_solve" ? "surprised" : "suspicious",
          },
          true,
        );
      }

      const chronologicalHistory = [...chatLogsRef.current]
        .filter((l) => l.id !== "init-1")
        .slice(0, 3)
        .reverse()
        .map((l) => ({
          role: (l.sender === "USER" ? "user" : "assistant") as
            | "user"
            | "assistant",
          content: l.text,
        }));

      const startTime = performance.now();
      if (resolvedIntent !== "autonomous_idle") {
        const visitorLog: ChatLog = {
          id: `vis-${Date.now()}`,
          sender: "USER",
          text: cleanPrompt,
          intent: resolvedIntent,
          timestamp: new Date().toLocaleTimeString(),
        };
        setChatLogs((prev) => [visitorLog, ...prev]);
      }

      const snapshotBase64 = captureFrameForIntent(resolvedIntent);

      try {
        const res = await fetch("/api/omni-brain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: cleanPrompt,
            imageBase64: snapshotBase64,
            intent: resolvedIntent,
            facesCount: facesCountRef.current,
            visibleTeamMembers: visibleTeamNamesRef.current,
            allTeamNames: teamRosterRef.current.map((m) => m.name),
            history: chronologicalHistory,
            offlineMode,
            customApiKey,
          }),
        });

        const data = await res.json();
        const elapsed = Math.round(performance.now() - startTime);
        setLastAiLatency(elapsed);

        const botReply: string = data.reply || "Hello!";
        const botMood: EyeMood = data.mood || "happy";
        const botLang: "en" | "hi" = data.lang || "en";
        const botMathSteps: string[] = Array.isArray(data.mathSteps)
          ? data.mathSteps
          : [];

        if (resolvedIntent === "guest_intro" && guestTargetAngle !== null) {
          const target = guestTargetAngle;
          setTimeout(() => {
            manualNeckHoldUntilRef.current = Date.now() + 6500;
            updateTelemetry({ pan: target, mood: "happy" }, true);
          }, 900);
        }

        const botLog: ChatLog = {
          id: `bot-${Date.now()}`,
          sender: "OMNI-BOT",
          text: botReply,
          mathSteps: botMathSteps,
          mood: botMood,
          intent: resolvedIntent,
          latencyMs: elapsed,
          timestamp: new Date().toLocaleTimeString(),
        };
        setChatLogs((prev) => [botLog, ...prev]);

        speakWithChatterbox(botReply, botMood, botLang);
      } catch (err) {
        console.error("Omni-Brain error:", err);
      } finally {
        setIsThinking(false);
        isThinkingRef.current = false;
      }
    },
    [
      offlineMode,
      customApiKey,
      captureFrameForIntent,
      speakWithChatterbox,
      updateTelemetry,
    ],
  );

  // Live Camera Feed Commentary Loop (Triggers every 25s of silence when enabled)
  useEffect(() => {
    if (!spontaneousMonologue) return;
    const timer = setInterval(() => {
      if (isSpeakingRef.current || isThinkingRef.current) return;
      if (
        lastInteractionMsRef.current > 0 &&
        Date.now() - lastInteractionMsRef.current >= 60000 &&
  facesCountRef.current >= 1 
      ) {
        triggerOmniBrain(
          "Look at the live webcam feed right now and make a spontaneous, witty observation about the person, crowd, or scene in front of you.",
          "autonomous_idle",
        );
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [spontaneousMonologue, triggerOmniBrain]);

  // ---------------------------------------------------------------------------
  // 12. HARDWARE VAD + GROQ WHISPER MIC ENGINE (NEVER CRASHES OR BLINKS!)
  // ---------------------------------------------------------------------------
  const stopHardwareMic = useCallback(() => {
    micActiveRef.current = false;
    setMicActive(false);
    setIsRecordingSpeech(false);
    setMicVolumeLevel(0);

    if (vadAnimFrameRef.current) {
      cancelAnimationFrame(vadAnimFrameRef.current);
      vadAnimFrameRef.current = null;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // Ignore
      }
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (micAudioCtxRef.current) {
      micAudioCtxRef.current.close().catch(() => {});
      micAudioCtxRef.current = null;
    }
    setMicStatusHint("Mic Paused");
  }, []);

  const startHardwareMic = useCallback(async () => {
    try {
      stopHardwareMic();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: selectedMicId
          ? {
              deviceId: { exact: selectedMicId },
              echoCancellation: true,
              noiseSuppression: true,
            }
          : { echoCancellation: true, noiseSuppression: true },
      });

      micStreamRef.current = stream;
      const audioCtx = new AudioContext();
      micAudioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      let speakingNow = false;
      let silenceStartMs = 0;
      let speechStartMs = 0;

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const chunks = audioChunksRef.current;
        audioChunksRef.current = [];
        if (
          chunks.length === 0 ||
          isSpeakingRef.current ||
          isThinkingRef.current
        ) {
          return;
        }

        const audioBlob = new Blob(chunks, { type: "audio/webm" });
        if (audioBlob.size < 1800) return;

        setMicStatusHint("⚡ Transcribing voice via Groq Whisper...");
        try {
          const form = new FormData();
          form.append("file", audioBlob, "speech.webm");
          form.append("lang", micLangRef.current);

          const res = await fetch("/api/stt", {
            method: "POST",
            body: form,
          });
          if (res.ok) {
            const data = await res.json();
            const text = (data.text || "").trim();
            if (text) {
              setMicStatusHint(`Heard: "${text}"`);
              triggerOmniBrain(text);
              return;
            }
          }
          setMicStatusHint("Listening — Speak anytime!");
        } catch {
          setMicStatusHint("Listening — Speak anytime!");
        }
      };

      micActiveRef.current = true;
      setMicActive(true);
      setMicStatusHint(
        "Always-On Mic Active — Speak into your MacBook or Collar Mic!",
      );

      const monitorVoiceActivity = () => {
        if (!micActiveRef.current || !micAudioCtxRef.current) return;

        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 2; i < 45; i++) {
          sum += dataArray[i] ?? 0;
        }
        const avgVolume = sum / 43;
        const normalizedLevel = Math.min(100, Math.round(avgVolume * 1.6));
        setMicVolumeLevel(normalizedLevel);

        // Pause recording while Omni-Bot is speaking or thinking
        if (isSpeakingRef.current || isThinkingRef.current) {
          if (speakingNow && recorder.state === "recording") {
            audioChunksRef.current = [];
            recorder.stop();
          }
          speakingNow = false;
          setIsRecordingSpeech(false);
          vadAnimFrameRef.current = requestAnimationFrame(monitorVoiceActivity);
          return;
        }

        const SPEECH_THRESHOLD = 18; // Sensitive enough for MacBook Air internal mic!
        const now = Date.now();

        if (normalizedLevel >= SPEECH_THRESHOLD) {
          silenceStartMs = now;
          if (!speakingNow) {
            speakingNow = true;
            speechStartMs = now;
            setIsRecordingSpeech(true);
            setMicStatusHint("🎙️ Hearing you speak...");
            audioChunksRef.current = [];
            if (recorder.state === "inactive") {
              recorder.start();
            }
          }
        } else if (speakingNow) {
          // If silent for 1.05 seconds after speaking for at least 350ms -> Send to Whisper!
          if (now - silenceStartMs >= 1050) {
            speakingNow = false;
            setIsRecordingSpeech(false);
            if (now - speechStartMs >= 350 && recorder.state === "recording") {
              recorder.stop();
            } else if (recorder.state === "recording") {
              audioChunksRef.current = [];
              recorder.stop();
            }
          }
        }

        vadAnimFrameRef.current = requestAnimationFrame(monitorVoiceActivity);
      };

      vadAnimFrameRef.current = requestAnimationFrame(monitorVoiceActivity);
    } catch (err: any) {
      console.error("Mic start error:", err);
      setMicStatusHint(
        "Microphone blocked! Click the camera/mic icon in Chrome's address bar to Allow.",
      );
      setMicActive(false);
    }
  }, [selectedMicId, stopHardwareMic, triggerOmniBrain]);

  const toggleMicButton = useCallback(() => {
    if (micActive) {
      stopHardwareMic();
    } else {
      startHardwareMic();
    }
  }, [micActive, startHardwareMic, stopHardwareMic]);

  const moods: EyeMood[] = [
    "normal",
    "happy",
    "suspicious",
    "surprised",
    "wink_left",
    "wink_right",
    "angry",
    "crying",
    "scared",
    "love",
    "dizzy",
  ];

  const isAngryMode = currentMood === "angry";

  // Helper to render the Digital Dual-OLED Eye Simulator with Bright Red Angry Eyes & New Emotes
  const renderDigitalEye = (isLeft: boolean) => {
    const eyeColor = isAngryMode
      ? "bg-rose-500 shadow-[0_0_20px_#f43f5e]"
      : currentMood === "love"
        ? "bg-pink-400 shadow-[0_0_16px_#f472b6]"
        : currentMood === "scared"
          ? "bg-purple-400 shadow-[0_0_14px_#c084fc]"
          : currentMood === "dizzy"
            ? "bg-amber-400 shadow-[0_0_14px_#fbbf24]"
            : "bg-sky-400 shadow-[0_0_16px_#38bdf8]";

    if (
      (currentMood === "wink_left" && isLeft) ||
      (currentMood === "wink_right" && !isLeft)
    ) {
      return <div className={`w-12 h-1.5 rounded-full ${eyeColor}`} />;
    }

    if (currentMood === "angry") {
      return (
        <div
          className={`relative w-12 h-8 rounded-lg ${eyeColor} flex items-center justify-center animate-pulse ${
            isLeft ? "rotate-12" : "-rotate-12"
          }`}
        >
          <div className="w-3 h-3 rounded-full bg-slate-950 flex items-center justify-center">
            <div className="w-1 h-1 rounded-full bg-rose-400" />
          </div>
        </div>
      );
    }

    if (currentMood === "crying") {
      return (
        <div className="relative flex flex-col items-center">
          <div
            className={`w-12 h-7 rounded-t-xl bg-sky-400 shadow-[0_0_14px_#38bdf8] ${
              isLeft ? "-rotate-12" : "rotate-12"
            } flex items-center justify-center`}
          >
            <div className="w-3 h-3 rounded-full bg-slate-950" />
          </div>
          <div className="flex gap-3 mt-1 animate-bounce">
            <span className="w-1.5 h-2.5 rounded-full bg-sky-300" />
            <span className="w-1.5 h-2.5 rounded-full bg-sky-300" />
          </div>
        </div>
      );
    }

    if (currentMood === "scared") {
      return (
        <div className="w-11 h-11 rounded-full border-2 border-purple-400 flex items-center justify-center animate-ping">
          <div className="w-2 h-2 rounded-full bg-purple-300" />
        </div>
      );
    }

    if (currentMood === "love") {
      return (
        <div className="text-3xl text-pink-400 drop-shadow-[0_0_12px_#f472b6] animate-bounce select-none">
          ♥
        </div>
      );
    }

    if (currentMood === "dizzy") {
      return (
        <div className="w-11 h-11 rounded-full border-2 border-amber-400 flex items-center justify-center font-mono font-extrabold text-xl text-amber-300 animate-spin select-none">
          ✕
        </div>
      );
    }

    if (currentMood === "happy") {
      return (
        <div className="w-12 h-6 rounded-t-full border-t-4 border-sky-400 shadow-[0_0_12px_#38bdf8]" />
      );
    }

    if (currentMood === "suspicious") {
      return <div className={`w-12 h-3 rounded-md ${eyeColor}`} />;
    }

    if (currentMood === "surprised") {
      return (
        <div
          className={`w-11 h-11 rounded-full ${eyeColor} flex items-center justify-center`}
        >
          <div className="w-2.5 h-2.5 rounded-full bg-slate-950" />
        </div>
      );
    }

    return (
      <div
        className={`w-11 h-9 rounded-xl ${eyeColor} flex items-center justify-center`}
      >
        <div className="w-3.5 h-3.5 rounded-full bg-slate-950 flex items-start justify-start p-0.5">
          <div className="w-1 h-1 rounded-full bg-white" />
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#050811] text-slate-100 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* HEADER */}
        <header
          className={`bg-slate-900/95 border rounded-2xl px-5 py-3.5 flex flex-wrap items-center justify-between gap-4 shadow-xl transition-colors ${
            isAngryMode
              ? "border-rose-500/80 shadow-rose-950/50"
              : "border-slate-800"
          }`}
        >
          <div className="flex items-center gap-3.5">
            <div
              className={`h-3.5 w-3.5 rounded-full ${
                isAngryMode
                  ? "bg-rose-500 shadow-[0_0_14px_#f43f5e] animate-ping"
                  : wsConnected
                    ? "bg-emerald-400 shadow-[0_0_12px_#34d399]"
                    : "bg-rose-500 animate-pulse"
              }`}
            />
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-lg font-extrabold tracking-wide text-white">
                OMNI-BOT // CONTROL DECK
              </h1>
              <span className="text-[11px] font-mono font-bold px-2.5 py-0.5 rounded-md bg-sky-500/15 text-sky-300 border border-sky-500/30">
                {wsConnected ? "ESP32:ONLINE" : "ESP32:OFFLINE"}
              </span>
              {isAngryMode && (
                <span className="text-[11px] font-mono font-extrabold px-2.5 py-0.5 rounded-md bg-rose-600 text-white shadow-[0_0_14px_#f43f5e]">
                  🔴 ANGRY MODE: RED EYES & RING
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <div className="px-3 py-1 rounded-lg bg-slate-950 border border-slate-800">
              CAM: <strong className="text-sky-400">{fps} FPS</strong>
            </div>
            <div className="px-3 py-1 rounded-lg bg-slate-950 border border-slate-800">
              PAN: <strong className="text-sky-400">{panAngle}°</strong>
            </div>
            <div className="px-3 py-1 rounded-lg bg-slate-950 border border-slate-800">
              AI:{" "}
              <strong className="text-emerald-400">{lastAiLatency}ms</strong>
            </div>

            <button
              onClick={() => setShowDebugDrawer(!showDebugDrawer)}
              className={`px-3.5 py-1 rounded-lg font-sans font-bold border transition ${
                showDebugDrawer
                  ? "bg-sky-500 text-slate-950 border-sky-400"
                  : "bg-slate-800 hover:bg-slate-700 text-sky-300 border-slate-700"
              }`}
            >
              {showDebugDrawer
                ? "✕ Close Config"
                : "🛠️ Team Face ID & System Config"}
            </button>
          </div>
        </header>

        {/* SYSTEM CONFIG & 5-MEMBER BIOMETRIC FACE ENROLLMENT DRAWER */}
        {showDebugDrawer && (
          <div className="bg-slate-900/95 border border-sky-500/40 rounded-2xl p-4 space-y-4 text-xs shadow-2xl">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="text-slate-400 block mb-1 font-bold uppercase">
                  ESP32 WebSocket Target:
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={wsUrl}
                    onChange={(e) => setWsUrl(e.target.value)}
                    className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sky-300 font-mono"
                  />
                  <button
                    onClick={connectWebSocket}
                    className="px-3 py-1.5 bg-sky-500/20 border border-sky-500/40 text-sky-300 rounded-lg font-bold"
                  >
                    Link
                  </button>
                </div>
              </div>

              <div>
                <label className="text-purple-400 block mb-1 font-bold uppercase">
                  Local TTS Endpoint:
                </label>
                <input
                  type="text"
                  value={chatterboxUrl}
                  onChange={(e) => setChatterboxUrl(e.target.value)}
                  className="w-full bg-slate-950 border border-purple-500/40 rounded-lg px-2.5 py-1.5 text-purple-200 font-mono"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1 font-bold uppercase">
                  USB Face Camera:
                </label>
                <select
                  value={selectedVideoId}
                  onChange={(e) => setSelectedVideoId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
                >
                  {videoDevices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1 font-bold uppercase">
                  Microphone (MacBook or Collar Mic):
                </label>
                <select
                  value={selectedMicId}
                  onChange={(e) => setSelectedMicId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
                >
                  {audioInDevices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Mic ${i + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* 5-MEMBER NEURAL BIOMETRIC FACE ENROLLMENT MATRIX */}
            <div className="pt-3 border-t border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="font-extrabold text-emerald-400 uppercase tracking-wider">
                  👥 5-Member Team 128-D Biometric Face ID Roster
                </span>
                <span className="font-mono text-[11px] text-slate-400">
                  Recognized in Frame:{" "}
                  <strong className="text-emerald-300">
                    {visibleTeamNames.length > 0
                      ? visibleTeamNames.join(", ")
                      : "None"}
                  </strong>
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-5 gap-2.5">
                {teamRoster.map((member, idx) => {
                  const isEnrolled = Boolean(
                    member.descriptor && member.descriptor.length === 128,
                  );
                  return (
                    <div
                      key={member.id}
                      className={`p-2.5 rounded-xl border ${
                        isEnrolled
                          ? "bg-emerald-950/25 border-emerald-500/40"
                          : "bg-slate-950 border-slate-800"
                      } space-y-2`}
                    >
                      <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>SLOT #{idx + 1}</span>
                        <span
                          className={
                            isEnrolled ? "text-emerald-400 font-bold" : ""
                          }
                        >
                          {isEnrolled ? "● ENROLLED" : "○ EMPTY"}
                        </span>
                      </div>
                      <input
                        type="text"
                        value={member.name}
                        onChange={(e) =>
                          updateTeamMemberName(idx, e.target.value)
                        }
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs font-bold text-white"
                      />
                      <div className="flex gap-1">
                        <button
                          onClick={() => enrollTeamMemberFace(idx)}
                          disabled={enrollingSlotIdx !== null}
                          className="flex-1 py-1 px-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 rounded-lg font-bold text-[11px]"
                        >
                          {enrollingSlotIdx === idx
                            ? "Scanning..."
                            : isEnrolled
                              ? "↻ Re-Scan"
                              : "📸 Enroll"}
                        </button>
                        {isEnrolled && (
                          <button
                            onClick={() => clearTeamMemberSlot(idx)}
                            className="py-1 px-2 bg-rose-500/20 border border-rose-500/40 text-rose-300 rounded-lg font-bold text-[11px]"
                          >
                            ✕
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* MAIN WORKSPACE */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* LEFT COLUMN: CAMERA HUD + DIGITAL EYE SIMULATOR & CONTROLS */}
          <div className="lg:col-span-6 flex flex-col gap-4">
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300">
                    OPTICAL FEED ({facesDetected} FACE
                    {facesDetected === 1 ? "" : "S"})
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs font-medium text-slate-300">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={autoTrackFace}
                      onChange={(e) => setAutoTrackFace(e.target.checked)}
                      className="accent-sky-400"
                    />
                    Auto-Track
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={passiveLifeHabits}
                      onChange={(e) => setPassiveLifeHabits(e.target.checked)}
                      className="accent-sky-400"
                    />
                    Alive Habits
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={invertPanDirection}
                      onChange={(e) => setInvertPanDirection(e.target.checked)}
                      className="accent-sky-400"
                    />
                    Flip L/R
                  </label>
                </div>
              </div>

              <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-slate-800">
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas
                  ref={hudCanvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />
                {!visionReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 text-sky-400 text-xs font-mono">
                    Initializing MediaPipe GPU Face Detector...
                  </div>
                )}
              </div>

              <div className="grid grid-cols-3 gap-2.5 mt-3.5">
                <button
                  onClick={() =>
                    triggerOmniBrain(
                      "Can you see and tell me what I am holding in my hand right now?",
                      "object_inspect",
                    )
                  }
                  disabled={isThinking}
                  className="py-2.5 px-2.5 rounded-xl bg-sky-500/15 hover:bg-sky-500/25 border border-sky-500/40 text-sky-200 text-xs font-bold transition flex items-center justify-center gap-1.5"
                >
                  <span>✋</span> INSPECT OBJECT
                </button>
                <button
                  onClick={() =>
                    triggerOmniBrain(
                      "Read the math equation or question written on this paper/board and solve it step by step!",
                      "math_solve",
                    )
                  }
                  disabled={isThinking}
                  className="py-2.5 px-2.5 rounded-xl bg-purple-500/15 hover:bg-purple-500/25 border border-purple-500/40 text-purple-200 text-xs font-bold transition flex items-center justify-center gap-1.5"
                >
                  <span>📐</span> SOLVE MATH
                </button>
                <button
                  onClick={() =>
                    triggerOmniBrain(
                      "Hey Omni, meet our special guest and teacher standing right here with me!",
                      "guest_intro",
                    )
                  }
                  disabled={isThinking}
                  className="py-2.5 px-2.5 rounded-xl bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-200 text-xs font-bold transition flex items-center justify-center gap-1.5"
                >
                  <span>🤝</span> INTRODUCE GUEST
                </button>
              </div>
            </div>

            {/* LIVE DUAL-EYE SIMULATOR (TURNS RED ON ANGRY) + SERVO & RING */}
            <div
              className={`bg-slate-900/90 border rounded-2xl p-4 space-y-4 shadow-xl transition-colors ${
                isAngryMode ? "border-rose-500/70" : "border-slate-800"
              }`}
            >
              {/* Digital Dual-OLED Eye Preview Box */}
              <div className="flex items-center justify-between bg-slate-950 border border-slate-800 rounded-xl px-4 py-3">
                <div>
                  <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400 block">
                    LIVE ROBOT EYES & RING STATE
                  </span>
                  <span
                    className={`text-xs font-mono font-bold uppercase ${
                      isAngryMode ? "text-rose-400" : "text-sky-400"
                    }`}
                  >
                    EMOTE: {currentMood.replace("_", " ")}
                  </span>
                </div>
                <div className="flex items-center gap-6 px-4 py-1">
                  {renderDigitalEye(true)}
                  {renderDigitalEye(false)}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-xs font-bold text-slate-300 mb-1">
                    <span>NECK SERVO OVERRIDE</span>
                    <span className="font-mono text-sky-400">{panAngle}°</span>
                  </div>
                  <input
                    type="range"
                    min={20}
                    max={160}
                    value={panAngle}
                    onChange={(e) => {
                      manualNeckHoldUntilRef.current = Date.now() + 5000;
                      updateTelemetry({ pan: Number(e.target.value) }, true);
                    }}
                    className="w-full accent-sky-400 cursor-pointer"
                  />
                  <div className="flex justify-between text-[11px] font-mono text-slate-400 mt-1">
                    <button
                      onClick={() => {
                        manualNeckHoldUntilRef.current = Date.now() + 5000;
                        updateTelemetry({ pan: 145 }, true);
                      }}
                      className="hover:text-sky-300"
                    >
                      [145° LEFT]
                    </button>
                    <button
                      onClick={() => {
                        manualNeckHoldUntilRef.current = 0;
                        updateTelemetry({ pan: 90 }, true);
                      }}
                      className="text-sky-400 font-bold hover:text-sky-300"
                    >
                      [90° CENTER]
                    </button>
                    <button
                      onClick={() => {
                        manualNeckHoldUntilRef.current = Date.now() + 5000;
                        updateTelemetry({ pan: 35 }, true);
                      }}
                      className="hover:text-sky-300"
                    >
                      [35° RIGHT]
                    </button>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-bold text-slate-300 mb-1">
                    <span>
                      {isAngryMode
                        ? "🔴 CRIMSON RED RING LIGHT"
                        : "ARC REACTOR RING LIGHT"}
                    </span>
                    <span
                      className={`font-mono ${
                        isAngryMode ? "text-rose-400 font-bold" : "text-sky-400"
                      }`}
                    >
                      {isAngryMode ? "RED OVERDRIVE" : `${vuVolume}/16 LEDs`}
                    </span>
                  </div>
                  <div className="grid grid-cols-16 gap-1 h-5 bg-slate-950 p-1 rounded-lg border border-slate-800">
                    {Array.from({ length: 16 }).map((_, idx) => (
                      <div
                        key={idx}
                        className={`rounded-xs transition-all ${
                          isAngryMode
                            ? "bg-rose-500 shadow-[0_0_8px_#f43f5e]"
                            : idx < vuVolume
                              ? idx >= 12
                                ? "bg-fuchsia-400"
                                : "bg-sky-400"
                              : "bg-slate-800"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* 11-Emote Matrix */}
              <div>
                <span className="text-xs font-bold text-slate-300 block mb-2">
                  11 EXPRESSIVE EMOTES (INSTANT HARDWARE TRIGGER):
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {moods.map((m) => (
                    <button
                      key={m}
                      onClick={() => updateTelemetry({ mood: m }, true)}
                      className={`px-2.5 py-1.5 rounded-lg text-xs font-bold uppercase border transition ${
                        currentMood === m
                          ? m === "angry"
                            ? "bg-rose-600 text-white border-rose-400 shadow-[0_0_12px_#f43f5e]"
                            : "bg-sky-500 text-slate-950 border-sky-400"
                          : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {m === "crying" && "😭 "}
                      {m === "scared" && "😨 "}
                      {m === "love" && "😍 "}
                      {m === "dizzy" && "😵‍💫 "}
                      {m === "angry" && "😡 "}
                      {m.replace("_", " ")}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: GROQ WHISPER MIC + CHAT & MATH EQUATION STREAM */}
          <div className="lg:col-span-6 flex flex-col bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl justify-between gap-4">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300">
                    GROQ WHISPER VOICE & EQUATION STREAM
                  </span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-sky-300">
                    {activeIntentBadge === "math_solve"
                      ? "📐 STEP-BY-STEP MATH"
                      : "🧠 5-TURN MEMORY"}
                  </span>
                </div>

                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px] font-bold">
                  <button
                    onClick={() => setMicLang("en-IN")}
                    className={`px-2 py-0.5 rounded ${
                      micLang === "en-IN"
                        ? "bg-sky-500 text-slate-950"
                        : "text-slate-400"
                    }`}
                  >
                    EN + Hinglish
                  </button>
                  <button
                    onClick={() => setMicLang("hi-IN")}
                    className={`px-2 py-0.5 rounded ${
                      micLang === "hi-IN"
                        ? "bg-sky-500 text-slate-950"
                        : "text-slate-400"
                    }`}
                  >
                    Pure Hindi
                  </button>
                </div>
              </div>

              {/* Hardware VAD + Groq Whisper Mic Button */}
              <div className="space-y-1.5">
                <button
                  onClick={toggleMicButton}
                  disabled={isThinking}
                  className={`w-full py-3.5 px-4 rounded-xl font-extrabold text-sm border transition flex items-center justify-center gap-2.5 shadow-lg ${
                    isRecordingSpeech
                      ? "bg-rose-500/30 border-rose-400 text-rose-200 animate-pulse"
                      : micActive
                        ? "bg-emerald-500/20 border-emerald-400 text-emerald-200"
                        : "bg-sky-400 hover:bg-sky-300 text-slate-950 border-transparent"
                  }`}
                >
                  {isRecordingSpeech
                    ? "🎙️ HEARING YOU SPEAK... (PAUSE 1s TO SEND)"
                    : micActive
                      ? "🎙️ ALWAYS-ON MIC ACTIVE (MACBOOK OR COLLAR MIC — CLICK TO PAUSE)"
                      : "🎙️ START ALWAYS-ON MIC (GROQ WHISPER VAD)"}
                </button>

                {/* Live Hardware Audio Level Bar */}
                <div className="flex items-center gap-2.5 px-1">
                  <div className="w-32 h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all duration-75 ${
                        micVolumeLevel >= 18
                          ? "bg-emerald-400"
                          : "bg-sky-500/50"
                      }`}
                      style={{ width: `${micVolumeLevel}%` }}
                    />
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 truncate">
                    {micStatusHint}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5 pt-1">
                {[
                  {
                    label: "📐 Demo Math Equation",
                    text: "Solve this quadratic equation step by step: x squared minus 5x plus 6 equals 0",
                  },
                  {
                    label: "🚐 Maruti Omni Van Tease",
                    text: "Oye Omni, tera naam Maruti Omni van pe rakha hai kya?",
                  },
                  {
                    label: "😡 Crowd Warning (Red Angry)",
                    text: "__CROWD_ALERT__",
                  },
                  {
                    label: "😭 Pretend to Cry",
                    text: "Omni, the judges said they might cut your marks, show us how sad you are!",
                  },
                ].map((item) => (
                  <button
                    key={item.label}
                    onClick={() =>
                      item.text === "__CROWD_ALERT__"
                        ? executeCrowdControlAlert()
                        : triggerOmniBrain(item.text)
                    }
                    className="text-xs bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 px-2.5 py-1.5 rounded-lg font-medium transition"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* CHAT LOGS WITH DEDICATED STEP-BY-STEP MATH EQUATION CARDS */}
            <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-3.5 h-80 overflow-y-auto space-y-2.5">
              {chatLogs.map((log) => (
                <div
                  key={log.id}
                  className={`p-3 rounded-xl text-xs leading-relaxed border ${
                    log.sender === "OMNI-BOT"
                      ? log.mood === "angry"
                        ? "bg-rose-950/30 border-rose-500/50 text-rose-100"
                        : "bg-slate-900 border-sky-500/30 text-slate-100"
                      : "bg-slate-900/40 border-slate-800 text-slate-300 ml-6"
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-mono">
                    <span
                      className={`font-bold ${
                        log.mood === "angry" ? "text-rose-400" : "text-sky-400"
                      }`}
                    >
                      {log.sender}
                      {log.mood ? ` • [${log.mood.toUpperCase()}]` : ""}
                      {log.latencyMs ? ` • ${log.latencyMs}ms` : ""}
                    </span>
                    <span>{log.timestamp}</span>
                  </div>
                  <p className="text-[13px]">{log.text}</p>

                  {log.mathSteps && log.mathSteps.length > 0 && (
                    <div className="mt-2.5 p-3 rounded-lg bg-[#04070e] border border-purple-500/40 font-mono text-sm space-y-1.5 shadow-inner">
                      <div className="text-[10px] font-bold uppercase tracking-wider text-purple-400 border-b border-slate-800 pb-1 mb-1.5">
                        ∑ Step-by-Step Mathematical Solution
                      </div>
                      {log.mathSteps.map((step, sIdx) => {
                        const isFinalStep =
                          sIdx === (log.mathSteps?.length ?? 0) - 1;
                        return (
                          <div
                            key={sIdx}
                            className={`flex items-center gap-2.5 px-2 py-1 rounded ${
                              isFinalStep
                                ? "bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 font-bold"
                                : "text-sky-200"
                            }`}
                          >
                            <span className="text-[11px] text-slate-500 select-none">
                              {isFinalStep ? "∴" : `(${sIdx + 1})`}
                            </span>
                            <span className="tracking-wide">{step}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!promptInput.trim()) return;
                triggerOmniBrain(promptInput);
                setPromptInput("");
              }}
              className="flex gap-2"
            >
              <input
                type="text"
                value={promptInput}
                onChange={(e) => setPromptInput(e.target.value)}
                placeholder="Speak into your MacBook mic or type a question..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-sky-400"
              />
              <button
                type="submit"
                disabled={isThinking}
                className="bg-sky-400 hover:bg-sky-300 text-slate-950 font-extrabold px-5 py-2.5 rounded-xl text-xs transition"
              >
                SEND
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
