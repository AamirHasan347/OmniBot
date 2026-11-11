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
  | "surprised";

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
  isOperator: boolean;
}

// Compute compact spatial signature from face bounding box
function extractFaceSignature(
  video: HTMLVideoElement,
  box: { originX: number; originY: number; width: number; height: number }
): number[] | null {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.drawImage(
    video,
    Math.max(0, box.originX),
    Math.max(0, box.originY),
    Math.max(1, box.width),
    Math.max(1, box.height),
    0,
    0,
    16,
    16
  );

  const data = ctx.getImageData(0, 0, 16, 16).data;
  const bins = new Array(12).fill(0);

  for (let i = 0; i < 256; i++) {
    const region = Math.min(3, Math.floor(i / 64));
    bins[region * 3 + 0] += data[i * 4 + 0] / 255;
    bins[region * 3 + 1] += data[i * 4 + 1] / 255;
    bins[region * 3 + 2] += data[i * 4 + 2] / 255;
  }

  return bins.map((v) => v / 64);
}

function compareFaceSignatures(sigA: number[], sigB: number[]): number {
  if (sigA.length !== sigB.length) return 0;
  let diffSum = 0;
  for (let i = 0; i < sigA.length; i++) {
    diffSum += Math.abs(sigA[i] - sigB[i]);
  }
  const avgDiff = diffSum / sigA.length;
  return Math.max(0, Math.min(1, 1 - avgDiff * 2.8));
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
        lower.split(" ").length <= 6
          ? "Looking to my right now!"
          : undefined,
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
      mood: "suspicious",
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
  const [wsUrl, setWsUrl] = useState<string>(
    process.env.NEXT_PUBLIC_ESP32_WS_URL || "ws://192.168.10.44:81"
  );
  const [chatterboxUrl, setChatterboxUrl] = useState<string>(
    "http://127.0.0.1:4123/tts"
  );
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [ttsEngineStatus, setTtsEngineStatus] = useState<string>(
    "Chatterbox Local (4123)"
  );
  const [showDebugDrawer, setShowDebugDrawer] = useState<boolean>(false);
  const [customApiKey, setCustomApiKey] = useState<string>("");

  const [panAngle, setPanAngle] = useState<number>(90);
  const [currentMood, setCurrentMood] = useState<EyeMood>("normal");
  const [vuVolume, setVuVolume] = useState<number>(0);

  const [wsPacketsSent, setWsPacketsSent] = useState<number>(0);
  const [lastWsPayload, setLastWsPayload] = useState<string>(
    '{"pan":90,"mood":"normal","volume":0}'
  );
  const [lastAiLatency, setLastAiLatency] = useState<number>(0);
  const [operatorEnrolled, setOperatorEnrolled] = useState<boolean>(false);
  const [operatorDetectedNow, setOperatorDetectedNow] =
    useState<boolean>(false);

  const [autoTrackFace, setAutoTrackFace] = useState<boolean>(true);
  const [passiveLifeHabits, setPassiveLifeHabits] = useState<boolean>(true);
  const [spontaneousMonologue, setSpontaneousMonologue] =
    useState<boolean>(false);
  const [alwaysOnMic, setAlwaysOnMic] = useState<boolean>(true);
  const [invertPanDirection, setInvertPanDirection] = useState<boolean>(false);
  const [offlineMode, setOfflineMode] = useState<boolean>(false);

  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [audioInDevices, setAudioInDevices] = useState<MediaDeviceInfo[]>([]);
  const [micLang, setMicLang] = useState<"en-IN" | "hi-IN">("en-IN");
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [selectedMicId, setSelectedMicId] = useState<string>("");

  const [visionReady, setVisionReady] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [facesDetected, setFacesDetected] = useState<number>(0);

  const [isListening, setIsListening] = useState<boolean>(false);
  const [micLevel, setMicLevel] = useState<number>(0);
  const [micStatusHint, setMicStatusHint] = useState<string>(
    "Click 'Start Always-On Mic' to begin"
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
      text: "Systems online. Local Chatterbox Multilingual TTS ready on port 4123.",
      mood: "normal",
      intent: "chat",
      timestamp: "READY",
    },
  ]);

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const faceDetectorRef = useRef<FaceDetector | null>(null);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);

  const operatorSignatureRef = useRef<number[] | null>(null);
  const latestPrimaryBoxRef = useRef<{
    originX: number;
    originY: number;
    width: number;
    height: number;
  } | null>(null);
  const trackedFacesRef = useRef<TrackedFace[]>([]);
  const operatorInFrameRef = useRef<boolean>(false);

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
  const alwaysOnMicRef = useRef<boolean>(true);
  const isSpeakingRef = useRef<boolean>(false);
  const isThinkingRef = useRef<boolean>(false);
  const manualNeckHoldUntilRef = useRef<number>(0);

  const facesCountRef = useRef<number>(0);
  const lastInteractionMsRef = useRef<number>(0);
  const speakingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recognitionRef = useRef<any>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    lastInteractionMsRef.current = Date.now();
    try {
      const savedSig = localStorage.getItem("omnibot_operator_face_sig");
      if (savedSig) {
        operatorSignatureRef.current = JSON.parse(savedSig);
        setOperatorEnrolled(true);
      }
    } catch {
      // Ignore
    }
  }, []);

  useEffect(() => {
    autoTrackRef.current = autoTrackFace;
  }, [autoTrackFace]);

  useEffect(() => {
    invertPanRef.current = invertPanDirection;
  }, [invertPanDirection]);

  useEffect(() => {
    alwaysOnMicRef.current = alwaysOnMic;
  }, [alwaysOnMic]);

  const enrollOperatorFace = useCallback(() => {
    const video = videoRef.current;
    const box = latestPrimaryBoxRef.current;
    if (!video || !box) {
      alert("Stand in front of the camera first so your face box is visible!");
      return;
    }
    const sig = extractFaceSignature(video, box);
    if (sig) {
      operatorSignatureRef.current = sig;
      setOperatorEnrolled(true);
      try {
        localStorage.setItem("omnibot_operator_face_sig", JSON.stringify(sig));
      } catch {
        // Ignore
      }
    }
  }, []);

  const clearOperatorEnrollment = useCallback(() => {
    operatorSignatureRef.current = null;
    setOperatorEnrolled(false);
    setOperatorDetectedNow(false);
    operatorInFrameRef.current = false;
    try {
      localStorage.removeItem("omnibot_operator_face_sig");
    } catch {
      // Ignore
    }
  }, []);

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
      instant = false
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
          Math.min(16, Math.round(partial.volume))
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
    [sendNowIfChanged]
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
        setSelectedVideoId(usbCam ? usbCam.deviceId : cams[0].deviceId);
      }
      if (!selectedMicId && mics.length > 0) {
        const usbMic = mics.find((m) => m.label.toLowerCase().includes("usb"));
        setSelectedMicId(usbMic ? usbMic.deviceId : mics[0].deviceId);
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

  useEffect(() => {
    let isMounted = true;
    async function loadMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.55,
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
          setFacesDetected(detections.length);
          facesCountRef.current = detections.length;

          const currentFaces: TrackedFace[] = [];
          let foundOperatorThisFrame = false;

          if (detections.length > 0) {
            const sorted = [...detections].sort((a, b) => {
              const areaA = (a.boundingBox?.width || 0) * (a.boundingBox?.height || 0);
              const areaB = (b.boundingBox?.width || 0) * (b.boundingBox?.height || 0);
              return areaB - areaA;
            });

            if (sorted[0].boundingBox) {
              latestPrimaryBoxRef.current = sorted[0].boundingBox;
            }

            sorted.forEach((det, idx) => {
              const box = det.boundingBox;
              if (!box) return;

              const faceCenterX = box.originX + box.width / 2;
              const faceCenterY = box.originY + box.height / 2;
              const normalizedX = Math.max(0, Math.min(1, faceCenterX / width));
              const mappedAngle = invertPanRef.current
                ? 20 + normalizedX * 140
                : 160 - normalizedX * 140;

              let isOperator = false;
              if (operatorSignatureRef.current) {
                const liveSig = extractFaceSignature(video, box);
                if (liveSig) {
                  const similarity = compareFaceSignatures(
                    operatorSignatureRef.current,
                    liveSig
                  );
                  if (similarity >= 0.82) {
                    isOperator = true;
                    foundOperatorThisFrame = true;
                  }
                }
              }

              currentFaces.push({
                centerX: faceCenterX,
                centerY: faceCenterY,
                panAngle: mappedAngle,
                area: box.width * box.height,
                isOperator,
              });

              const color = isOperator
                ? "#34d399"
                : idx === 0
                ? "#38bdf8"
                : "#fbbf24";
              ctx.strokeStyle = color;
              ctx.lineWidth = idx === 0 ? 2.5 : 2;
              ctx.strokeRect(box.originX, box.originY, box.width, box.height);

              ctx.fillStyle = color;
              ctx.font = "bold 11px monospace";
              const tagLabel = isOperator
                ? `CREATOR RECOGNIZED • ${Math.round(mappedAngle)}°`
                : idx === 0
                ? `SUBJECT #1 • ${Math.round(mappedAngle)}°`
                : `GUEST #${idx + 1} • ${Math.round(mappedAngle)}°`;

              ctx.fillText(
                tagLabel,
                box.originX,
                Math.max(14, box.originY - 6)
              );

              ctx.beginPath();
              ctx.arc(faceCenterX, faceCenterY, 3.5, 0, Math.PI * 2);
              ctx.fill();
            });

            if (
              autoTrackRef.current &&
              Date.now() > manualNeckHoldUntilRef.current &&
              currentFaces.length > 0
            ) {
              updateTelemetry({ pan: currentFaces[0].panAngle });
            }
          }

          trackedFacesRef.current = currentFaces;
          operatorInFrameRef.current = foundOperatorThisFrame;
          setOperatorDetectedNow(foundOperatorThisFrame);
        }
      }
      animFrameId = requestAnimationFrame(renderVisionLoop);
    };

    animFrameId = requestAnimationFrame(renderVisionLoop);
    return () => cancelAnimationFrame(animFrameId);
  }, [updateTelemetry]);

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
          gentleAngles[Math.floor(Math.random() * gentleAngles.length)];
        updateTelemetry({ pan: nextPan }, true);
      }

      const roll = Math.random();
      if (roll < 0.2) {
        updateTelemetry({ mood: "happy" }, true);
        setTimeout(() => {
          if (!isSpeakingRef.current)
            updateTelemetry({ mood: "normal" }, true);
        }, 1800);
      } else if (roll < 0.32) {
        updateTelemetry({ mood: "wink_left" }, true);
        setTimeout(() => {
          if (!isSpeakingRef.current)
            updateTelemetry({ mood: "normal" }, true);
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
      const targetW = intent === "math_solve" ? 640 : 480;
      const targetH = intent === "math_solve" ? 480 : 360;
      offscreen.width = targetW;
      offscreen.height = targetH;

      const ctx = offscreen.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, targetW, targetH);
      return offscreen.toDataURL(
        "image/jpeg",
        intent === "math_solve" ? 0.85 : 0.68
      );
    },
    []
  );

  const speakWithChatterbox = useCallback(
    async (text: string, mood: EyeMood, lang: "en" | "hi" = "en") => {
      if (typeof window === "undefined") return;

      try {
        recognitionRef.current?.stop();
      } catch {
        // Ignore
      }

      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current = null;
      }
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (speakingTimerRef.current) {
        clearInterval(speakingTimerRef.current);
      }

      setIsSpeaking(true);
      isSpeakingRef.current = true;
      updateTelemetry({ mood }, true);

      const resumeMicAfterSpeech = () => {
        if (alwaysOnMicRef.current) {
          setTimeout(() => {
            if (!isSpeakingRef.current && !isThinkingRef.current) {
              try {
                recognitionRef.current?.start();
              } catch {
                // Ignore
              }
            }
          }, 350);
        }
      };

      try {
        const res = await fetch(chatterboxUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            language_id: lang,
            exaggeration: 0.55,
            cfg_weight: 0.4,
          }),
        });

        if (res.ok) {
          setTtsEngineStatus(`Chatterbox (${lang.toUpperCase()})`);
          const blob = await res.blob();
          const audioUrl = URL.createObjectURL(blob);
          const audio = new Audio(audioUrl);
          activeAudioRef.current = audio;

          const audioCtx = new AudioContext();
          const source = audioCtx.createMediaElementSource(audio);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 128;
          source.connect(analyser);
          analyser.connect(audioCtx.destination);
          const freqData = new Uint8Array(analyser.frequencyBinCount);

          speakingTimerRef.current = setInterval(() => {
            analyser.getByteFrequencyData(freqData);
            let sum = 0;
            for (let i = 0; i < 24; i++) sum += freqData[i];
            const avg = sum / 24;
            const mappedVu = Math.min(16, Math.max(2, Math.round(avg / 11)));
            updateTelemetry({ volume: mappedVu }, true);
          }, 55);

          const cleanup = () => {
            if (speakingTimerRef.current) {
              clearInterval(speakingTimerRef.current);
              speakingTimerRef.current = null;
            }
            audioCtx.close().catch(() => {});
            URL.revokeObjectURL(audioUrl);
            setIsSpeaking(false);
            isSpeakingRef.current = false;
            updateTelemetry({ volume: 0, mood: "normal" }, true);
            resumeMicAfterSpeech();
          };

          audio.onended = cleanup;
          audio.onerror = cleanup;
          await audio.play();
          return;
        }
      } catch {
        setTtsEngineStatus("Browser Fallback (Start server.py)");
      }

      // Browser Fallback if local Chatterbox server isn't running
      const cleanForBrowser = text.replace(/\[(chuckle|sigh|laugh)\]/gi, "");
      const utterance = new SpeechSynthesisUtterance(cleanForBrowser);
      utterance.lang = lang === "hi" ? "hi-IN" : "en-IN";
      utterance.rate = 1.02;
      utterance.pitch = 0.9;

      let tick = 0;
      speakingTimerRef.current = setInterval(() => {
        tick++;
        const wave = Math.abs(Math.sin(tick * 0.45)) * 10 + Math.random() * 5;
        updateTelemetry(
          { volume: Math.min(16, Math.max(3, Math.round(wave))) },
          true
        );
      }, 60);

      const fallbackCleanup = () => {
        if (speakingTimerRef.current) {
          clearInterval(speakingTimerRef.current);
          speakingTimerRef.current = null;
        }
        setIsSpeaking(false);
        isSpeakingRef.current = false;
        updateTelemetry({ volume: 0, mood: "normal" }, true);
        resumeMicAfterSpeech();
      };

      utterance.onend = fallbackCleanup;
      utterance.onerror = fallbackCleanup;
      window.speechSynthesis.speak(utterance);
    },
    [chatterboxUrl, updateTelemetry]
  );

  const triggerOmniBrain = useCallback(
    async (rawText: string, forcedIntent?: TaskIntent) => {
      const cleanPrompt = rawText.trim();
      if (!cleanPrompt) return;

      lastInteractionMsRef.current = Date.now();

      if (!forcedIntent) {
        const reflex = parsePhysicalVoiceReflex(cleanPrompt);
        if (reflex.targetAngle !== null) {
          manualNeckHoldUntilRef.current = Date.now() + 5500;
          updateTelemetry(
            { pan: reflex.targetAngle, mood: reflex.mood },
            true
          );

          if (reflex.pureCommandReply) {
            const nowTime = new Date().toLocaleTimeString();
            setChatLogs((prev) => [
              {
                id: `bot-${Date.now()}`,
                sender: "OMNI-BOT",
                text: reflex.pureCommandReply!,
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
            speakWithChatterbox(reflex.pureCommandReply, reflex.mood, "en");
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
          const speakerFace = faces.find((f) => f.isOperator) || faces[0];
          const guestFace = faces.find((f) => f !== speakerFace) || faces[1];

          manualNeckHoldUntilRef.current = Date.now() + 8000;
          updateTelemetry(
            { pan: speakerFace.panAngle, mood: "surprised" },
            true
          );
          guestTargetAngle = guestFace.panAngle;
        } else if (faces.length === 1) {
          const currentFacePan = faces[0].panAngle;
          guestTargetAngle = currentFacePan > 90 ? 60 : 120;
        }
      } else {
        updateTelemetry(
          {
            mood: resolvedIntent === "math_solve" ? "surprised" : "suspicious",
          },
          true
        );
      }

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
            recognizedCreatorInFrame: operatorInFrameRef.current,
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

        if (resolvedIntent === "guest_intro" && guestTargetAngle !== null) {
          setTimeout(() => {
            manualNeckHoldUntilRef.current = Date.now() + 6500;
            updateTelemetry({ pan: guestTargetAngle!, mood: "happy" }, true);
          }, 900);
        }

        const botLog: ChatLog = {
          id: `bot-${Date.now()}`,
          sender: "OMNI-BOT",
          text: botReply,
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
    ]
  );

  useEffect(() => {
    if (!spontaneousMonologue) return;
    const timer = setInterval(() => {
      if (isSpeakingRef.current || isThinkingRef.current) return;
      if (
        lastInteractionMsRef.current > 0 &&
        Date.now() - lastInteractionMsRef.current >= 30000
      ) {
        triggerOmniBrain(
          "Make a short observation about what is happening in front of your camera right now.",
          "autonomous_idle"
        );
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [spontaneousMonologue, triggerOmniBrain]);

  const startAlwaysOnRecognition = useCallback(async () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setMicStatusHint("Use Google Chrome for Always-On Voice!");
      return;
    }

    if (!micAudioCtxRef.current) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: selectedMicId ? { deviceId: { exact: selectedMicId } } : true,
        });
        micStreamRef.current = stream;
        const audioCtx = new AudioContext();
        micAudioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const checkMicVolume = () => {
          if (!micAudioCtxRef.current) return;
          analyser.getByteFrequencyData(dataArray);
          const avg =
            dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
          setMicLevel(Math.min(100, Math.round(avg * 2.2)));
          requestAnimationFrame(checkMicVolume);
        };
        requestAnimationFrame(checkMicVolume);
      } catch {
        // Ignore
      }
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // Ignore
      }
    }

    const recognition = new SpeechRecognition();
    recognition.lang = micLang;
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    let finalTranscript = "";

    recognition.onstart = () => {
      setIsListening(true);
      setMicStatusHint("Always-On Mic Active — Listening");
    };

    recognition.onresult = (event: any) => {
      if (isSpeakingRef.current) return;
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += chunk;
        } else {
          interim += chunk;
        }
      }
      setPromptInput(finalTranscript || interim);
    };

    recognition.onerror = () => {
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      const spoken = finalTranscript.trim();
      finalTranscript = "";

      if (spoken && !isSpeakingRef.current) {
        setMicStatusHint(`Heard: "${spoken}"`);
        setPromptInput("");
        triggerOmniBrain(spoken);
      } else if (
        alwaysOnMicRef.current &&
        !isSpeakingRef.current &&
        !isThinkingRef.current
      ) {
        setTimeout(() => {
          if (
            alwaysOnMicRef.current &&
            !isSpeakingRef.current &&
            !isThinkingRef.current
          ) {
            try {
              recognition.start();
            } catch {
              // Ignore
            }
          }
        }, 300);
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      // Ignore
    }
  }, [micLang, selectedMicId, triggerOmniBrain]);

  const toggleMicButton = useCallback(() => {
    if (isListening) {
      setAlwaysOnMic(false);
      alwaysOnMicRef.current = false;
      recognitionRef.current?.stop();
      setIsListening(false);
      setMicStatusHint("Mic Paused");
    } else {
      setAlwaysOnMic(true);
      alwaysOnMicRef.current = true;
      startAlwaysOnRecognition();
    }
  }, [isListening, startAlwaysOnRecognition]);

  const moods: EyeMood[] = [
    "normal",
    "happy",
    "suspicious",
    "surprised",
    "wink_left",
    "wink_right",
    "angry",
  ];

  return (
    <div className="min-h-screen bg-[#050811] text-slate-100 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* HEADER */}
        <header className="bg-slate-900/95 border border-slate-800 rounded-2xl px-5 py-3.5 flex flex-wrap items-center justify-between gap-4 shadow-xl">
          <div className="flex items-center gap-3.5">
            <div
              className={`h-3.5 w-3.5 rounded-full ${
                wsConnected
                  ? "bg-emerald-400 shadow-[0_0_12px_#34d399]"
                  : "bg-rose-500 animate-pulse"
              }`}
            />
            <div className="flex items-center gap-2.5">
              <h1 className="text-lg font-extrabold tracking-wide text-white">
                OMNI-BOT // CONTROL DECK
              </h1>
              <span className="text-[11px] font-mono font-bold px-2.5 py-0.5 rounded-md bg-sky-500/15 text-sky-300 border border-sky-500/30">
                {wsConnected ? "ESP32:ONLINE" : "ESP32:OFFLINE"}
              </span>
              <span className="text-[11px] font-mono font-bold px-2.5 py-0.5 rounded-md bg-purple-500/15 text-purple-300 border border-purple-500/30">
                TTS: {ttsEngineStatus}
              </span>
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
              AI: <strong className="text-emerald-400">{lastAiLatency}ms</strong>
            </div>

            <button
              onClick={() => setShowDebugDrawer(!showDebugDrawer)}
              className={`px-3.5 py-1 rounded-lg font-sans font-bold border transition ${
                showDebugDrawer
                  ? "bg-sky-500 text-slate-950 border-sky-400"
                  : "bg-slate-800 hover:bg-slate-700 text-sky-300 border-slate-700"
              }`}
            >
              {showDebugDrawer ? "✕ Close Config" : "🛠️ System Config & Debug"}
            </button>
          </div>
        </header>

        {/* SYSTEM CONFIG & FACE ENROLLMENT DRAWER */}
        {showDebugDrawer && (
          <div className="bg-slate-900/95 border border-sky-500/40 rounded-2xl p-4 space-y-4 text-xs shadow-2xl">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
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
                  Chatterbox Endpoint:
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
                  Collar Mic Device:
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

              <div>
                <label className="text-emerald-400 block mb-1 font-bold uppercase">
                  Operator Face Calibration:
                </label>
                <div className="flex gap-1.5">
                  <button
                    onClick={enrollOperatorFace}
                    className="flex-1 py-1.5 px-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 rounded-lg font-bold"
                  >
                    {operatorEnrolled ? "✓ Re-Scan Face" : "📸 Enroll Face"}
                  </button>
                  {operatorEnrolled && (
                    <button
                      onClick={clearOperatorEnrollment}
                      className="py-1.5 px-2 bg-rose-500/20 border border-rose-500/40 text-rose-300 rounded-lg font-bold"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 font-mono text-[11px] text-slate-400">
              <div>
                LAST WS PACKET:{" "}
                <code className="text-sky-300 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                  {lastWsPayload}
                </code>
              </div>
              <div className="flex flex-wrap items-center gap-4">
                <span>
                  OPERATOR IN FRAME:{" "}
                  <strong
                    className={
                      operatorDetectedNow
                        ? "text-emerald-400"
                        : "text-slate-500"
                    }
                  >
                    {operatorDetectedNow ? "YES" : "NO"}
                  </strong>
                </span>
                <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 font-sans">
                  <input
                    type="checkbox"
                    checked={spontaneousMonologue}
                    onChange={(e) => setSpontaneousMonologue(e.target.checked)}
                    className="accent-sky-400"
                  />
                  Live Camera Idle Comments (30s)
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 font-sans">
                  <input
                    type="checkbox"
                    checked={offlineMode}
                    onChange={(e) => setOfflineMode(e.target.checked)}
                    className="accent-amber-400"
                  />
                  Offline Mode
                </label>
              </div>
            </div>
          </div>
        )}

        {/* MAIN WORKSPACE */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* LEFT COLUMN: CAMERA HUD + CONTROLS */}
          <div className="lg:col-span-6 flex flex-col gap-4">
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300">
                  OPTICAL SENSOR FEED ({facesDetected} FACE
                  {facesDetected === 1 ? "" : "S"} DETECTED)
                </span>
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
                      "object_inspect"
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
                      "Read the math equation or question written on this paper/board and solve it!",
                      "math_solve"
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
                      "guest_intro"
                    )
                  }
                  disabled={isThinking}
                  className="py-2.5 px-2.5 rounded-xl bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-200 text-xs font-bold transition flex items-center justify-center gap-1.5"
                >
                  <span>🤝</span> INTRODUCE GUEST
                </button>
              </div>
            </div>

            {/* SERVO OVERRIDE & VU */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 space-y-4 shadow-xl">
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
                    <span>ARC REACTOR FFT VU</span>
                    <span className="font-mono text-sky-400">
                      {vuVolume}/16 LEDs
                    </span>
                  </div>
                  <div className="grid grid-cols-16 gap-1 h-5 bg-slate-950 p-1 rounded-lg border border-slate-800">
                    {Array.from({ length: 16 }).map((_, idx) => (
                      <div
                        key={idx}
                        className={`rounded-xs transition-all ${
                          idx < vuVolume
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

              <div>
                <span className="text-xs font-bold text-slate-300 block mb-2">
                  DUAL OLED EYE EXPRESSIONS:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {moods.map((m) => (
                    <button
                      key={m}
                      onClick={() => updateTelemetry({ mood: m }, true)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase border transition ${
                        currentMood === m
                          ? "bg-sky-500 text-slate-950 border-sky-400"
                          : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {m.replace("_", " ")}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: VOICE + CHAT STREAM */}
          <div className="lg:col-span-6 flex flex-col bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl justify-between gap-4">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300">
                    VOICE & CHAT STREAM
                  </span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-sky-300">
                    {activeIntentBadge === "guest_intro"
                      ? "🤝 GUEST INTRO PIVOT"
                      : activeIntentBadge === "math_solve"
                      ? "📐 MATH OCR"
                      : activeIntentBadge === "object_inspect"
                      ? "✋ OBJECT ID"
                      : "💬 CHATTERBOX MULTILINGUAL"}
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
                    EN-IN
                  </button>
                  <button
                    onClick={() => setMicLang("hi-IN")}
                    className={`px-2 py-0.5 rounded ${
                      micLang === "hi-IN"
                        ? "bg-sky-500 text-slate-950"
                        : "text-slate-400"
                    }`}
                  >
                    HI-IN
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <button
                  onClick={toggleMicButton}
                  disabled={isThinking}
                  className={`w-full py-3.5 px-4 rounded-xl font-extrabold text-sm border transition flex items-center justify-center gap-2.5 shadow-lg ${
                    isListening
                      ? "bg-emerald-500/25 border-emerald-400 text-emerald-200 animate-pulse"
                      : isSpeaking
                      ? "bg-sky-500/20 border-sky-400 text-sky-200"
                      : "bg-sky-400 hover:bg-sky-300 text-slate-950 border-transparent"
                  }`}
                >
                  {isListening
                    ? "🎙️ ALWAYS-ON MIC ACTIVE (SPEAK ANYTIME)"
                    : isThinking
                    ? "⚡ PROCESSING..."
                    : isSpeaking
                    ? "🔊 SPEAKING VIA CHATTERBOX..."
                    : "🎙️ START ALWAYS-ON COLLAR MIC"}
                </button>

                <div className="flex items-center gap-2 px-1">
                  <div className="w-28 h-1.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className="h-full bg-emerald-400 transition-all duration-75"
                      style={{ width: `${micLevel}%` }}
                    />
                  </div>
                  <span className="text-[11px] text-slate-400 truncate font-mono">
                    {micStatusHint}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5 pt-1">
                {[
                  {
                    label: "🤝 Introduce Teacher",
                    text: "Hey Omni, meet my class 10 teacher standing here with me!",
                  },
                  {
                    label: "🇮🇳 Hindi Chat Test",
                    text: "ओमनी भाई, आज एग्जीबिशन में कैसा लग रहा है?",
                  },
                  {
                    label: "✋ Inspect Object",
                    text: "Can you see and tell me what I am holding in my hand?",
                  },
                  {
                    label: "👀 Look Right",
                    text: "Omni, look to your right!",
                  },
                ].map((item) => (
                  <button
                    key={item.label}
                    onClick={() => triggerOmniBrain(item.text)}
                    className="text-xs bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 px-2.5 py-1.5 rounded-lg font-medium transition"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* CHAT LOGS */}
            <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-3.5 h-80 overflow-y-auto space-y-2.5">
              {chatLogs.map((log) => (
                <div
                  key={log.id}
                  className={`p-3 rounded-xl text-xs leading-relaxed border ${
                    log.sender === "OMNI-BOT"
                      ? "bg-slate-900 border-sky-500/30 text-slate-100"
                      : "bg-slate-900/40 border-slate-800 text-slate-300 ml-6"
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-mono">
                    <span className="font-bold text-sky-400">
                      {log.sender}
                      {log.intent === "guest_intro" &&
                        " • [🤝 GUEST INTRO PIVOT]"}
                      {log.mood ? ` • [${log.mood.toUpperCase()}]` : ""}
                      {log.latencyMs ? ` • ${log.latencyMs}ms` : ""}
                    </span>
                    <span>{log.timestamp}</span>
                  </div>
                  <p className="text-[13px]">{log.text}</p>
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
                placeholder="Say 'Hey Omni, meet my class 10 teacher', or ask in English/Hindi..."
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