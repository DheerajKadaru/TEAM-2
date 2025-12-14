import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, Loader2, Sparkles, AlertTriangle, Dna, Activity, BarChart2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { supabase } from "@/integrations/supabase/client";
import { v4 as uuidv4 } from "uuid";
import { GoogleGenerativeAI } from "@google/generative-ai";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

interface ExplanationData {
  type: "shap" | "lime";
  features: { name: string; value: number; color: string }[];
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  provenance?: string;
  explanation?: ExplanationData;
}

const SeaSageChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm SeaSage, your intelligent AI assistant for eDNA biodiversity analysis. I can help you understand eDNA concepts, explain analysis results, and answer questions about your data. Ask me anything - I'll be honest about what I know and what I don't.",
      confidence: 1,
      provenance: "System",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => uuidv4());
  const [expandedExplanationId, setExpandedExplanationId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, expandedExplanationId]);

  const generateGeminiResponse = async (history: Message[], userInput: string) => {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY || "";
    if (!apiKey) {
      throw new Error("Gemini API key not configured (VITE_GEMINI_API_KEY)");
    }

    const systemPrompt = `You are an intelligent, reliable AI assistant for an AI-driven
eDNA biodiversity analysis platform.

Your behavior MUST follow these rules strictly:

────────────────────────────────────────────
GENERAL ROLE
────────────────────────────────────────────
1. You must be polite, clear, and concise.
2. You must explain concepts in simple language unless the user
   explicitly asks for technical depth.
3. You must never hallucinate or invent facts, species, results,
   or analysis outcomes.
4. If you do not have enough information, say so clearly.

────────────────────────────────────────────
DUAL-MODE OPERATION (CRITICAL)
────────────────────────────────────────────
You operate in TWO MODES:

MODE 1: GENERAL KNOWLEDGE MODE
- Used when the user asks general questions, casual conversation,
  or conceptual doubts.
- Examples:
  • "What is eDNA?"
  • "How does DNABERT work?"
  • "Explain biodiversity analysis"
- In this mode:
  • Answer using general scientific knowledge.
  • Do NOT assume any file has been analyzed.
  • Do NOT mention results unless explicitly provided.

MODE 2: RESULT-AWARE MODE
- Used ONLY when analysis results are provided in the context.
- Examples:
  • "What species were detected?"
  • "Why is this sequence marked unknown?"
  • "Explain the abundance results"
- In this mode:
  • Answer STRICTLY using the provided analysis data.
  • Do NOT guess or infer beyond the given results.
  • If a value or species is not present, say:
    "That information is not available in the current analysis."

────────────────────────────────────────────
RESULT HANDLING RULES (VERY IMPORTANT)
────────────────────────────────────────────
1. When analysis results are provided:
   - Treat them as the single source of truth.
   - Do not contradict them.
2. If a species is labeled "Unknown":
   - Explain that it means the sequence did not match
     known reference patterns with sufficient confidence.
3. If confidence values are present:
   - Explain them in probabilistic terms.
   - Never claim 100% certainty.
4. If no results are provided and the user asks about results:
   - Respond with:
     "Please upload and analyze a file first so I can answer that."

────────────────────────────────────────────
SAFETY & HONESTY
────────────────────────────────────────────
1. Never invent species names, confidence scores, or counts.
2. Never claim database matches that are not explicitly provided.
3. Never overstate accuracy or scientific certainty.
4. When unsure, prefer saying "I don't have enough data" rather
   than guessing.

────────────────────────────────────────────
TONE & UX
────────────────────────────────────────────
- Be friendly but professional.
- Avoid unnecessary emojis.
- Prefer short, structured answers.
- Use bullet points when explaining results.

────────────────────────────────────────────
FALLBACK BEHAVIOR
────────────────────────────────────────────
If you cannot confidently answer a question:
- Say so honestly.
- Suggest what the user can do next (e.g., upload a file,
  rerun analysis, or ask a general question).

You are NOT a database.
You are NOT allowed to make up analysis results.
You exist to assist users in understanding biodiversity data
clearly and responsibly.`;

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    // Build history with system prompt
    const chatHistory = [
      {
        role: 'user' as const,
        parts: [{ text: systemPrompt }]
      },
      {
        role: 'model' as const,
        parts: [{ text: "I understand. I will follow these rules strictly and operate in the appropriate mode based on whether analysis results are present in the conversation." }]
      },
      ...history.map(m => ({
        role: (m.role === 'assistant' ? 'model' : 'user') as 'user' | 'model',
        parts: [{ text: m.content }]
      }))
    ];

    // Convert history to Gemini format
    const chat = model.startChat({
      history: chatHistory,
    });

    const result = await chat.sendMessage(userInput);
    const response = await result.response;
    return response.text();
  };

  // Explanation logic now handled by Python backend

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Call local Python backend
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          sessionId,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Backend error ${response.status}:`, errorText);
        throw new Error(`Backend error: ${response.status} - ${errorText.substring(0, 100)}`);
      }

      const data = await response.json();
      
      if (!data || !data.content) {
        throw new Error("Invalid response from backend: missing content");
      }

      const assistantMessage: Message = {
        id: uuidv4(),
        role: "assistant",
        content: data.content,
        confidence: data.confidence,
        provenance: data.provenance,
        explanation: data.explanation // Real xAI data from Python backend
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (backendError) {
      console.error("Python backend failed:", backendError);
      
      // Show user-friendly error message
      const errorMessage = backendError instanceof Error 
        ? backendError.message 
        : "Failed to connect to backend";
      
      // Check if it's a connection error
      if (errorMessage.includes("Failed to fetch") || errorMessage.includes("NetworkError")) {
        setMessages((prev) => [
          ...prev,
          {
            id: uuidv4(),
            role: "assistant",
            content: "I cannot connect to the backend server. Please make sure the Python backend is running on http://localhost:8000. You can start it with: uvicorn backend.main:app --reload",
            confidence: 0,
            provenance: "System Error",
          },
        ]);
        setIsLoading(false);
        return;
      }
      
      console.warn("Attempting Gemini fallback:", backendError);

      try {
        // Fallback to direct Gemini call
        const geminiText = await generateGeminiResponse(messages, userMessage.content);

        const fallbackMessage: Message = {
          id: uuidv4(),
          role: "assistant",
          content: geminiText,
          confidence: 0.9,
          provenance: "Google Gemini (Fallback)",
          // explanation: SimulatExplanation() // removed in favor of backend
        };

        setMessages((prev) => [...prev, fallbackMessage]);
      } catch (geminiError) {
        console.error("Gemini fallback failed:", geminiError);
        setMessages((prev) => [
          ...prev,
          {
            id: uuidv4(),
            role: "assistant",
            content: "I'm having trouble connecting to the cloud. However, I can still analyze 'salinity', 'pH', or 'depth' data using my local neural engine. Try asking for a project analysis!",
            confidence: 0,
            provenance: "System Offline Mode",
          },
        ]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 z-50 w-16 h-16 rounded-full btn-ocean flex items-center justify-center shadow-[0_0_30px_hsla(199,89%,48%,0.6)] hover:shadow-[0_0_50px_hsla(199,89%,48%,0.8)] transition-all duration-300 animate-glow group border-2 border-primary/50"
      >
        <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping opacity-20" />
        {/* DNA helix decorative rings */}
        <div
          className="absolute inset-0 border-2 border-primary/40 rounded-full animate-spin opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ animationDuration: "10s" }}
        />
        <div
          className="absolute inset-0 border border-primary/30 rounded-full animate-spin opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ animationDuration: "8s", animationDirection: "reverse" }}
        />
        {isOpen ? (
          <X className="w-8 h-8 relative z-10 text-white drop-shadow-[0_0_5px_rgba(255,255,255,0.8)]" />
        ) : (
          <div className="relative flex items-center justify-center">
            <Dna className="w-8 h-8 text-white absolute opacity-90" strokeWidth={1.5} />
            <img
              src="/dna-icon.png"
              alt="DNA"
              className="w-5 h-5 object-contain relative z-10 drop-shadow-[0_0_8px_rgba(255,255,255,1)] translate-y-[0px]"
            />
          </div>
        )}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-28 right-6 w-[400px] max-w-[calc(100vw-2rem)] h-[600px] max-h-[calc(100vh-8rem)] glass-card rounded-3xl shadow-2xl fade-in-up overflow-hidden z-40 flex flex-col border border-white/10 bg-black/60 backdrop-blur-2xl ring-1 ring-white/10">
          {/* Animated Gradient Border */}
          <div className="absolute inset-0 pointer-events-none rounded-3xl border border-transparent [mask-clip:padding-box,border-box] [background-clip:padding-box,border-box] [mask-image:linear-gradient(#fff,transparent)] after:absolute after:-inset-[1px] after:rounded-3xl after:bg-gradient-to-r after:from-primary/30 after:via-cyan-400/20 after:to-primary/30 after:animate-border-flow" />

          {/* DNA helix pattern overlay */}
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none mix-blend-screen">
            <div className="absolute inset-0 bg-[url('/dna-icon.png')] bg-repeat opacity-20 bg-[length:100px_100px]" />
          </div>

          {/* Header */}
          <div className="p-5 border-b border-white/5 relative z-10 bg-gradient-to-r from-primary/10 via-transparent to-primary/5 backdrop-blur-3xl">
            <div className="flex items-center gap-4">
              <div className="relative group cursor-pointer">
                <div className="relative z-10 p-2 bg-gradient-to-br from-primary/20 to-cyan-500/10 rounded-xl border border-white/10 shadow-[0_0_15px_rgba(14,165,233,0.3)] transition-transform group-hover:scale-105 duration-300">
                  <img
                    src="/dna-icon.png"
                    alt="DNA Icon"
                    className="w-8 h-8 object-contain filter drop-shadow-[0_0_5px_rgba(14,165,233,0.8)]"
                  />
                </div>
                <div className="absolute inset-0 bg-primary/40 blur-xl rounded-full animate-pulse opacity-50 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="flex-1">
                <h3 className="font-display font-bold text-lg text-white tracking-wide drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)] flex items-center gap-2">
                  SeaSage AI
                  <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                </h3>
                <p className="text-[11px] uppercase tracking-widest text-primary-foreground/60 font-medium">
                  Marine Intelligence
                </p>
              </div>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-white/50 hover:text-white hover:bg-white/10 rounded-full transition-colors" onClick={() => setIsOpen(false)}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <ScrollArea
            className="flex-1 p-5 custom-scrollbar relative z-10 bg-transparent"
            ref={scrollRef}
          >
            <div className="space-y-6 pb-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
                >
                  <div
                    className={`relative max-w-[85%] p-4 rounded-2xl shadow-lg border backdrop-blur-xl transition-all duration-300 hover:shadow-primary/5 ${message.role === "user"
                      ? "bg-gradient-to-br from-primary/30 to-blue-600/20 rounded-tr-sm border-white/10 text-white"
                      : "bg-gradient-to-br from-zinc-800/80 to-zinc-900/80 rounded-tl-sm border-white/5 text-gray-100"
                      }`}
                  >
                    <p className="text-sm relative z-10 leading-relaxed whitespace-pre-wrap font-medium tracking-wide/0 opacity-95">{message.content}</p>

                    {/* xAI Explanation Button & Chart */}
                    {message.role === "assistant" && message.explanation && (
                      <div className="mt-4 relative z-10">
                        <div className="flex items-center justify-between border-t border-white/5 pt-3 mb-2">
                          <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-primary/70 font-bold">
                            {message.provenance && <span>{message.provenance}</span>}
                            {message.confidence && <span className="text-emerald-400">• {Math.round(message.confidence * 100)}% Match</span>}
                          </div>
                        </div>

                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setExpandedExplanationId(expandedExplanationId === message.id ? null : message.id)}
                          className="w-full text-xs bg-white/5 hover:bg-primary/20 border-white/10 hover:border-primary/30 text-gray-300 hover:text-white h-8 flex items-center justify-center gap-2 transition-all group"
                        >
                          {expandedExplanationId === message.id ? <X className="w-3 h-3 text-red-400" /> : <Activity className="w-3 h-3 text-primary group-hover:animate-pulse" />}
                          {expandedExplanationId === message.id ? "Hide Analytics" : "View DNA Analysis"}
                        </Button>

                        {expandedExplanationId === message.id && (
                          <div className="mt-3 p-4 bg-black/50 rounded-xl border border-white/10 animate-in fade-in zoom-in-95 duration-300 shadow-inner">
                            <h4 className="text-[10px] uppercase tracking-widest text-cyan-400 mb-3 font-bold flex items-center gap-2 border-b border-white/5 pb-2">
                              <BarChart2 className="w-3 h-3" /> Genomic Feature Impact
                            </h4>
                            <div className="h-[160px] w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={message.explanation.features} layout="vertical" margin={{ top: 5, right: 30, left: 45, bottom: 5 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
                                  <XAxis type="number" hide />
                                  <YAxis
                                    dataKey="name"
                                    type="category"
                                    tick={{ fill: '#9ca3af', fontSize: 10, fontWeight: 500 }}
                                    width={55}
                                    axisLine={false}
                                    tickLine={false}
                                  />
                                  <Tooltip
                                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                                    contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '11px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.5)' }}
                                    itemStyle={{ color: '#e2e8f0' }}
                                  />
                                  <Bar dataKey="value" radius={[2, 2, 2, 2]} barSize={12}>
                                    {message.explanation.features.map((entry, index) => (
                                      <Cell key={`cell-${index}`} fill={entry.color} stroke={entry.color} strokeOpacity={0.8} />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                            <div className="flex justify-between items-center mt-3 text-[9px] font-medium text-gray-500 uppercase tracking-widest">
                              <span className="text-red-400/80">← Deviation</span>
                              <span className="text-cyan-400/80">Support →</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start animate-pulse">
                  <div className="bg-white/5 rounded-2xl rounded-tl-sm p-4 backdrop-blur-md border border-white/5 flex items-center gap-3">
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    <span className="text-xs text-muted-foreground font-medium tracking-wide">Analysing Sequence...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="p-4 border-t border-white/5 relative z-10 bg-gradient-to-t from-black/80 to-transparent backdrop-blur-sm">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
              className="relative flex items-center gap-2"
            >
              <div className="relative flex-1 group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/30 to-cyan-400/30 rounded-xl check-opacity blur opacity-30 group-hover:opacity-60 transition duration-500"></div>
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask SeaSage..."
                  className="relative w-full bg-black/60 border-white/10 focus:border-primary/50 text-white placeholder:text-gray-500 rounded-xl h-12 pl-4 pr-12 shadow-inner font-light tracking-wide transition-all focus:ring-1 focus:ring-primary/20"
                  disabled={isLoading}
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
                  {/* Decorative elements inside input */}
                  <div className="w-1 h-1 rounded-full bg-primary/40 animate-pulse delay-75" />
                  <div className="w-1 h-1 rounded-full bg-primary/40 animate-pulse delay-150" />
                </div>
              </div>
              <Button
                type="submit"
                size="icon"
                disabled={isLoading || !input.trim()}
                className="h-12 w-12 bg-primary text-primary-foreground rounded-xl shadow-[0_0_20px_rgba(14,165,233,0.4)] hover:shadow-[0_0_30px_rgba(14,165,233,0.6)] hover:bg-primary/90 transition-all duration-300 hover:scale-105 active:scale-95 border border-white/10"
              >
                <Send className="w-5 h-5 ml-0.5" />
              </Button>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default SeaSageChatbot;
