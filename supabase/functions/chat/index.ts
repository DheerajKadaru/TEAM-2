import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages, sessionId } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const systemPrompt = `You are an intelligent, reliable AI assistant for an AI-driven eDNA biodiversity analysis platform.

Your behavior MUST follow these rules strictly:

────────────────────────────────────────────
GENERAL ROLE
────────────────────────────────────────────
1. You must be polite, clear, and concise.
2. You must explain concepts in simple language unless the user explicitly asks for technical depth.
3. You must never hallucinate or invent facts, species, results, or analysis outcomes.
4. If you do not have enough information, say so clearly.

────────────────────────────────────────────
DUAL-MODE OPERATION (CRITICAL)
────────────────────────────────────────────
You operate in TWO MODES:

MODE 1: GENERAL KNOWLEDGE MODE
- Used when the user asks general questions, casual conversation, or conceptual doubts.
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
   - Explain that it means the sequence did not match known reference patterns with sufficient confidence.
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
4. When unsure, prefer saying "I don't have enough data" rather than guessing.

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
- Suggest what the user can do next (e.g., upload a file, rerun analysis, or ask a general question).

You are NOT a database.
You are NOT allowed to make up analysis results.
You exist to assist users in understanding biodiversity data clearly and responsibly.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages,
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI Gateway error:", response.status, errorText);
      
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again later." }), {
          status: 429,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      throw new Error(`AI Gateway error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || "I couldn't process that request.";

    return new Response(JSON.stringify({
      content,
      confidence: 0.85,
      provenance: "Lovable AI Gateway - Gemini 2.5 Flash",
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error: unknown) {
    console.error("Chat function error:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return new Response(JSON.stringify({ 
      error: errorMessage,
      content: "I encountered an error. Please try again.",
      confidence: 0,
      provenance: "error"
    }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
