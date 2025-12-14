import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Mock taxa database for classification
const taxaDatabase = [
  { name: "Thunnus albacares", class: "Teleostei", confidence: 0.95 },
  { name: "Carcharodon carcharias", class: "Chondrichthyes", confidence: 0.92 },
  { name: "Tursiops truncatus", class: "Mammalia", confidence: 0.98 },
  { name: "Chelonia mydas", class: "Reptilia", confidence: 0.94 },
  { name: "Octopus vulgaris", class: "Cephalopoda", confidence: 0.89 },
  { name: "Aurelia aurita", class: "Cnidaria", confidence: 0.87 },
  { name: "Penaeus monodon", class: "Crustacea", confidence: 0.91 },
  { name: "Rhincodon typus", class: "Chondrichthyes", confidence: 0.96 },
];

const endangeredSpecies = ["Rhincodon typus", "Chelonia mydas", "Carcharodon carcharias"];
const invasiveSpecies = ["Pterois volitans", "Caulerpa taxifolia"];

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { upload_id } = await req.json();
    
    if (!upload_id) {
      throw new Error("upload_id is required");
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    console.log(`Processing upload: ${upload_id}`);

    // Update upload status to processing
    await supabase
      .from("uploads")
      .update({ status: "processing" })
      .eq("id", upload_id);

    // Simulate ASV generation (in real pipeline, this would be DADA2/Deblur)
    const numASVs = Math.floor(Math.random() * 50) + 20;
    const asvs = Array.from({ length: numASVs }, (_, i) => ({
      id: `ASV_${String(i + 1).padStart(4, "0")}`,
      sequence: generateRandomSequence(250),
      reads: Math.floor(Math.random() * 10000) + 100,
      length: 250,
    }));

    // Simulate taxonomy classification
    const taxa = asvs.map((asv) => {
      const randomTaxa = taxaDatabase[Math.floor(Math.random() * taxaDatabase.length)];
      const isNovel = Math.random() > 0.85;
      return {
        asv_id: asv.id,
        species: isNovel ? "Unknown sp." : randomTaxa.name,
        class: randomTaxa.class,
        confidence: isNovel ? Math.random() * 0.5 + 0.3 : randomTaxa.confidence,
        is_novel: isNovel,
        pct_identity: isNovel ? Math.random() * 30 + 50 : Math.random() * 5 + 95,
      };
    });

    // Calculate novelty scores
    const noveltyScores = taxa
      .filter((t) => t.is_novel || t.pct_identity < 85)
      .map((t) => ({
        asv_id: t.asv_id,
        novelty_score: 1 - t.pct_identity / 100,
        cluster_cohesion: Math.random() * 0.5 + 0.5,
        reproducibility: Math.random() * 0.4 + 0.6,
      }));

    // Calculate biodiversity indices
    const speciesCounts = taxa.reduce((acc, t) => {
      acc[t.species] = (acc[t.species] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const totalCount = Object.values(speciesCounts).reduce((a, b) => a + b, 0);
    const proportions = Object.values(speciesCounts).map((c) => c / totalCount);
    
    const shannonIndex = -proportions.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
    const simpsonIndex = 1 - proportions.reduce((sum, p) => sum + p * p, 0);
    const chao1 = Object.keys(speciesCounts).length + (noveltyScores.length * (noveltyScores.length - 1)) / 2;

    const biodiversityIndices = {
      shannon: Math.round(shannonIndex * 100) / 100,
      simpson: Math.round(simpsonIndex * 100) / 100,
      chao1: Math.round(chao1),
      richness: Object.keys(speciesCounts).length,
      evenness: Math.round((shannonIndex / Math.log(Object.keys(speciesCounts).length)) * 100) / 100,
    };

    // Abundance data
    const abundanceData = {
      total_reads: asvs.reduce((sum, a) => sum + a.reads, 0),
      unique_asvs: asvs.length,
      class_distribution: Object.entries(
        taxa.reduce((acc, t) => {
          acc[t.class] = (acc[t.class] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
      ).map(([name, count]) => ({ name, count })),
    };

    // Insert analysis results
    const { error: resultsError } = await supabase.from("analysis_results").insert({
      upload_id,
      asvs,
      taxa,
      novelty_scores: noveltyScores,
      biodiversity_indices: biodiversityIndices,
      abundance_data: abundanceData,
      summary: `Analysis complete. Found ${asvs.length} ASVs across ${Object.keys(speciesCounts).length} species. Shannon diversity: ${biodiversityIndices.shannon}. ${noveltyScores.length} potential novel sequences detected.`,
      provenance: {
        pipeline_version: "1.0.0",
        model: "DNABERT-mock",
        timestamp: new Date().toISOString(),
      },
    });

    if (resultsError) throw resultsError;

    // Generate alerts for endangered/invasive/novel species
    const alerts = [];
    
    for (const t of taxa) {
      if (endangeredSpecies.includes(t.species)) {
        alerts.push({
          upload_id,
          alert_type: "endangered",
          severity: "high",
          title: `Endangered Species Detected: ${t.species}`,
          description: `Endangered species ${t.species} detected with ${Math.round(t.confidence * 100)}% confidence.`,
          species_name: t.species,
          status: "active",
        });
      }
    }

    for (const ns of noveltyScores.filter((n) => n.novelty_score > 0.3)) {
      alerts.push({
        upload_id,
        alert_type: "novel_species",
        severity: ns.novelty_score > 0.5 ? "critical" : "medium",
        title: `Potential Novel Species: ${ns.asv_id}`,
        description: `Novel sequence detected with ${Math.round(ns.novelty_score * 100)}% novelty score. Requires taxonomic verification.`,
        species_name: `Unknown (${ns.asv_id})`,
        status: "active",
      });
    }

    if (alerts.length > 0) {
      await supabase.from("alerts").insert(alerts);
    }

    // Update upload status to completed
    await supabase
      .from("uploads")
      .update({ status: "completed" })
      .eq("id", upload_id);

    console.log(`Processing complete for upload: ${upload_id}`);

    return new Response(JSON.stringify({
      success: true,
      upload_id,
      summary: {
        asvs_count: asvs.length,
        species_count: Object.keys(speciesCounts).length,
        novel_candidates: noveltyScores.length,
        alerts_generated: alerts.length,
        biodiversity_indices: biodiversityIndices,
      },
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error: unknown) {
    console.error("Processing error:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});

function generateRandomSequence(length: number): string {
  const bases = ["A", "T", "G", "C"];
  return Array.from({ length }, () => bases[Math.floor(Math.random() * 4)]).join("");
}
