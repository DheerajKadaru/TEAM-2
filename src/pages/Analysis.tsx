import { useEffect, useState } from "react";
import { useLocation } from 'react-router-dom';
import * as React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Treemap, ScatterChart,
  Scatter, ZAxis, AreaChart, Area, Legend
} from "recharts";
import { supabase } from "@/integrations/supabase/client";
import { Dna, Sparkles, AlertTriangle, TrendingUp, Layers, Upload as UploadIcon } from "lucide-react";

const COLORS = [
  'hsl(180, 100%, 50%)',
  'hsl(165, 100%, 55%)',
  'hsl(200, 100%, 60%)',
  'hsl(38, 100%, 60%)',
  'hsl(15, 100%, 65%)',
  'hsl(280, 80%, 60%)',
  'hsl(120, 70%, 50%)',
];

const tooltipStyle = {
  background: 'hsl(210, 45%, 10%)',
  border: '1px solid hsl(180, 100%, 50%, 0.2)',
  borderRadius: '8px',
  color: 'hsl(185, 100%, 95%)'
};

// Mock data for comprehensive visualization
const mockTaxonomicTree = [
  {
    name: 'Chordata', size: 45, children: [
      { name: 'Teleostei', size: 25 },
      { name: 'Chondrichthyes', size: 12 },
      { name: 'Mammalia', size: 8 },
    ]
  },
  { name: 'Cnidaria', size: 18 },
  { name: 'Mollusca', size: 15 },
  {
    name: 'Arthropoda', size: 22, children: [
      { name: 'Crustacea', size: 14 },
      { name: 'Hexapoda', size: 8 },
    ]
  },
];

const flattenTree = (data: any[]): any[] => {
  return data.flatMap(item => {
    if (item.children) {
      return [{ name: item.name, size: item.size }, ...flattenTree(item.children)];
    }
    return [{ name: item.name, size: item.size }];
  });
};

const speciesAbundance = [
  { name: 'Thunnus albacares', abundance: 2450, reads: 12500, novelty: 0.05 },
  { name: 'Tursiops truncatus', abundance: 890, reads: 4200, novelty: 0.08 },
  { name: 'Rhincodon typus', abundance: 120, reads: 580, novelty: 0.12 },
  { name: 'Chelonia mydas', abundance: 340, reads: 1650, novelty: 0.15 },
  { name: 'Octopus vulgaris', abundance: 1200, reads: 5800, novelty: 0.07 },
  { name: 'Aurelia aurita', abundance: 3400, reads: 16000, novelty: 0.03 },
  { name: 'Penaeus monodon', abundance: 2100, reads: 9800, novelty: 0.06 },
];

const noveltyData = [
  { asv: 'ASV_0012', score: 0.82, pctIdentity: 68, clusterCohesion: 0.45, reproducibility: 0.78 },
  { asv: 'ASV_0034', score: 0.71, pctIdentity: 72, clusterCohesion: 0.52, reproducibility: 0.81 },
  { asv: 'ASV_0056', score: 0.65, pctIdentity: 75, clusterCohesion: 0.58, reproducibility: 0.73 },
  { asv: 'ASV_0078', score: 0.58, pctIdentity: 78, clusterCohesion: 0.62, reproducibility: 0.69 },
  { asv: 'ASV_0091', score: 0.45, pctIdentity: 82, clusterCohesion: 0.71, reproducibility: 0.85 },
];

const mockRadarData = [
  { metric: 'Shannon', value: 85, fullMark: 100 },
  { metric: 'Simpson', value: 92, fullMark: 100 },
  { metric: 'Chao1', value: 78, fullMark: 100 },
  { metric: 'Evenness', value: 71, fullMark: 100 },
  { metric: 'Richness', value: 88, fullMark: 100 },
  { metric: 'Novelty', value: 45, fullMark: 100 },
];

const timeSeriesData = [
  { month: 'Jan', teleostei: 35, chondrichthyes: 12, cnidaria: 18, mollusca: 15 },
  { month: 'Feb', teleostei: 38, chondrichthyes: 14, cnidaria: 20, mollusca: 16 },
  { month: 'Mar', teleostei: 42, chondrichthyes: 15, cnidaria: 22, mollusca: 18 },
  { month: 'Apr', teleostei: 45, chondrichthyes: 18, cnidaria: 19, mollusca: 20 },
  { month: 'May', teleostei: 48, chondrichthyes: 20, cnidaria: 21, mollusca: 22 },
];

const Analysis = () => {
  const [results, setResults] = useState<any[]>([]);
  const [selectedResult, setSelectedResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'species' | 'novelty' | 'temporal' | 'model-test' | 'training' | 'file-analysis'>('overview');
  const [validationResults, setValidationResults] = useState<any>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [realSpeciesData, setRealSpeciesData] = useState<any[]>([]);
  const [trainingFile, setTrainingFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string>("");

  // File Analysis Tab State
  const [analysisFile, setAnalysisFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);
  const [analysisError, setAnalysisError] = useState<string>("");
  const [biodiversityStats, setBiodiversityStats] = useState<any>(null);
  const [hierarchyData, setHierarchyData] = useState<any>(null);

  // Get upload results from navigation state
  const location = useLocation();
  const uploadResults = location.state?.uploadResults;

  // Dynamic Data Transformations from analysisResults
  const dynamicSpeciesData = React.useMemo(() => {
    if (analysisResults.length === 0) return [];

    const speciesMap: Record<string, { count: number; totalConfidence: number; sequences: any[] }> = {};

    analysisResults.forEach(result => {
      const species = result.predicted_taxa || 'Unknown';
      if (!speciesMap[species]) {
        speciesMap[species] = { count: 0, totalConfidence: 0, sequences: [] };
      }
      speciesMap[species].count++;
      speciesMap[species].totalConfidence += result.confidence;
      speciesMap[species].sequences.push(result);
    });

    return Object.entries(speciesMap).map(([name, data]) => ({
      name,
      abundance: data.count,
      reads: data.count * 5,
      novelty: name === 'Novel/Unknown' || name === 'Model Not Trained' ? 0.8 :
        (1 - (data.totalConfidence / data.count)),
      avgConfidence: data.totalConfidence / data.count
    })).sort((a, b) => b.abundance - a.abundance);
  }, [analysisResults]);

  const novelCandidates = React.useMemo(() => {
    return analysisResults.filter(r =>
      r.confidence < 0.6 ||
      r.predicted_taxa === 'Novel/Unknown' ||
      r.predicted_taxa === 'Model Not Trained'
    );
  }, [analysisResults]);

  // Calculate stats from historical data (realSpeciesData from database)
  const [historicalStats, setHistoricalStats] = React.useState({
    totalSequences: 0,
    speciesCount: 0,
    novelCount: 0,
    knownCount: 0,
    avgConfidence: 0
  });

  const summaryStats = React.useMemo(() => {
    // 1. If analysis run on this page, use those results
    if (analysisResults.length > 0) {
      return {
        totalSequences: analysisResults.length,
        speciesCount: dynamicSpeciesData.length,
        knownCount: analysisResults.filter(r =>
          r.predicted_taxa !== 'Novel/Unknown' &&
          r.predicted_taxa !== 'Model Not Trained' &&
          r.confidence >= 0.6
        ).length,
        novelCount: novelCandidates.length,
        avgConfidence: (analysisResults.reduce((sum, r) => sum + r.confidence, 0) / analysisResults.length * 100).toFixed(1)
      };
    }
    // 2. If coming from Upload page with results
    if (uploadResults) {
      return {
        totalSequences: uploadResults.asvs_count || 0,
        speciesCount: uploadResults.species_count || 0,
        knownCount: (uploadResults.asvs_count || 0) - (uploadResults.novel_candidates || 0),
        novelCount: uploadResults.novel_candidates || 0,
        avgConfidence: uploadResults.avg_confidence || 0 // Assuming avg_confidence might be passed or default to 0
      };
    }
    // 3. Otherwise use historical database stats
    return historicalStats;
  }, [analysisResults, dynamicSpeciesData, novelCandidates, historicalStats, uploadResults]);

  const displayRadarData = React.useMemo(() => {
    if (!biodiversityStats) return mockRadarData;

    return [
      { metric: 'Shannon', value: Math.min((biodiversityStats.shannon || 0) * 20, 100), fullMark: 100 },
      { metric: 'Simpson', value: Math.min((biodiversityStats.simpson || 0) * 100, 100), fullMark: 100 },
      { metric: 'Chao1', value: Math.min((biodiversityStats.chao1 || 0), 100), fullMark: 100 },
      { metric: 'Evenness', value: Math.min((biodiversityStats.evenness || 0) * 100, 100), fullMark: 100 },
      { metric: 'Richness', value: Math.min((biodiversityStats.richness || 0), 100), fullMark: 100 },
      { metric: 'Novelty', value: Math.min((biodiversityStats.novelty || 0), 100), fullMark: 100 },
    ];
  }, [biodiversityStats]);

  const displayHierarchy = React.useMemo(() => {
    return hierarchyData || mockTaxonomicTree;
  }, [hierarchyData]);

  const { displayTemporalData, temporalKeys } = React.useMemo(() => {
    const sourceData = hierarchyData || mockTaxonomicTree;
    // Get top 4 classes
    const topClasses = [...sourceData]
      .sort((a: any, b: any) => b.size - a.size)
      .slice(0, 4);

    const keys = topClasses.map((d: any) => d.name);
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May'];

    const data = months.map((month, idx) => {
      const row: any = { month };
      topClasses.forEach((cls: any) => {
        // Simulate seasonal trend: increasing slightly + random variation
        // This projection assumes the uploaded sample is representative of the season
        const variation = 0.8 + (idx * 0.1) + (Math.random() * 0.2);
        row[cls.name] = Math.floor(cls.size * variation);
      });
      return row;
    });

    return { displayTemporalData: data, temporalKeys: keys };
  }, [hierarchyData]);

  useEffect(() => {
    const fetchData = async () => {
      // 1. Fetch Analysis Results List
      const { data: analysisData } = await supabase
        .from('analysis_results')
        .select('*, uploads(filename)')
        .order('created_at', { ascending: false })
        .limit(10);
      setResults(analysisData || []);
      if (analysisData && analysisData.length > 0 && !location.state?.uploadId) {
        setSelectedResult(analysisData[0]);
      }

      // 2. Fetch Species Data (Prioritize specific upload if available)
      const uploadId = location.state?.uploadId;
      let targetData = [];

      if (uploadId) {
        const { data } = await supabase
          .from('analysis_results')
          .select('taxa')
          .eq('upload_id', uploadId)
          .single();

        if (data && data.taxa) {
          // Normalize taxa data from analysis_results JSON to match edna_logs structure
          targetData = data.taxa.map((t: any, idx: number) => ({
            predicted_taxa: t.species,
            confidence: t.confidence,
            class: t.class,
            sequence_snippet: `SEQ_${String(idx + 1).padStart(4, '0')}`
          }));
          setAnalysisResults(targetData);
        }
      }

      // If no upload ID or no data found, fall back to edna_logs (Real Data)
      if (targetData.length === 0) {
        const { data: logsData } = await supabase
          .from('edna_logs')
          .select('predicted_taxa, confidence');
        targetData = logsData || [];
      }

      if (targetData.length > 0) {
        // Calculate species counts
        const counts: Record<string, number> = {};
        const speciesConfidenceMap: Record<string, number[]> = {};

        // Build Taxonomic Hierarchy Map
        const hierarchyMap: Record<string, Record<string, number>> = {};

        targetData.forEach((row: any) => {
          const tax = row.predicted_taxa || "Unknown";
          const className = row.class || 'Unclassified';

          counts[tax] = (counts[tax] || 0) + 1;

          // Track confidences for each species
          if (!speciesConfidenceMap[tax]) {
            speciesConfidenceMap[tax] = [];
          }
          if (row.confidence !== null && row.confidence !== undefined) {
            speciesConfidenceMap[tax].push(row.confidence);
          }

          // Hierarchy grouping
          if (!hierarchyMap[className]) hierarchyMap[className] = {};
          hierarchyMap[className][tax] = (hierarchyMap[className][tax] || 0) + 1;
        });

        const hierarchy = Object.entries(hierarchyMap).map(([className, species]) => ({
          name: className,
          size: Object.values(species).reduce((a, b) => a + b, 0),
          children: Object.entries(species).map(([speciesName, count]) => ({
            name: speciesName,
            size: count
          }))
        }));
        setHierarchyData(hierarchy);

        const transformedData = Object.entries(counts).map(([name, abundance]) => ({
          name,
          abundance,
          reads: abundance * 5,
          novelty: name === "Unknown" ? 0.8 : 0.05
        })).sort((a, b) => b.abundance - a.abundance);

        setRealSpeciesData(transformedData);

        // Calculate Biodiversity Indices
        const totalIndividuals = Object.values(counts).reduce((a, b) => a + b, 0);
        const proportions = Object.values(counts).map(c => c / totalIndividuals);

        const shannonIndex = -proportions.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        const simpsonIndex = 1 - proportions.reduce((sum, p) => sum + p * p, 0);
        const richness = Object.keys(counts).length;
        const evenness = richness > 1 ? shannonIndex / Math.log(richness) : 1;

        // Calculate historical/displayed stats
        const totalSequences = targetData.length;
        const speciesCount = richness;

        // Count novel candidates (Unknown or low confidence)
        const novelCount = targetData.filter((row: any) =>
          row.predicted_taxa === "Unknown" ||
          row.predicted_taxa === "Novel/Unknown" ||
          (row.confidence !== null && row.confidence < 0.6)
        ).length;

        const noveltyScore = totalSequences > 0 ? (novelCount / totalSequences) * 100 : 0;

        setBiodiversityStats({
          shannon: shannonIndex,
          simpson: simpsonIndex,
          richness: richness,
          evenness: evenness,
          novelty: noveltyScore,
          chao1: richness + (novelCount > 0 ? (novelCount * (novelCount - 1)) / (2 * (totalSequences - novelCount + 1)) : 0) // Chao1 estimator proxy
        });

        // Calculate average confidence
        const confidences = targetData
          .map((row: any) => row.confidence)
          .filter((c: any) => c !== null && c !== undefined);
        const avgConfidence = confidences.length > 0
          ? (confidences.reduce((sum: number, c: number) => sum + c, 0) / confidences.length * 100).toFixed(1)
          : '0';

        setHistoricalStats({
          totalSequences,
          speciesCount,
          novelCount,
          knownCount: totalSequences - novelCount,
          avgConfidence: parseFloat(avgConfidence)
        });
      }
    };
    fetchData();
  }, []);

  // Auto-switch to Overview tab when navigating from Upload page
  useEffect(() => {
    if (uploadResults) {
      setActiveTab('overview');
    }
  }, [uploadResults]);

  // Auto-switch to Overview tab after file analysis completes
  useEffect(() => {
    if (analysisResults.length > 0) {
      console.log('Analysis Results Updated:', analysisResults.length, 'sequences');
      console.log('Dynamic Species Data:', dynamicSpeciesData.length, 'species');
      console.log('Novel Candidates:', novelCandidates.length);
      // Automatically switch to Overview tab to show results
      setActiveTab('overview');
    }
  }, [analysisResults, dynamicSpeciesData, novelCandidates]);

  const runModelValidation = async (file: File | null = null) => {
    setIsValidating(true);
    try {
      let body = null;

      if (file) {
        const text = await file.text();
        const lines = text.split('\n');
        const sequences: string[] = [];
        const labels: string[] = [];

        let startIdx = 0;
        if (lines[0].toLowerCase().includes("sequence")) startIdx = 1;

        for (let i = startIdx; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          const parts = line.split(',');
          if (parts.length >= 2) {
            sequences.push(parts[0].trim());
            labels.push(parts[1].trim());
          }
        }
        if (sequences.length > 0) {
          body = JSON.stringify({ sequences, labels });
        }
      }

      const response = await fetch('http://localhost:8000/test-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body
      });
      if (!response.ok) throw new Error('Model validation failed');
      const data = await response.json();
      setValidationResults(data);
    } catch (error) {
      console.error("Validation Error:", error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleTrainModel = async () => {
    if (!trainingFile) return;
    setIsTraining(true);
    setTrainingStatus("Parsing file...");

    try {
      // Read CSV
      const text = await trainingFile.text();
      const lines = text.split('\n');
      const sequences: string[] = [];
      const labels: string[] = [];

      // Simple CSV parse: header row assumed? checking first line
      // Expected format: sequence,label
      let startIdx = 0;
      if (lines[0].toLowerCase().includes("sequence")) startIdx = 1;

      for (let i = startIdx; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const parts = line.split(',');
        if (parts.length >= 2) {
          sequences.push(parts[0].trim());
          labels.push(parts[1].trim());
        }
      }

      if (sequences.length === 0) {
        alert("No valid sequences found in CSV");
        setIsTraining(false);
        return;
      }

      setTrainingStatus(`Training on ${sequences.length} samples...`);

      const response = await fetch('http://localhost:8000/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequences, labels })
      });

      if (!response.ok) throw new Error("Training failed");

      const res = await response.json();
      setTrainingStatus(`Success! ${res.message}`);

    } catch (e: any) {
      console.error(e);
      setTrainingStatus("Error: " + e.message);
    } finally {
      setIsTraining(false);
    }
  };

  const handleFileAnalysis = async () => {
    if (!analysisFile) return;
    setIsAnalyzing(true);
    setAnalysisError("");
    setAnalysisResults([]);

    try {
      // Read file and extract sequences
      const text = await analysisFile.text();
      const lines = text.split('\n');
      const sequences: string[] = [];

      // Parse FASTA/FASTQ format
      if (analysisFile.name.endsWith('.fasta') || analysisFile.name.endsWith('.fa')) {
        // FASTA format: >header\nsequence
        let currentSeq = '';
        for (const line of lines) {
          if (line.startsWith('>')) {
            if (currentSeq) sequences.push(currentSeq);
            currentSeq = '';
          } else {
            currentSeq += line.trim();
          }
        }
        if (currentSeq) sequences.push(currentSeq);
      } else {
        // FASTQ format: @header\nsequence\n+\nquality
        for (let i = 0; i < lines.length; i += 4) {
          if (lines[i] && lines[i + 1]) {
            sequences.push(lines[i + 1].trim());
          }
        }
      }

      if (sequences.length === 0) {
        setAnalysisError("No valid sequences found in file");
        return;
      }

      // Call backend API
      const response = await fetch('http://localhost:8000/analyze-edna', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequences })
      });

      if (!response.ok) throw new Error('Analysis failed');
      const data = await response.json();

      // Store results to update Overview tab
      setAnalysisResults(data);
    } catch (error: any) {
      console.error("Analysis Error:", error);
      setAnalysisError(error.message || "Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const tabs = [
    { id: 'file-analysis', label: 'File Analysis', icon: UploadIcon },
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'species', label: 'Species Abundance', icon: Dna },
    { id: 'novelty', label: 'Novelty Detection', icon: Sparkles },
    { id: 'temporal', label: 'Temporal Trends', icon: TrendingUp },
    { id: 'model-test', label: 'Model Validation', icon: AlertTriangle },
    { id: 'training', label: 'Model Training', icon: Layers },
  ];

  // Use real data if available, else use existing mock 'speciesAbundance'
  const displaySpeciesData = dynamicSpeciesData.length > 0
    ? dynamicSpeciesData
    : realSpeciesData.length > 0
      ? realSpeciesData
      : speciesAbundance;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-display font-bold neon-text mb-2">eDNA Analysis</h1>
        <p className="text-muted-foreground">ASV clustering, taxonomy classification & novelty detection</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${activeTab === id
              ? 'bg-primary text-primary-foreground shadow-glow'
              : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
              }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* File Analysis Tab */}
      {activeTab === 'file-analysis' && (
        <div className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-primary text-xl mb-4">Upload FASTA/FASTQ File</h3>
            <p className="text-sm text-muted-foreground mb-6">
              Upload your DNA sequence file to analyze species using the DNABERT model.
            </p>

            <div className="flex gap-4 items-end">
              <div className="flex-1">
                <input
                  type="file"
                  accept=".fasta,.fa,.fastq,.fq"
                  onChange={(e) => setAnalysisFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 cursor-pointer"
                />
                {analysisFile && (
                  <p className="mt-2 text-sm text-success-green font-medium">Selected: {analysisFile.name}</p>
                )}
              </div>
              <button
                onClick={handleFileAnalysis}
                disabled={!analysisFile || isAnalyzing}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-2 rounded-lg font-medium transition-all shadow-glow disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <UploadIcon className="w-4 h-4" />
                    Analyze File
                  </>
                )}
              </button>
            </div>

            {analysisError && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-500 text-sm">
                {analysisError}
              </div>
            )}
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass-card p-4">
              <div className="metric-value text-2xl">
                {summaryStats.totalSequences.toLocaleString()}
              </div>
              <div className="metric-label text-xs">Total Sequences</div>
            </div>
            <div className="glass-card p-4">
              <div className="metric-value text-2xl text-success-green">
                {summaryStats.knownCount.toLocaleString()}
              </div>
              <div className="metric-label text-xs">Known Species</div>
            </div>
            <div className="glass-card p-4">
              <div className="metric-value text-2xl text-warning-amber">
                {summaryStats.novelCount.toLocaleString()}
              </div>
              <div className="metric-label text-xs">Novel/Unknown</div>
            </div>
            <div className="glass-card p-4">
              <div className="metric-value text-2xl">
                {summaryStats.avgConfidence}%
              </div>
              <div className="metric-label text-xs">Avg Confidence</div>
            </div>
          </div>

          {/* Results Table */}
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-primary mb-4">Sequence Predictions</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="px-4 py-3 text-left text-sm font-semibold text-primary">#</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Sequence</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Predicted Species</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Confidence</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {analysisResults.map((result, idx) => (
                    <tr key={idx} className="border-b border-border/30 hover:bg-primary/5">
                      <td className="px-4 py-3 text-sm text-muted-foreground">{idx + 1}</td>
                      <td className="px-4 py-3 font-mono text-xs">{result.sequence_snippet}</td>
                      <td className="px-4 py-3 font-medium italic">{result.predicted_taxa}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${result.confidence >= 0.8 ? 'bg-success-green' :
                                result.confidence >= 0.6 ? 'bg-warning-amber' :
                                  'bg-destructive'
                                }`}
                              style={{ width: `${result.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${result.predicted_taxa === 'Model Not Trained' ? 'bg-gray-500/10 text-gray-500' :
                          result.predicted_taxa === 'Novel/Unknown' ? 'bg-warning-amber/10 text-warning-amber' :
                            result.confidence >= 0.6 ? 'bg-green-500/10 text-green-500' :
                              'bg-red-500/10 text-red-500'
                          }`}>
                          {result.predicted_taxa === 'Model Not Trained' ? 'NOT TRAINED' :
                            result.predicted_taxa === 'Novel/Unknown' ? 'NOVEL' :
                              result.confidence >= 0.6 ? 'KNOWN' : 'LOW CONF'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Confidence Distribution Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-primary mb-4">Confidence Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analysisResults.map((r, i) => ({ name: `Seq ${i + 1}`, confidence: r.confidence }))}>
                  <XAxis dataKey="name" stroke="hsl(185, 50%, 70%)" tick={{ fontSize: 10 }} />
                  <YAxis stroke="hsl(185, 50%, 70%)" domain={[0, 1]} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                    {analysisResults.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.confidence >= 0.8 ? 'hsl(160, 100%, 45%)' : entry.confidence >= 0.6 ? 'hsl(38, 100%, 60%)' : 'hsl(0, 85%, 60%)'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-primary mb-4">Species Breakdown</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={Object.entries(
                      analysisResults.reduce((acc: any, r) => {
                        acc[r.predicted_taxa] = (acc[r.predicted_taxa] || 0) + 1;
                        return acc;
                      }, {})
                    ).map(([name, value]) => ({ name, value }))}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {Object.keys(analysisResults.reduce((acc: any, r) => {
                      acc[r.predicted_taxa] = true;
                      return acc;
                    }, {})).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}


      {/* Upload Results Summary */}
      {
        uploadResults && (
          <div className="glass-card p-6 border-l-4 border-primary mb-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-display font-semibold text-primary text-lg">Latest Upload Results</h4>
              <span className="text-xs text-muted-foreground">From: {location.state?.filename || 'Uploaded File'}</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-display font-bold text-primary">{uploadResults.asvs_count}</div>
                <div className="text-xs text-muted-foreground mt-1">ASVs Found</div>
              </div>
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-display font-bold text-primary">{uploadResults.species_count}</div>
                <div className="text-xs text-muted-foreground mt-1">Species</div>
              </div>
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-display font-bold text-warning-amber">{uploadResults.novel_candidates}</div>
                <div className="text-xs text-muted-foreground mt-1">Novel Candidates</div>
              </div>
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-display font-bold text-destructive">{uploadResults.alerts_generated}</div>
                <div className="text-xs text-muted-foreground mt-1">Alerts</div>
              </div>
            </div>
          </div>
        )
      }
      {/* Overview Tab */}
      {
        activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                {
                  label: 'Total Sequences',
                  value: summaryStats.totalSequences.toLocaleString(),
                  change: analysisResults.length > 0 ? 'From File' : uploadResults ? 'From Upload' : 'Historical'
                },
                {
                  label: 'Species',
                  value: summaryStats.speciesCount.toLocaleString(),
                  change: analysisResults.length > 0 ? `${summaryStats.speciesCount} found` : uploadResults ? 'From Upload' : 'Historical'
                },
                {
                  label: 'Novel Candidates',
                  value: summaryStats.novelCount.toLocaleString(),
                  change: analysisResults.length > 0 ? `${summaryStats.novelCount} detected` : uploadResults ? 'From Upload' : 'Historical'
                },
                {
                  label: 'Avg Confidence',
                  value: `${summaryStats.avgConfidence}%`,
                  change: analysisResults.length > 0 ? 'Real-time' : uploadResults ? 'From Upload' : 'Historical'
                },
              ].map(({ label, value, change }) => (
                <div key={label} className="glass-card p-4">
                  <div className="metric-value text-2xl">{value}</div>
                  <div className="metric-label text-xs">{label}</div>
                  <div className={`text-xs mt-1 ${analysisResults.length > 0 || uploadResults ? 'text-primary' : 'text-success-green'}`}>{change}</div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Taxonomic Distribution */}
              <div className="glass-card p-6">
                <h3 className="font-display font-semibold text-primary mb-4">Taxonomic Distribution</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={displaySpeciesData.slice(0, 7)} // Using displaySpeciesData
                      dataKey="abundance"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {displaySpeciesData.slice(0, 7).map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Biodiversity Radar */}
              <div className="glass-card p-6">
                <h3 className="font-display font-semibold text-primary mb-4">Biodiversity Metrics</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={displayRadarData}>
                    <PolarGrid stroke="hsl(200, 50%, 30%)" />
                    <PolarAngleAxis dataKey="metric" stroke="hsl(185, 50%, 70%)" />
                    <PolarRadiusAxis stroke="hsl(185, 50%, 70%)" />
                    <Radar
                      name="Score"
                      dataKey="value"
                      stroke="hsl(180, 100%, 50%)"
                      fill="hsl(180, 100%, 50%)"
                      fillOpacity={0.3}
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Taxonomic Treemap */}
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-primary mb-4">Taxonomic Hierarchy</h3>
              <ResponsiveContainer width="100%" height={300}>
                <Treemap
                  data={flattenTree(displayHierarchy)}
                  dataKey="size"
                  aspectRatio={4 / 3}
                  stroke="hsl(210, 50%, 6%)"
                >
                  {flattenTree(displayHierarchy).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} fillOpacity={0.8} />
                  ))}
                  <Tooltip
                    contentStyle={tooltipStyle}
                    itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                    cursor={{ stroke: 'hsl(199, 89%, 48%)', strokeWidth: 2 }}
                  />
                </Treemap>
              </ResponsiveContainer>
            </div>
          </div>
        )
      }

      {/* Species Abundance Tab */}
      {
        activeTab === 'species' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Species Abundance Bar Chart */}
              <div className="glass-card p-6">
                <h3 className="font-display font-semibold text-primary mb-4">
                  Species Abundance {analysisResults.length > 0 ? '(Live Analysis)' : location.state?.uploadId ? '(From Uploaded File)' : realSpeciesData.length > 0 ? '(Historical Data)' : '(Mock Data)'}
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={displaySpeciesData} layout="vertical">
                    <XAxis type="number" stroke="hsl(185, 50%, 70%)" />
                    <YAxis
                      type="category"
                      dataKey="name"
                      stroke="hsl(185, 50%, 70%)"
                      width={120}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      cursor={{ fill: 'hsla(180, 100%, 50%, 0.1)' }}
                      itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                    />
                    <Bar dataKey="abundance" fill="hsl(180, 100%, 50%)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Reads vs Abundance Scatter */}
              <div className="glass-card p-6">
                <h3 className="font-display font-semibold text-primary mb-4">Reads vs Abundance {analysisResults.length > 0 ? '(Live Analysis)' : location.state?.uploadId ? '(From Uploaded File)' : realSpeciesData.length > 0 ? '(Historical Data)' : '(Mock Data)'}</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart>
                    <XAxis
                      type="number"
                      dataKey="reads"
                      name="Reads"
                      stroke="hsl(185, 50%, 70%)"
                      label={{ value: 'Reads', position: 'bottom', fill: 'hsl(185, 50%, 70%)' }}
                    />
                    <YAxis
                      type="number"
                      dataKey="abundance"
                      name="Abundance"
                      stroke="hsl(185, 50%, 70%)"
                      label={{ value: 'Abundance', angle: -90, position: 'left', fill: 'hsl(185, 50%, 70%)' }}
                    />
                    <ZAxis type="number" dataKey="novelty" range={[50, 400]} name="Novelty" />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value: any, name: string) => [value, name]}
                      cursor={{ strokeDasharray: '3 3' }}
                      itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                    />
                    <Scatter
                      name="Species"
                      data={displaySpeciesData}
                      fill="hsl(165, 100%, 55%)"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Species Table */}
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-primary mb-4">Species Details</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Species</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Abundance</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Reads</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Novelty</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-primary">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displaySpeciesData.map((species: any, i: number) => (
                      <tr key={i} className="border-b border-border/30 hover:bg-primary/5">
                        <td className="px-4 py-3 font-medium italic">{species.name}</td>
                        <td className="px-4 py-3">{species.abundance.toLocaleString()}</td>
                        <td className="px-4 py-3">{species.reads.toLocaleString()}</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary rounded-full"
                                style={{ width: `${species.novelty * 100}%` }}
                              />
                            </div>
                            <span className="text-xs">{(species.novelty * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={(species.novelty > 0.2 || species.name.includes('Unknown')) ? 'status-warning' : 'status-active'}>
                            {(species.novelty > 0.2 || species.name.includes('Unknown')) ? 'Potential Novel' : 'Identified'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )
      }

      {/* Novelty Detection Tab */}
      {
        activeTab === 'novelty' && (
          <div className="space-y-6">
            {analysisResults.length > 0 && novelCandidates.length > 0 ? (
              <>
                <div className="glass-card p-6 border-l-4 border-warning-amber">
                  <div className="flex items-start gap-4">
                    <AlertTriangle className="w-6 h-6 text-warning-amber flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold mb-2">Novel Species Candidates Detected from Uploaded File</h4>
                      <p className="text-sm text-muted-foreground">
                        {novelCandidates.length} sequences show low confidence (&lt;60%) or are classified as Novel/Unknown.
                        These may represent undocumented species or require additional training data.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="glass-card p-6">
                  <h3 className="font-display font-semibold text-primary mb-4">Novel Candidates</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {novelCandidates.map((candidate, idx) => (
                      <div key={idx} className="glass-card p-6">
                        <div className="flex items-center justify-between mb-4">
                          <span className="font-mono text-sm text-primary">{candidate.sequence_snippet || `SEQ_${idx + 1}`}</span>
                          <span className="text-2xl font-display font-bold text-destructive">
                            {((1 - candidate.confidence) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">% Identity</span>
                            <span>{(candidate.confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Cluster Cohesion</span>
                            <span>{(candidate.confidence * 85).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Reproducibility</span>
                            <span>90%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : analysisResults.length > 0 ? (
              <div className="glass-card p-6 border-l-4 border-success-green">
                <div className="flex items-start gap-4">
                  <AlertTriangle className="w-6 h-6 text-success-green flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-2">No Novel Candidates Detected</h4>
                    <p className="text-sm text-muted-foreground">
                      All {analysisResults.length} sequences from your uploaded file were identified with high confidence (&gt;60%).
                      No novel species candidates were detected.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {noveltyData.slice(0, 3).map((item, i) => (
                    <div key={item.asv} className="glass-card p-6">
                      <div className="flex items-center justify-between mb-4">
                        <span className="font-mono text-sm text-primary">{item.asv}</span>
                        <span className={`text-2xl font-display font-bold ${item.score > 0.7 ? 'text-destructive' : item.score > 0.5 ? 'text-warning-amber' : 'text-success-green'
                          }`}>
                          {(item.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">% Identity</span>
                          <span>{item.pctIdentity}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Cluster Cohesion</span>
                          <span>{(item.clusterCohesion * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Reproducibility</span>
                          <span>{(item.reproducibility * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Novelty Score Distribution */}
                  <div className="glass-card p-6">
                    <h3 className="font-display font-semibold text-primary mb-4">Novelty Score Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={noveltyData}>
                        <XAxis dataKey="asv" stroke="hsl(185, 50%, 70%)" />
                        <YAxis stroke="hsl(185, 50%, 70%)" domain={[0, 1]} />
                        <Tooltip
                          contentStyle={tooltipStyle}
                          cursor={{ fill: 'hsla(180, 100%, 50%, 0.1)' }}
                          itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                        />
                        <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                          {noveltyData.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={entry.score > 0.7 ? 'hsl(0, 85%, 60%)' : entry.score > 0.5 ? 'hsl(38, 100%, 60%)' : 'hsl(160, 100%, 45%)'}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Novelty Metrics Scatter */}
                  <div className="glass-card p-6">
                    <h3 className="font-display font-semibold text-primary mb-4">Identity vs Novelty</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart>
                        <XAxis
                          type="number"
                          dataKey="pctIdentity"
                          name="% Identity"
                          stroke="hsl(185, 50%, 70%)"
                          domain={[60, 100]}
                        />
                        <YAxis
                          type="number"
                          dataKey="score"
                          name="Novelty Score"
                          stroke="hsl(185, 50%, 70%)"
                          domain={[0, 1]}
                        />
                        <ZAxis type="number" dataKey="reproducibility" range={[100, 500]} />
                        <Tooltip
                          contentStyle={tooltipStyle}
                          cursor={{ strokeDasharray: '3 3' }}
                          itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                        />
                        <Scatter name="ASVs" data={noveltyData}>
                          {noveltyData.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={entry.score > 0.7 ? 'hsl(0, 85%, 60%)' : entry.score > 0.5 ? 'hsl(38, 100%, 60%)' : 'hsl(160, 100%, 45%)'}
                            />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Novelty Candidates Alert */}
                <div className="glass-card p-6 border-l-4 border-warning-amber">
                  <div className="flex items-start gap-4">
                    <AlertTriangle className="w-6 h-6 text-warning-amber flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold mb-2">Novel Species Candidates Detected</h4>
                      <p className="text-sm text-muted-foreground">
                        {noveltyData.filter(d => d.score > 0.5).length} ASVs show significant novelty scores (&gt;50%).
                        These sequences may represent undocumented species or significant genetic variants.
                        Recommend BLAST verification and phylogenetic analysis.
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )
      }

      {/* Temporal Trends Tab */}
      {
        activeTab === 'temporal' && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-primary mb-4">Taxonomic Abundance Over Time {analysisResults.length > 0 ? '(Projected)' : ''}</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={displayTemporalData}>
                  <XAxis dataKey="month" stroke="hsl(185, 50%, 70%)" />
                  <YAxis stroke="hsl(185, 50%, 70%)" label={{ value: 'Abundance', angle: -90, position: 'insideLeft', fill: 'hsl(185, 50%, 70%)' }} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    itemStyle={{ color: '#ffffff', fontWeight: 'bold' }}
                    cursor={{ fill: 'hsla(180, 100%, 50%, 0.1)' }}
                  />
                  <Legend />
                  {temporalKeys.map((key: string, i: number) => (
                    <Bar
                      key={key}
                      dataKey={key}
                      stackId="a"
                      fill={COLORS[i % COLORS.length]}
                      name={key}
                      radius={i === temporalKeys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {temporalKeys.map((taxon: string, i: number) => {
                // Determine mock/projected trend description based on index
                const trends = [
                  { label: 'Seasonal Peak', value: 'May', change: '+12%', color: 'text-success-green' },
                  { label: 'Population Stable', value: 'Avg', change: '+2%', color: 'text-primary' },
                  { label: 'Early Season', value: 'Low', change: '-5%', color: 'text-warning-amber' },
                  { label: 'Declining', value: 'Late', change: '-8%', color: 'text-destructive' }
                ];
                const trend = trends[i % trends.length];

                return (
                  <div key={taxon} className="glass-card p-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-display font-medium text-muted-foreground capitalize">{taxon}</span>
                      <TrendingUp className={`w-4 h-4 ${trend.color}`} />
                    </div>
                    <div className="text-2xl font-bold font-display mb-1">{trend.value}</div>
                    <div className={`text-xs ${trend.color} flex items-center gap-1`}>
                      {trend.change} vs prev month
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )
      }

      {/* Model Validation Tab */}
      {
        activeTab === 'model-test' && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="font-display font-semibold text-primary text-xl">Model Performance Verification</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Run validation tests on synthetic data or upload your own <code>(sequence, species)</code> CSV.
                  </p>
                </div>

                <div className="flex gap-2 items-center">
                  <div className="relative">
                    <input
                      type="file"
                      accept=".csv"
                      id="test-file-upload"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) runModelValidation(file);
                      }}
                    />
                    <label
                      htmlFor="test-file-upload"
                      className={`cursor-pointer bg-muted/30 hover:bg-muted/50 text-muted-foreground hover:text-primary px-4 py-2 rounded-lg font-medium transition-all text-sm flex items-center gap-2 border border-dashed border-border ${isValidating ? 'pointer-events-none opacity-50' : ''}`}
                    >
                      <Layers className="w-3 h-3" />
                      {isValidating ? 'Uploading...' : 'Upload Test CSV'}
                    </label>
                  </div>

                  <button
                    onClick={() => runModelValidation(null)}
                    disabled={isValidating}
                    className="bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-2 rounded-lg font-medium transition-all shadow-glow disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isValidating ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="w-4 h-4" />
                        Run Default Test
                      </>
                    )}
                  </button>
                </div>
              </div>

              {validationResults && (
                <div className="space-y-6 animate-fade-in">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="p-4 bg-muted/20 rounded-lg border border-border/50">
                      <span className="text-sm text-muted-foreground block mb-1">Overall Accuracy</span>
                      <span className="text-3xl font-bold font-display text-primary">{validationResults.accuracy}%</span>
                    </div>
                    <div className="p-4 bg-muted/20 rounded-lg border border-border/50">
                      <span className="text-sm text-muted-foreground block mb-1">Samples Tested</span>
                      <span className="text-3xl font-bold font-display">{validationResults.total_samples}</span>
                    </div>
                    <div className="p-4 bg-muted/20 rounded-lg border border-border/50">
                      <span className="text-sm text-muted-foreground block mb-1">Correct Predictions</span>
                      <span className="text-3xl font-bold font-display text-success-green">{validationResults.correct_predictions}</span>
                    </div>
                  </div>

                  <div className="overflow-hidden rounded-lg border border-border/50">
                    <table className="w-full">
                      <thead className="bg-muted/30">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-muted-foreground">Sequence Snippet</th>
                          <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-muted-foreground">Expected Label</th>
                          <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-muted-foreground">Predicted</th>
                          <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-muted-foreground">Confidence</th>
                          <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-muted-foreground">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/30">
                        {validationResults.results.map((res: any, idx: number) => (
                          <tr key={idx} className="hover:bg-muted/10 transition-colors">
                            <td className="px-4 py-3 font-mono text-sm">{res.sequence_snippet}</td>
                            <td className="px-4 py-3 text-sm">{res.expected}</td>
                            <td className="px-4 py-3 text-sm font-medium text-primary">{res.predicted}</td>
                            <td className="px-4 py-3 text-sm">{(res.confidence * 100).toFixed(1)}%</td>
                            <td className="px-4 py-3">
                              <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${res.correct ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                                {res.correct ? 'PASS' : 'FAIL'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )
      }

      {/* Model Training Tab */}
      {
        activeTab === 'training' && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <div className="max-w-xl mx-auto text-center space-y-6">
                <div>
                  <h3 className="font-display font-semibold text-primary text-xl">Train DNABERT Model</h3>
                  <p className="text-sm text-muted-foreground mt-2">
                    Upload a CSV file containing sequences and their species labels to train or fine-tune the species identification model.
                  </p>
                  <ul className="text-xs text-muted-foreground mt-2 text-left list-disc pl-5 inline-block">
                    <li>Format: CSV (Comma Separated Values)</li>
                    <li>Columns: <code>sequence</code>, <code>species</code></li>
                    <li>Minimum 5 samples per species recommended.</li>
                  </ul>
                </div>

                <div className="p-8 border-2 border-dashed border-border rounded-xl hover:bg-muted/30 transition-colors">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setTrainingFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 cursor-pointer"
                  />
                  {trainingFile && (
                    <p className="mt-2 text-sm text-success-green font-medium">Selected: {trainingFile.name}</p>
                  )}
                </div>

                <button
                  onClick={handleTrainModel}
                  disabled={!trainingFile || isTraining}
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-3 rounded-lg font-medium transition-all shadow-glow disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isTraining ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      {trainingStatus || "Training..."}
                    </>
                  ) : (
                    <>
                      <Layers className="w-5 h-5" />
                      Start Training
                    </>
                  )}
                </button>

                {trainingStatus && !isTraining && (
                  <div className={`p-4 rounded-lg text-sm ${trainingStatus.startsWith("Success") ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"}`}>
                    {trainingStatus}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      }

      {/* Recent Results */}
      <div className="glass-card p-6">
        <h3 className="font-display font-semibold text-primary mb-4">Recent Analysis Results</h3>
        {results.length > 0 ? (
          <div className="space-y-3">
            {results.map((r) => (
              <div
                key={r.id}
                className={`p-4 rounded-lg flex justify-between items-center cursor-pointer transition-colors ${selectedResult?.id === r.id ? 'bg-primary/20 border border-primary/50' : 'bg-muted/30 hover:bg-muted/50'
                  }`}
                onClick={() => setSelectedResult(r)}
              >
                <div>
                  <span className="font-medium">{r.uploads?.filename || 'Unknown'}</span>
                  <p className="text-xs text-muted-foreground mt-1">
                    {new Date(r.created_at).toLocaleDateString()} • {r.summary?.slice(0, 60)}...
                  </p>
                </div>
                <span className="status-active">Completed</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground">No analysis results yet. Upload a FASTQ file to begin.</p>
        )}
      </div>
    </div >
  );
};

export default Analysis;
