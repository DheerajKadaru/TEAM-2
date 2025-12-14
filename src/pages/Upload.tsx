import { useState, useCallback } from "react";
import { Upload as UploadIcon, FileText, CheckCircle, Loader2, Play, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { Link } from "react-router-dom";

const Upload = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [processed, setProcessed] = useState(false);
  const [processingSummary, setProcessingSummary] = useState<any>(null);
  const { toast } = useToast();

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.name.endsWith('.fastq') || droppedFile.name.endsWith('.fasta') || droppedFile.name.endsWith('.fq') || droppedFile.name.endsWith('.fa'))) {
      setFile(droppedFile);
    }
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const filePath = `${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage.from('uploads').upload(filePath, file);
      if (uploadError) throw uploadError;

      const { data, error } = await supabase.from('uploads').insert({
        filename: file.name,
        file_path: filePath,
        file_size: file.size,
        file_type: file.name.endsWith('.fastq') || file.name.endsWith('.fq') ? 'fastq' : 'fasta',
        status: 'pending',
      }).select().single();

      if (error) throw error;
      setUploadId(data.id);
      toast({ title: "Upload successful!", description: `File ${file.name} uploaded. Click Process to analyze.` });
    } catch (err) {
      console.error(err);
      toast({ variant: "destructive", title: "Upload failed", description: "Please try again." });
    } finally {
      setUploading(false);
    }
  };

  const handleProcess = async () => {
    if (!uploadId) return;
    setProcessing(true);
    try {
      const { data, error } = await supabase.functions.invoke('process-fastq', {
        body: { upload_id: uploadId },
      });

      if (error) throw error;

      setProcessed(true);
      setProcessingSummary(data.summary);
      toast({
        title: "Analysis Complete!",
        description: `Found ${data.summary.asvs_count} ASVs, ${data.summary.species_count} species. ${data.summary.alerts_generated} alerts generated.`
      });
    } catch (err) {
      console.error(err);
      toast({ variant: "destructive", title: "Processing failed", description: "Please try again." });
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-display font-bold neon-text mb-2">FASTQ/FASTA Upload</h1>
        <p className="text-muted-foreground">Upload your eDNA sequencing data for analysis</p>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`glass-card p-12 border-2 border-dashed transition-all duration-300 relative overflow-hidden ${isDragging ? 'border-primary bg-primary/5 scale-105' : 'border-border'
          }`}
      >
        {/* DNA helix pattern overlay */}
        <div className="absolute inset-0 opacity-5">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 20px, hsl(220, 90%, 50%) 20px, hsl(220, 90%, 50%) 40px)`,
            }}
          />
        </div>

        {/* Flowing DNA effect when dragging */}
        {isDragging && (
          <div className="absolute inset-0 opacity-15">
            <div
              className="absolute inset-0"
              style={{
                background: `linear-gradient(135deg, transparent 0%, hsl(220, 90%, 50%) 50%, transparent 100%)`,
                backgroundSize: '200% 200%',
                animation: 'waterFlow 2s ease-in-out infinite',
              }}
            />
          </div>
        )}

        <div className="relative z-10">
          <div className="text-center">
            {file ? (
              <div className="space-y-4">
                <FileText className="w-16 h-16 mx-auto text-primary" />
                <p className="font-semibold">{file.name}</p>
                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>

                {!uploadId && (
                  <Button onClick={handleUpload} disabled={uploading} className="btn-ocean">
                    {uploading ? <><Loader2 className="w-4 h-4 animate-spin mr-2" /> Uploading...</> : 'Upload File'}
                  </Button>
                )}

                {uploadId && !processed && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-center gap-2 text-primary">
                      <CheckCircle className="w-5 h-5" />
                      <span>Uploaded! ID: {uploadId.slice(0, 8)}...</span>
                    </div>
                    <Button onClick={handleProcess} disabled={processing} className="btn-ocean">
                      {processing ? (
                        <><Loader2 className="w-4 h-4 animate-spin mr-2" /> Processing eDNA...</>
                      ) : (
                        <><Play className="w-4 h-4 mr-2" /> Process & Analyze</>
                      )}
                    </Button>
                  </div>
                )}

                {processed && processingSummary && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center gap-2 text-success-green">
                      <CheckCircle className="w-5 h-5" />
                      <span>Analysis Complete!</span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div className="bg-muted/30 rounded-lg p-3 relative overflow-hidden group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity">
                          <div className="absolute inset-0" style={{ background: `linear-gradient(135deg, transparent, hsl(220, 90%, 50%), transparent)` }} />
                        </div>
                        <div className="text-2xl font-display font-bold text-primary relative z-10">{processingSummary.asvs_count}</div>
                        <div className="text-xs text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">ASVs Found</div>
                      </div>
                      <div className="bg-muted/30 rounded-lg p-3 relative overflow-hidden group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity">
                          <div className="absolute inset-0" style={{ background: `linear-gradient(135deg, transparent, hsl(220, 90%, 50%), transparent)` }} />
                        </div>
                        <div className="text-2xl font-display font-bold text-primary relative z-10">{processingSummary.species_count}</div>
                        <div className="text-xs text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">Species</div>
                      </div>
                      <div className="bg-muted/30 rounded-lg p-3 relative overflow-hidden group hover:bg-warning-amber/10 transition-all duration-300 hover:scale-105">
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity">
                          <div className="absolute inset-0" style={{ background: `linear-gradient(135deg, transparent, hsl(38, 100%, 60%), transparent)` }} />
                        </div>
                        <div className="text-2xl font-display font-bold text-warning-amber relative z-10">{processingSummary.novel_candidates}</div>
                        <div className="text-xs text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">Novel Candidates</div>
                      </div>
                      <div className="bg-muted/30 rounded-lg p-3 relative overflow-hidden group hover:bg-destructive/10 transition-all duration-300 hover:scale-105">
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity">
                          <div className="absolute inset-0" style={{ background: `linear-gradient(135deg, transparent, hsl(0, 85%, 60%), transparent)` }} />
                        </div>
                        <div className="text-2xl font-display font-bold text-destructive relative z-10">{processingSummary.alerts_generated}</div>
                        <div className="text-xs text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">Alerts</div>
                      </div>
                    </div>

                    <div className="flex gap-3 justify-center">
                      <Link
                        to="/analysis"
                        state={{
                          uploadResults: processingSummary,
                          uploadId: uploadId,
                          filename: file.name
                        }}
                        className="btn-ocean inline-flex items-center gap-2"
                      >
                        <BarChart3 className="w-4 h-4" /> View Report
                      </Link>
                      <Button variant="outline" onClick={() => { setFile(null); setUploadId(null); setProcessed(false); setProcessingSummary(null); }}>
                        Upload Another
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <UploadIcon className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <p className="text-lg font-medium mb-2">Drag & drop your FASTQ/FASTA file</p>
                <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
                <input type="file" accept=".fastq,.fasta,.fq,.fa" className="hidden" id="file-input" onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])} />
                <label htmlFor="file-input" className="btn-ocean cursor-pointer inline-block">Select File</label>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="glass-card p-6 relative overflow-hidden">
        {/* DNA structure background */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0 flex items-center justify-center">
            <UploadIcon className="w-32 h-32 text-primary rotate-12" />
          </div>
        </div>

        <div className="relative z-10">
          <h3 className="font-display font-semibold text-primary mb-3 relative">
            Supported Formats
            <span className="absolute -bottom-1 left-0 w-1/3 h-0.5 bg-primary/50 blur-sm" />
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-muted/30 rounded-lg p-3 text-center group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
              <code className="text-primary group-hover:text-accent transition-colors">.fastq</code>
              <p className="text-xs text-muted-foreground mt-1 group-hover:text-foreground/80 transition-colors">FASTQ format</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
              <code className="text-primary group-hover:text-accent transition-colors">.fq</code>
              <p className="text-xs text-muted-foreground mt-1 group-hover:text-foreground/80 transition-colors">FASTQ short</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
              <code className="text-primary group-hover:text-accent transition-colors">.fasta</code>
              <p className="text-xs text-muted-foreground mt-1 group-hover:text-foreground/80 transition-colors">FASTA format</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center group hover:bg-primary/10 transition-all duration-300 hover:scale-105">
              <code className="text-primary group-hover:text-accent transition-colors">.fa</code>
              <p className="text-xs text-muted-foreground mt-1 group-hover:text-foreground/80 transition-colors">FASTA short</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
