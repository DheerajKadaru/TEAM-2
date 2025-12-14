-- Create uploads table for FASTQ/FASTA files
CREATE TABLE public.uploads (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  file_type TEXT DEFAULT 'fastq',
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create analysis_results table
CREATE TABLE public.analysis_results (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  upload_id UUID NOT NULL REFERENCES public.uploads(id) ON DELETE CASCADE,
  asvs JSONB DEFAULT '[]',
  taxa JSONB DEFAULT '[]',
  biodiversity_indices JSONB DEFAULT '{}',
  novelty_scores JSONB DEFAULT '[]',
  abundance_data JSONB DEFAULT '{}',
  summary TEXT,
  provenance JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create alerts table
CREATE TABLE public.alerts (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  upload_id UUID REFERENCES public.uploads(id) ON DELETE SET NULL,
  alert_type TEXT NOT NULL CHECK (alert_type IN ('novel_species', 'endangered', 'invasive', 'anomaly', 'threshold')),
  severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  title TEXT NOT NULL,
  description TEXT,
  species_name TEXT,
  location JSONB DEFAULT '{}',
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'validated', 'rejected', 'needs_sampling')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create reports table
CREATE TABLE public.reports (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  upload_id UUID REFERENCES public.uploads(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  report_type TEXT DEFAULT 'analysis' CHECK (report_type IN ('analysis', 'biodiversity', 'conservation', 'summary')),
  file_path TEXT,
  format TEXT DEFAULT 'pdf' CHECK (format IN ('pdf', 'csv', 'json')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create stations table for map markers
CREATE TABLE public.stations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  latitude DOUBLE PRECISION NOT NULL,
  longitude DOUBLE PRECISION NOT NULL,
  station_type TEXT DEFAULT 'monitoring',
  status TEXT DEFAULT 'active',
  metadata JSONB DEFAULT '{}',
  last_sample_date TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create digital_twin_entities for tracking
CREATE TABLE public.digital_twin_entities (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  entity_type TEXT NOT NULL CHECK (entity_type IN ('species', 'vessel', 'buoy', 'sensor')),
  name TEXT NOT NULL,
  current_position JSONB DEFAULT '{}',
  trajectory JSONB DEFAULT '[]',
  status TEXT DEFAULT 'active',
  metadata JSONB DEFAULT '{}',
  last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create fisheries table
CREATE TABLE public.fisheries (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  vessel_name TEXT NOT NULL,
  species_caught JSONB DEFAULT '[]',
  catch_date TIMESTAMP WITH TIME ZONE,
  location JSONB DEFAULT '{}',
  quantity_kg DOUBLE PRECISION,
  sustainable_rating TEXT CHECK (sustainable_rating IN ('sustainable', 'moderate', 'unsustainable', 'unknown')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create chat_messages for SeaSage chatbot
CREATE TABLE public.chat_messages (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on all tables (public access for MVP)
ALTER TABLE public.uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.stations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.digital_twin_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fisheries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Create public read/write policies for MVP (no auth required)
CREATE POLICY "Public access for uploads" ON public.uploads FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for analysis_results" ON public.analysis_results FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for alerts" ON public.alerts FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for reports" ON public.reports FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for stations" ON public.stations FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for digital_twin_entities" ON public.digital_twin_entities FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for fisheries" ON public.fisheries FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access for chat_messages" ON public.chat_messages FOR ALL USING (true) WITH CHECK (true);

-- Create storage bucket for uploads
INSERT INTO storage.buckets (id, name, public) VALUES ('uploads', 'uploads', true);
INSERT INTO storage.buckets (id, name, public) VALUES ('reports', 'reports', true);

-- Storage policies
CREATE POLICY "Public upload access" ON storage.objects FOR ALL USING (bucket_id IN ('uploads', 'reports')) WITH CHECK (bucket_id IN ('uploads', 'reports'));

-- Insert sample stations for the map
INSERT INTO public.stations (name, description, latitude, longitude, station_type, status) VALUES
('Arabian Sea Station Alpha', 'Primary monitoring station for eDNA sampling', 19.0760, 72.8777, 'monitoring', 'active'),
('Bay of Bengal Station Beta', 'Biodiversity hotspot monitoring', 13.0827, 80.2707, 'monitoring', 'active'),
('Lakshadweep Research Point', 'Coral reef ecosystem monitoring', 10.5667, 72.6417, 'research', 'active'),
('Andaman Marine Station', 'Deep sea sampling station', 11.6234, 92.7265, 'sampling', 'active'),
('Gulf of Kutch Observatory', 'Mangrove ecosystem monitoring', 22.4707, 69.0770, 'observatory', 'active');

-- Insert sample digital twin entities
INSERT INTO public.digital_twin_entities (entity_type, name, current_position, status) VALUES
('species', 'Blue Whale Pod Alpha', '{"lat": 18.5, "lng": 73.2}', 'tracking'),
('vessel', 'RV Samudra Ratna', '{"lat": 15.3, "lng": 74.1}', 'active'),
('buoy', 'INCOIS Buoy 23', '{"lat": 12.9, "lng": 80.5}', 'operational'),
('sensor', 'Deep Sea Sensor Array', '{"lat": 10.8, "lng": 72.8}', 'active');

-- Insert sample fisheries data
INSERT INTO public.fisheries (vessel_name, species_caught, catch_date, location, quantity_kg, sustainable_rating) VALUES
('Matsya Kanya', '["Sardine", "Mackerel"]', now() - interval '2 days', '{"lat": 15.5, "lng": 73.8}', 450.5, 'sustainable'),
('Sagara Mitra', '["Tuna", "Skipjack"]', now() - interval '1 day', '{"lat": 13.2, "lng": 80.4}', 820.0, 'moderate'),
('Varuna Deep', '["Pomfret", "Prawns"]', now(), '{"lat": 19.1, "lng": 72.9}', 280.0, 'sustainable');

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_uploads_updated_at BEFORE UPDATE ON public.uploads FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON public.alerts FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();