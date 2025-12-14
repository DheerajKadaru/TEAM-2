import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from "react-leaflet";
import { Icon, divIcon, point } from "leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import "leaflet/dist/leaflet.css";
import { supabase } from "@/integrations/supabase/client";
import { Radio } from "lucide-react";

interface Station {
  id: string;
  name: string;
  description: string;
  latitude: number;
  longitude: number;
  station_type: string;
  status: string;
}

interface Entity {
  id: string;
  entity_type: string;
  name: string;
  current_position: unknown;
  status: string;
}

const entityIcons: Record<string, string> = {
  species: '🐋',
  vessel: '🚢',
  buoy: '🔴',
  sensor: '📡',
};

const stationColors: Record<string, string> = {
  monitoring: '#00CED1',
  research: '#40E0D0',
  sampling: '#48D1CC',
  observatory: '#00FFFF',
};

const createClusterCustomIcon = (cluster: any) => {
  return divIcon({
    html: `<div class="cluster-icon">${cluster.getChildCount()}</div>`,
    className: 'custom-marker-cluster',
    iconSize: point(40, 40, true),
  });
};

const parsePosition = (pos: unknown): { lat?: number; lng?: number } => {
  if (typeof pos === 'object' && pos !== null) {
    const p = pos as Record<string, unknown>;
    return {
      lat: typeof p.lat === 'number' ? p.lat : undefined,
      lng: typeof p.lng === 'number' ? p.lng : undefined,
    };
  }
  return {};
};

const DigitalTwin = () => {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [stations, setStations] = useState<Station[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const [{ data: e }, { data: s }] = await Promise.all([
        supabase.from('digital_twin_entities').select('*'),
        supabase.from('stations').select('*'),
      ]);
      setEntities(e || []);
      setStations(s || []);
    };
    fetchData();
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold neon-text mb-2">Digital Twin</h1>
        <p className="text-muted-foreground">Real-time marine ecosystem tracking and monitoring</p>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4">
        {Object.entries(stationColors).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="capitalize">{type}</span>
          </div>
        ))}
        <div className="flex items-center gap-2 text-sm">
          <Radio className="w-3 h-3 text-coral animate-pulse" />
          <span>Tracked Entity</span>
        </div>
      </div>

      <div className="glass-card overflow-hidden" style={{ height: '750px' }}>
        <style>{`
          .custom-marker-cluster {
            background: transparent;
          }
          .cluster-icon {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, hsl(180, 100%, 45%), hsl(165, 100%, 55%));
            border-radius: 50%;
            color: hsl(210, 50%, 6%);
            font-weight: bold;
            font-size: 14px;
            box-shadow: 0 0 20px hsla(180, 100%, 50%, 0.5);
            border: 2px solid hsla(180, 100%, 70%, 0.8);
          }
        `}</style>
        <MapContainer center={[15.0, 75.0]} zoom={5} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
            attribution='&copy; <a href="https://www.google.com/maps">Google Maps</a>'
          />
          <MarkerClusterGroup
            chunkedLoading
            iconCreateFunction={createClusterCustomIcon}
          >
            {stations.map((s) => (
              <CircleMarker
                key={s.id}
                center={[s.latitude, s.longitude]}
                radius={10}
                fillColor={stationColors[s.station_type] || '#00CED1'}
                color={stationColors[s.station_type] || '#00CED1'}
                weight={2}
                opacity={0.8}
                fillOpacity={0.5}
              >
                <Popup>
                  <div className="p-2">
                    <strong className="text-primary">{s.name}</strong>
                    <br />
                    <span className="text-sm">{s.description}</span>
                    <br />
                    <span className="text-xs capitalize">{s.station_type}</span>
                    <br />
                    <span className="status-active mt-1 inline-block">{s.status}</span>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MarkerClusterGroup>

          {/* Tracked entities with pulsing effect */}
          {entities.map((e) => {
            const pos = parsePosition(e.current_position);
            if (!pos?.lat || !pos?.lng) return null;
            return (
              <CircleMarker
                key={e.id}
                center={[pos.lat, pos.lng]}
                radius={8}
                fillColor="#FF6B6B"
                color="#FF6B6B"
                weight={3}
                opacity={1}
                fillOpacity={0.7}
              >
                <Popup>
                  <div className="p-2">
                    <span className="text-xl mr-2">{entityIcons[e.entity_type] || '📍'}</span>
                    <strong>{e.name}</strong>
                    <br />
                    <span className="text-xs capitalize">{e.entity_type}</span>
                    <br />
                    <span className="status-active mt-1 inline-block">{e.status}</span>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <h3 className="font-display font-semibold text-primary mb-4">Monitoring Stations ({stations.length})</h3>
          <div className="grid gap-3 max-h-64 overflow-y-auto custom-scrollbar">
            {stations.map((s) => (
              <div key={s.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: stationColors[s.station_type] || '#00CED1' }}
                  />
                  <div>
                    <p className="font-medium text-sm">{s.name}</p>
                    <p className="text-xs text-muted-foreground capitalize">{s.station_type}</p>
                  </div>
                </div>
                <span className="status-active">{s.status}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-card p-6">
          <h3 className="font-display font-semibold text-primary mb-4">Tracked Entities ({entities.length})</h3>
          <div className="grid gap-3 max-h-64 overflow-y-auto custom-scrollbar">
            {entities.map((e) => (
              <div key={e.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{entityIcons[e.entity_type] || '📍'}</span>
                  <div>
                    <p className="font-medium text-sm">{e.name}</p>
                    <p className="text-xs text-muted-foreground capitalize">{e.entity_type}</p>
                  </div>
                </div>
                <span className="status-active">{e.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DigitalTwin;
