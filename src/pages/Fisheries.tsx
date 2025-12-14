import { useEffect, useState } from "react";
import { Fish, Anchor, Scale } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

const Fisheries = () => {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    const fetch = async () => {
      const { data } = await supabase.from('fisheries').select('*').order('catch_date', { ascending: false });
      setData(data || []);
    };
    fetch();
  }, []);

  const getRatingClass = (r: string) => {
    if (r === 'sustainable') return 'status-active';
    if (r === 'moderate') return 'status-warning';
    return 'status-critical';
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold neon-text mb-2">Fisheries Data</h1>
        <p className="text-muted-foreground">Sustainable fishing monitoring and catch records</p>
      </div>

      <div className="glass-card overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/50">
              <th className="px-6 py-4 text-left text-sm font-semibold text-primary">Vessel</th>
              <th className="px-6 py-4 text-left text-sm font-semibold text-primary">Species Caught</th>
              <th className="px-6 py-4 text-left text-sm font-semibold text-primary">Quantity (kg)</th>
              <th className="px-6 py-4 text-left text-sm font-semibold text-primary">Date</th>
              <th className="px-6 py-4 text-left text-sm font-semibold text-primary">Rating</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.id} className="border-b border-border/30 hover:bg-primary/5">
                <td className="px-6 py-4 flex items-center gap-2">
                  <Anchor className="w-4 h-4 text-primary" />
                  {row.vessel_name}
                </td>
                <td className="px-6 py-4">{(row.species_caught || []).join(', ')}</td>
                <td className="px-6 py-4">{row.quantity_kg?.toLocaleString()}</td>
                <td className="px-6 py-4">{row.catch_date ? new Date(row.catch_date).toLocaleDateString() : '-'}</td>
                <td className="px-6 py-4"><span className={getRatingClass(row.sustainable_rating)}>{row.sustainable_rating}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Fisheries;
