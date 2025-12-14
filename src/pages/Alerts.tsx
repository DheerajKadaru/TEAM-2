import { useEffect, useState } from "react";
import { AlertTriangle, Search, Eye, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const Alerts = () => {
  const [alerts, setAlerts] = useState<any[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    const fetch = async () => {
      const { data } = await supabase.from('alerts').select('*').order('created_at', { ascending: false });
      setAlerts(data || []);
    };
    fetch();
  }, []);

  const handleValidate = async (id: string, status: string) => {
    await supabase.from('alerts').update({ status }).eq('id', id);
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status } : a));
    toast({ title: `Alert ${status}`, description: `Alert has been marked as ${status}` });
  };

  const getSeverityClass = (s: string) => {
    if (s === 'critical') return 'status-critical';
    if (s === 'high') return 'status-warning';
    return 'status-active';
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold neon-text mb-2">Conservation Alerts</h1>
        <p className="text-muted-foreground">Novel species detection, endangered & invasive alerts</p>
      </div>

      <div className="space-y-4">
        {alerts.length > 0 ? alerts.map((alert) => (
          <div key={alert.id} className="glass-card p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <AlertTriangle className={`w-6 h-6 ${alert.severity === 'critical' ? 'text-destructive' : 'text-warning-amber'}`} />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="font-semibold">{alert.title}</h3>
                    <span className={getSeverityClass(alert.severity)}>{alert.severity}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-muted">{alert.alert_type?.replace('_', ' ')}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{alert.description}</p>
                  {alert.species_name && <p className="text-sm text-primary mt-1">Species: {alert.species_name}</p>}
                </div>
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" className="gap-1"><Eye className="w-4 h-4" /> Inspect</Button>
                {alert.status === 'active' && (
                  <>
                    <Button size="sm" variant="outline" className="gap-1 text-success-green border-success-green/50" onClick={() => handleValidate(alert.id, 'validated')}><CheckCircle className="w-4 h-4" /> Validate</Button>
                    <Button size="sm" variant="outline" className="gap-1 text-destructive border-destructive/50" onClick={() => handleValidate(alert.id, 'rejected')}><XCircle className="w-4 h-4" /> Reject</Button>
                  </>
                )}
              </div>
            </div>
          </div>
        )) : (
          <div className="glass-card p-12 text-center">
            <AlertTriangle className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No active alerts. Upload eDNA data to detect novel species.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Alerts;
