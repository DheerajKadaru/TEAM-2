import { useEffect, useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { AlertTriangle, Info } from "lucide-react";

export const AlertPopupManager = () => {
    const { toast } = useToast();
    const hasFetched = useRef(false);

    useEffect(() => {
        if (hasFetched.current) return;
        hasFetched.current = true;

        const fetchAlerts = async () => {
            const { data } = await supabase
                .from("alerts")
                .select("id, title, description, severity")
                .eq("status", "active")
                .in("severity", ["high", "critical"])
                .limit(3);

            if (data && data.length > 0) {
                data.forEach((alert, index) => {
                    setTimeout(() => {
                        toast({
                            variant: alert.severity === "critical" ? "destructive" : "default",
                            title: (
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="h-4 w-4" />
                                    <span>{alert.title}</span>
                                </div>
                            ) as any, // escaping type check for specialized title
                            description: alert.description,
                            duration: 10000, // Show for 10 seconds
                        });
                    }, index * 800 + 1000); // Stagger start time
                });
            }
        };

        fetchAlerts();

        // Optional: Realtime subscription could go here
        const channel = supabase
            .channel('schema-db-changes')
            .on(
                'postgres_changes',
                {
                    event: 'INSERT',
                    schema: 'public',
                    table: 'alerts',
                    filter: 'status=eq.active',
                },
                (payload) => {
                    const newAlert = payload.new as any;
                    if (['high', 'critical'].includes(newAlert.severity)) {
                        toast({
                            variant: newAlert.severity === "critical" ? "destructive" : "default",
                            title: (
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="h-4 w-4" />
                                    <span>{newAlert.title}</span>
                                </div>
                            ) as any,
                            description: newAlert.description,
                            duration: 10000,
                        });
                    }
                }
            )
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };

    }, [toast]);

    return null;
};
