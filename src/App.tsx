import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "@/components/Layout";
import Home from "@/pages/Home";
import Upload from "@/pages/Upload";
import Analysis from "@/pages/Analysis";
import Biodiversity from "@/pages/Biodiversity";
import Alerts from "@/pages/Alerts";
import Reports from "@/pages/Reports";
import DigitalTwin from "@/pages/DigitalTwin";
import Fisheries from "@/pages/Fisheries";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="upload" element={<Upload />} />
            <Route path="analysis" element={<Analysis />} />
            <Route path="biodiversity" element={<Biodiversity />} />
            <Route path="alerts" element={<Alerts />} />
            <Route path="reports" element={<Reports />} />
            <Route path="digital-twin" element={<DigitalTwin />} />
            <Route path="fisheries" element={<Fisheries />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider >
);

export default App;
