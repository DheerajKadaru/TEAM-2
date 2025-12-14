import { Link } from "react-router-dom";
import { Upload, BarChart3, Leaf, AlertTriangle, FileText, Globe, Fish, Dna, ArrowRight } from "lucide-react";

const features = [
  { path: "/upload", label: "FASTQ Upload", icon: Upload, desc: "Upload eDNA sequencing data" },
  { path: "/analysis", label: "Analysis", icon: BarChart3, desc: "View ASV & taxonomy results" },
  { path: "/biodiversity", label: "Biodiversity", icon: Leaf, desc: "Ecological indices & metrics" },
  { path: "/alerts", label: "Conservation Alerts", icon: AlertTriangle, desc: "Novel & endangered species" },
  { path: "/reports", label: "Reports", icon: FileText, desc: "Download PDF/CSV reports" },
  { path: "/digital-twin", label: "Digital Twin", icon: Globe, desc: "Real-time marine tracking" },
  { path: "/fisheries", label: "Fisheries", icon: Fish, desc: "Sustainable catch data" },
];

const Home = () => {
  return (
    <div className="space-y-12">
      {/* Hero Section with Liquid Glass Boxes */}
      <section className="relative py-16 text-center overflow-hidden">
        {/* Liquid Glass Box - Center background */}
        <div className="relative z-10 p-8 rounded-3xl mx-auto w-fit max-w-4xl liquid-glass-box-hero border-white/5">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="relative group">
              <Dna className="w-16 h-16 text-primary animate-pulse group-hover:scale-110 transition-transform duration-300" style={{ filter: 'drop-shadow(0 0 15px hsla(199,89%,48%,0.5))' }} />
              <div className="absolute inset-0 bg-primary/40 blur-xl rounded-full animate-pulse" />
            </div>
            <h1 className="text-5xl md:text-7xl font-display font-bold text-foreground relative tracking-tight drop-shadow-2xl">
              S.A.G.A.R
            </h1>
          </div>
          <p className="text-2xl max-w-2xl mx-auto mb-6 font-bold text-muted-foreground drop-shadow-lg">
            <span className="text-primary drop-shadow-[0_0_10px_rgba(0,180,255,0.5)]">S</span>mart{" "}
            <span className="text-primary drop-shadow-[0_0_10px_rgba(0,180,255,0.5)]">A</span>quatic{" "}
            <span className="text-primary drop-shadow-[0_0_10px_rgba(0,180,255,0.5)]">G</span>enomic{" "}
            <span className="text-primary drop-shadow-[0_0_10px_rgba(0,180,255,0.5)]">A</span>nalysis &{" "}
            <span className="text-primary drop-shadow-[0_0_10px_rgba(0,180,255,0.5)]">R</span>esearch
          </p>
          <p className="text-lg max-w-xl mx-auto mb-10 relative font-medium text-muted-foreground/90 leading-relaxed drop-shadow-md">
            Advanced marine eDNA analysis platform for biodiversity monitoring,
            species detection, and ocean conservation.
          </p>
          <Link to="/upload" className="btn-ocean inline-flex items-center gap-2 relative group px-8 py-4 text-lg">
            <span className="relative z-10">Get Started</span>
            <ArrowRight className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform" />
            <div className="absolute inset-0 bg-primary/40 blur-2xl group-hover:blur-3xl transition-all opacity-60 group-hover:opacity-100" />
          </Link>
        </div>
      </section>

      {/* Features Grid with Ocean Theme */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {features.map(({ path, label, icon: Icon, desc }, i) => (
          <Link
            key={path}
            to={path}
            className="glass-card p-6 group hover:border-primary/50 transition-all duration-300 relative overflow-hidden hover:scale-105 hover:shadow-2xl"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            {/* Liquid glass effect overlay */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
              <div
                className="absolute inset-0"
                style={{
                  background: `linear-gradient(135deg, transparent 0%, hsla(200, 70%, 70%, 0.15) 50%, transparent 100%)`,
                  backgroundSize: '200% 200%',
                  animation: 'waterFlow 3s ease-in-out infinite',
                }}
              />
            </div>

            <div className="flex items-start gap-4 relative z-10">
              <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-all duration-300 relative border border-primary/30">
                <Icon className="w-6 h-6 text-primary group-hover:scale-110 transition-transform" style={{ filter: 'drop-shadow(0 0 8px hsla(199,89%,48%,0.5))' }} />
              </div>
              <div className="flex-1">
                <h3 className="font-display font-semibold text-card-foreground group-hover:text-primary transition-colors relative">
                  {label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
                </h3>
                <p className="text-sm mt-1 text-muted-foreground group-hover:text-foreground transition-colors">{desc}</p>
              </div>
            </div>
          </Link>
        ))}
      </section>


    </div>
  );
};

export default Home;
