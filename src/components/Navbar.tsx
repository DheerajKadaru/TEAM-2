import { Link, useLocation } from "react-router-dom";
import {
  Home,
  Upload,
  BarChart3,
  Leaf,
  AlertTriangle,
  FileText,
  Globe,
  Fish,
  Dna
} from "lucide-react";

const navItems = [
  { path: "/", label: "Home", icon: Home },
  { path: "/upload", label: "FASTQ Upload", icon: Upload },
  { path: "/analysis", label: "Analysis", icon: BarChart3 },
  { path: "/biodiversity", label: "Biodiversity", icon: Leaf },
  { path: "/alerts", label: "Alerts", icon: AlertTriangle },
  { path: "/reports", label: "Reports", icon: FileText },
  { path: "/digital-twin", label: "Digital Twin", icon: Globe },
  { path: "/fisheries", label: "Fisheries", icon: Fish },
];

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-40 glass-card border-b border-border/50 backdrop-blur-xl">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group relative">
            <div className="relative">
              <Dna className="w-8 h-8 text-primary animate-pulse group-hover:scale-110 transition-transform duration-300" style={{ filter: 'drop-shadow(0 0 12px hsla(199,89%,48%,0.5))' }} />
              <div className="absolute inset-0 bg-primary/40 blur-lg rounded-full animate-pulse" />
            </div>
            <div className="relative">
              <h1 className="text-xl font-display font-bold text-foreground group-hover:scale-105 transition-transform duration-300">
                S.A.G.A.R
              </h1>
              <p className="text-[10px] tracking-widest relative text-muted-foreground">
                MARINE eDNA PLATFORM
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
              </p>
            </div>
          </Link>

          {/* Navigation Links with water flow effect */}
          <div className="hidden lg:flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`nav-link flex items-center gap-2 relative group ${location.pathname === path ? "active" : ""
                  }`}
              >
                <Icon className="w-4 h-4 group-hover:scale-110 transition-transform duration-300" />
                <span className="relative">
                  {label}
                  {location.pathname === path && (
                    <span className="absolute -bottom-1 left-0 w-full h-0.5 bg-primary/50 blur-sm" />
                  )}
                </span>
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden lg:block">

            </div>

            {/* Mobile Menu */}
            <div className="lg:hidden flex items-center gap-2 overflow-x-auto scrollbar-hide">

              {navItems.map(({ path, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={`p-2 rounded-lg transition-all duration-300 relative ${location.pathname === path
                    ? "bg-primary/25 text-primary shadow-lg shadow-primary/25"
                    : "text-muted-foreground hover:text-primary hover:bg-primary/15"
                    }`}
                >
                  <Icon className="w-5 h-5" />
                  {location.pathname === path && (
                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/2 h-0.5 bg-primary rounded-full shadow-[0_0_8px_hsla(199,89%,48%,0.5)]" />
                  )}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
