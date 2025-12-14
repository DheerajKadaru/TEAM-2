import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import SeaSageChatbot from "./SeaSageChatbot";
import OceanBackground from "./OceanBackground";
import { AlertPopupManager } from "./AlertPopupManager";

const Layout = () => {
  return (
    <div className="min-h-screen relative">
      <OceanBackground />
      <AlertPopupManager />
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <Outlet />
      </main>
      <SeaSageChatbot />
    </div>
  );
};

export default Layout;
