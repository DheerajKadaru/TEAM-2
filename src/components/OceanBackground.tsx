import { useEffect, useRef } from "react";

const OceanBackground = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      // Ensure video plays
      const playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise.catch((error) => {
          console.log("Video autoplay prevented:", error);
        });
      }
    }
  }, []);

  return (
    <>
      {/* Ocean video background */}
      <div className="ocean-video-bg">
        <video
          ref={videoRef}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          className="ocean-video-element"
        >
          <source src="/ocean.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        {/* Overlay for better text readability - reduced opacity */}
        <div className="ocean-video-overlay" />
      </div>

      {/* Fallback gradient background - only shows if video fails */}
      <div className="ocean-bg-fallback" />

      {/* Animated ocean bubbles effect */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        {/* Floating ocean bubbles */}
        {[...Array(15)].map((_, i) => (
          <div
            key={`bubble-${i}`}
            className="absolute rounded-full blur-[2px]"
            style={{
              width: Math.random() * 20 + 6,
              height: Math.random() * 20 + 6,
              left: `${Math.random() * 100}%`,
              bottom: '-50px',
              background: `radial-gradient(circle, hsla(200, 60%, 80%, 0.12), hsla(200, 50%, 70%, 0.04))`,
              animation: `bubbleRise ${Math.random() * 12 + 10}s linear infinite`,
              animationDelay: `${Math.random() * 5}s`,
              boxShadow: `0 0 ${Math.random() * 8 + 4}px hsla(200, 60%, 70%, 0.15)`,
            }}
          />
        ))}
      </div>


    </>
  );
};

export default OceanBackground;
