import React from "react";

export default function ShapeVisualizer({ shape, matched }) {
  const getShapeElement = () => {
    const size = 60;
    const center = size / 2;
    const style = {
      width: size,
      height: size,
      position: "relative",
      opacity: matched ? 0.7 : 1,
    };

    // Remove fallback to "Circle" for "Unknown"
    switch (shape) {
      case "Circle":
        return (
          <div style={style}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <circle
                cx={center}
                cy={center}
                r={size / 2 - 5}
                fill="#3B82F6"
                stroke="#1E40AF"
                strokeWidth="2"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
              Circle
            </span>
          </div>
        );
      case "Triangle":
        return (
          <div style={style}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <polygon
                points={`${center},5 ${size - 5},${size - 5} 5,${size - 5}`}
                fill="#22C55E"
                stroke="#166534"
                strokeWidth="2"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
              Triangle
            </span>
          </div>
        );
      case "Square":
        return (
          <div style={style}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <rect
                x="5"
                y="5"
                width={size - 10}
                height={size - 10}
                fill="#EF4444"
                stroke="#991B1B"
                strokeWidth="2"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
              Square
            </span>
          </div>
        );
      case "Pentagon":
        // Calculate pentagon points
        const points = [];
        for (let i = 0; i < 5; i++) {
          const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
          const radius = size / 2 - 7;
          const x = center + radius * Math.cos(angle);
          const y = center + radius * Math.sin(angle);
          points.push(`${x},${y}`);
        }

        return (
          <div style={style}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <polygon
                points={points.join(" ")}
                fill="#FACC15"
                stroke="#A16207"
                strokeWidth="2"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
              Pentagon
            </span>
          </div>
        );
      case "Unknown":
      default:
        // Render a gray circle with "Unknown" label
        return (
          <div style={style}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <circle
                cx={center}
                cy={center}
                r={size / 2 - 5}
                fill="#9CA3AF"
                stroke="#374151"
                strokeWidth="2"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
              Unknown
            </span>
          </div>
        );
    }
  };

  return getShapeElement();
}
