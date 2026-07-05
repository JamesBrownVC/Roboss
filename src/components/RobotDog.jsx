export default function RobotDog({ running = false, className = "", title }) {
  const ariaProps = title ? { role: "img", "aria-label": title } : { "aria-hidden": "true" };

  return (
    <img
      src="/robot-dog-transparent.png"
      alt={title || "Roboss Robot Dog"}
      {...ariaProps}
      className={`roboss-robot-dog ${running ? "roboss-dog-running" : ""} object-contain ${className}`}
      draggable="false"
    />
  );
}
