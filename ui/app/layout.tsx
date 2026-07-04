import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Roboss — Video Generator",
  description: "Generate videos with Gemini Omni Flash",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
