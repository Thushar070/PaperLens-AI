import type { Metadata } from "next";
import type { ReactNode } from "react";
import { AppShell } from "./components/app-shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "PaperLens — Research Paper Discovery",
  description: "Browse, classify, and search computer science research papers with PaperLens.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return <html lang="en"><body><AppShell>{children}</AppShell></body></html>;
}
