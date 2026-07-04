import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./components/AppLayout.jsx";
import Studio from "./pages/Studio.jsx";
import Analytics from "./pages/Stats.jsx";
import Monitor from "./pages/Monitor.jsx";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Navigate to="/studio" replace />} />
          <Route path="/studio" element={<Studio />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="*" element={<Navigate to="/studio" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
