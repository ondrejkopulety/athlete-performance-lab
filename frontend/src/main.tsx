import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Fonty se hostují s aplikací, ne z Google CDN – dashboard běží za
// autentizací a nemá důvod hlásit se ven na cizí server.
import "@fontsource/dm-sans/400.css";
import "@fontsource/dm-sans/500.css";
import "@fontsource/dm-sans/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/700.css";

import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
