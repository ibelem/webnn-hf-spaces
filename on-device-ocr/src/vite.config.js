import { defineConfig } from "vite";
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs}',
          dest: 'onnxruntime-web-1-23-2-dists' // Copies to <output dir>/onnxruntime
        }
      ]
    })
  ],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    target: "esnext",
    outDir: "../",
  }
});
