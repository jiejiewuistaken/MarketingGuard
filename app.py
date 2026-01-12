from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OCR Image Editor</title>
    <style>
      :root {
        color-scheme: light;
      }
      body {
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: #f6f7fb;
        color: #222;
      }
      header {
        padding: 16px 20px;
        font-size: 18px;
        font-weight: 600;
        background: #101828;
        color: #fff;
      }
      .container {
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 16px;
        padding: 16px;
      }
      .panel {
        background: #fff;
        border: 1px solid #e2e5eb;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
      }
      label {
        display: block;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #344054;
      }
      input[type="file"] {
        margin-bottom: 12px;
      }
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
      }
      button {
        border: none;
        background: #2563eb;
        color: #fff;
        padding: 8px 14px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
      }
      button:disabled {
        background: #94a3b8;
        cursor: not-allowed;
      }
      #status {
        font-size: 12px;
        color: #475467;
        min-height: 18px;
        margin-bottom: 12px;
      }
      #image-stage {
        position: relative;
        border: 1px dashed #c7cbd3;
        border-radius: 8px;
        background: #fbfbfc;
        padding: 8px;
        display: inline-block;
        max-width: 100%;
      }
      #preview {
        display: block;
        max-width: 100%;
        height: auto;
        border-radius: 4px;
      }
      #overlay {
        position: absolute;
        left: 8px;
        top: 8px;
        pointer-events: none;
      }
      .ocr-box {
        position: absolute;
        border: 1px solid rgba(37, 99, 235, 0.5);
        background: rgba(255, 255, 255, 0.65);
        color: #111;
        padding: 1px 3px;
        overflow: hidden;
        pointer-events: auto;
        border-radius: 2px;
        min-width: 10px;
      }
      .ocr-box:focus {
        outline: 2px solid rgba(37, 99, 235, 0.9);
        background: rgba(255, 255, 255, 0.9);
      }
      textarea {
        width: 100%;
        min-height: 420px;
        resize: vertical;
        border: 1px solid #d5d9e0;
        border-radius: 8px;
        padding: 10px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
          "Courier New", monospace;
        font-size: 13px;
      }
      .hint {
        font-size: 12px;
        color: #667085;
        margin-top: 8px;
      }
      .hidden {
        display: none;
      }
      @media (max-width: 960px) {
        .container {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>OCR Image Editor</header>
    <div class="container">
      <section class="panel">
        <label for="file-input">Upload image (JPG or PNG)</label>
        <input id="file-input" type="file" accept="image/png,image/jpeg" />
        <div class="controls">
          <button id="ocr-btn">Run OCR</button>
          <button id="download-btn">Download edited image</button>
        </div>
        <div id="status"></div>
        <div id="image-stage">
          <img id="preview" alt="Preview" />
          <div id="overlay"></div>
        </div>
        <div class="hint">
          Tip: Click any blue box to edit its text. Download will redraw the edited
          text over the original image.
        </div>
      </section>
      <section class="panel">
        <label for="ocr-text">OCR result</label>
        <textarea id="ocr-text" placeholder="OCR output will appear here." readonly></textarea>
      </section>
    </div>
    <canvas id="download-canvas" class="hidden"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
    <script>
      const fileInput = document.getElementById("file-input");
      const preview = document.getElementById("preview");
      const overlay = document.getElementById("overlay");
      const ocrBtn = document.getElementById("ocr-btn");
      const downloadBtn = document.getElementById("download-btn");
      const statusEl = document.getElementById("status");
      const ocrText = document.getElementById("ocr-text");
      const downloadCanvas = document.getElementById("download-canvas");

      const OCR_LANG = "eng+chi_sim";
      let currentDataUrl = "";
      let currentScale = 1;
      let ocrBoxes = [];

      fileInput.addEventListener("change", handleFileSelect);
      ocrBtn.addEventListener("click", runOCR);
      downloadBtn.addEventListener("click", downloadEditedImage);
      window.addEventListener("resize", () => {
        if (!currentDataUrl) return;
        syncOverlay();
        renderBoxes(ocrBoxes);
      });

      function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) return;
        if (!["image/jpeg", "image/png"].includes(file.type)) {
          setStatus("Please upload a JPG or PNG image.");
          fileInput.value = "";
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          currentDataUrl = reader.result;
          preview.onload = () => {
            syncOverlay();
            clearOCR();
            setStatus("Image loaded. Click 'Run OCR' to start.");
          };
          preview.src = currentDataUrl;
        };
        reader.readAsDataURL(file);
      }

      function syncOverlay() {
        if (!preview.naturalWidth || !preview.naturalHeight) return;
        const displayWidth = preview.clientWidth;
        const displayHeight = preview.clientHeight;
        currentScale = displayWidth / preview.naturalWidth;
        overlay.style.width = displayWidth + "px";
        overlay.style.height = displayHeight + "px";
      }

      function clearOCR() {
        overlay.innerHTML = "";
        ocrText.value = "";
        ocrBoxes = [];
      }

      function setStatus(text) {
        statusEl.textContent = text;
      }

      async function runOCR() {
        if (!currentDataUrl) {
          setStatus("Please upload an image first.");
          return;
        }
        ocrBtn.disabled = true;
        setStatus("OCR running...");
        clearOCR();
        try {
          const result = await Tesseract.recognize(currentDataUrl, OCR_LANG, {
            logger: (message) => {
              if (message.status && message.progress !== undefined) {
                const percent = Math.round(message.progress * 100);
                setStatus(`${message.status} (${percent}%)`);
              }
            },
          });
          const data = result.data || {};
          ocrText.value = (data.text || "").trim();
          const lines = data.lines && data.lines.length ? data.lines : data.words || [];
          ocrBoxes = lines.filter((line) => line.text && line.text.trim());
          if (ocrBoxes.length) {
            renderBoxes(ocrBoxes);
            setStatus(`OCR done. ${ocrBoxes.length} editable box(es).`);
          } else {
            setStatus("OCR done. No text detected.");
          }
        } catch (error) {
          console.error(error);
          setStatus("OCR failed. Please try another image.");
        } finally {
          ocrBtn.disabled = false;
        }
      }

      function renderBoxes(lines) {
        overlay.innerHTML = "";
        const scale = currentScale || 1;
        lines.forEach((line) => {
          const box = line.bbox;
          if (!box) return;
          const width = Math.max(1, box.x1 - box.x0);
          const height = Math.max(1, box.y1 - box.y0);
          const el = document.createElement("div");
          el.className = "ocr-box";
          el.contentEditable = "true";
          el.spellcheck = false;
          el.dataset.x = box.x0;
          el.dataset.y = box.y0;
          el.dataset.w = width;
          el.dataset.h = height;
          el.style.left = box.x0 * scale + "px";
          el.style.top = box.y0 * scale + "px";
          el.style.width = width * scale + "px";
          el.style.height = height * scale + "px";
          el.style.fontSize = Math.max(10, height * scale * 0.8) + "px";
          el.textContent = line.text;
          overlay.appendChild(el);
        });
      }

      function downloadEditedImage() {
        if (!currentDataUrl || !preview.naturalWidth) {
          setStatus("Please upload an image first.");
          return;
        }
        const img = new Image();
        img.onload = () => {
          downloadCanvas.width = img.naturalWidth;
          downloadCanvas.height = img.naturalHeight;
          const ctx = downloadCanvas.getContext("2d");
          ctx.drawImage(img, 0, 0);
          const boxes = overlay.querySelectorAll(".ocr-box");
          boxes.forEach((box) => {
            const text = box.textContent.trim();
            if (!text) return;
            const x = parseFloat(box.dataset.x);
            const y = parseFloat(box.dataset.y);
            const w = parseFloat(box.dataset.w);
            const h = parseFloat(box.dataset.h);
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = "#111111";
            ctx.font = `${Math.max(10, h * 0.8)}px sans-serif`;
            ctx.textBaseline = "top";
            ctx.fillText(text, x, y);
          });
          downloadCanvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "edited.png";
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
          }, "image/png");
        };
        img.src = currentDataUrl;
      }
    </script>
  </body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        return


def run_server():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"Serving on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()