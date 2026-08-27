const fs = require('fs');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync('index.html', 'utf8');

const dom = new JSDOM(html, {
  url: "http://127.0.0.1:8080/chat",
  runScripts: "outside-only",
});

dom.window.fetch = async (url, options) => {
  console.log("Mock fetch called:", url);
  return {
    ok: true,
    json: async () => ({}),
    body: {
      getReader: () => {
        let done = false;
        return {
          read: async () => {
            if (done) return { done: true };
            done = true;
            return { value: Buffer.from("data: [DONE]\n\n"), done: false };
          }
        };
      }
    }
  };
};

// Now run scripts
const scriptEl = dom.window.document.querySelector("script");
dom.window.eval(scriptEl.textContent);

setTimeout(() => {
  console.log("Typing message...");
  const input = dom.window.document.getElementById('chat-input');
  input.value = "хай";
  
  console.log("Clicking send...");
  const btn = dom.window.document.getElementById('btn-send');
  btn.click();
}, 500);

setTimeout(() => {
  console.log("Done.");
  process.exit(0);
}, 2000);
