import Cocoa
import SwiftUI
import WebKit

// WaypostBar — a native menu-bar app (like Ollama).
// Runs the waypost local server from venv as a subprocess and keeps
// its status in the menu bar: running / starting / failed.

// The project path is embedded at build time (see macos/build.sh) via
// generated/BuildConfig.swift:   let projectDir = "..."

@main
struct WaypostBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        // No window — only a menu-bar icon. The menu is built by AppDelegate.
        Settings { EmptyView() }
    }
}

// Controller for opening pages in a dedicated native macOS window.
final class AppWindowController: NSObject, NSWindowDelegate, WKUIDelegate, WKNavigationDelegate {
    static let shared = AppWindowController()
    private var window: NSWindow?
    private var webView: WKWebView?

    func show(url: URL, title: String = "Waypost") {
        let req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
        if let win = window, let wv = webView {
            wv.load(req)
            win.title = title
            win.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "developerExtrasEnabled")
        config.applicationNameForUserAgent = "WaypostApp/5.0"

        let script = """
        (function() {
            function applyMacApp() {
                if (document.documentElement) { document.documentElement.classList.add('macos-app'); }
                if (document.body) { document.body.classList.add('macos-app'); }
            }
            applyMacApp();
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', applyMacApp);
            }
        })();
        """
        let userScript = WKUserScript(
            source: script,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        config.userContentController.addUserScript(userScript)

        let wv = WKWebView(frame: .zero, configuration: config)
        wv.uiDelegate = self
        wv.navigationDelegate = self
        wv.load(req)
        self.webView = wv

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 960, height: 740),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        win.minSize = NSSize(width: 640, height: 500)
        win.center()
        win.title = title
        win.titlebarAppearsTransparent = true
        win.titleVisibility = .hidden
        win.isReleasedWhenClosed = false
        win.delegate = self
        win.contentView = wv

        self.window = win
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowWillClose(_ notification: Notification) {
        // Keep controller ready for next show() call
    }

    // MARK: - WKNavigationDelegate
    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        if let url = navigationAction.request.url {
            let isLocal = (url.host == "127.0.0.1" || url.host == "localhost")
            if navigationAction.navigationType == .linkActivated && (!isLocal || navigationAction.targetFrame == nil) {
                NSWorkspace.shared.open(url)
                decisionHandler(.cancel)
                return
            }
        }
        decisionHandler(.allow)
    }

    // MARK: - WKUIDelegate
    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if let url = navigationAction.request.url {
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping ([URL]?) -> Void
    ) {
        let openPanel = NSOpenPanel()
        openPanel.allowsMultipleSelection = parameters.allowsMultipleSelection
        openPanel.canChooseDirectories = parameters.allowsDirectories
        openPanel.canChooseFiles = true
        openPanel.resolvesAliases = true
        openPanel.title = "Choose Files to Attach"
        openPanel.prompt = "Attach"

        if let win = self.window {
            openPanel.beginSheetModal(for: win) { response in
                if response == .OK {
                    completionHandler(openPanel.urls)
                } else {
                    completionHandler(nil)
                }
            }
        } else {
            let response = openPanel.runModal()
            if response == .OK {
                completionHandler(openPanel.urls)
            } else {
                completionHandler(nil)
            }
        }
    }
}

// Backward-compatible alias
typealias ChatWindowController = AppWindowController

// AppDelegate holds the NSStatusItem and the subprocess. SwiftUI MenuBarExtra
// can too, but NSStatusItem gives direct control over the lifecycle
// and the icon without unnecessary re-creations.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var status: NSStatusItem!
    private let server = ServerController()
    private var poll: Timer?

    func applicationDidFinishLaunching(_ note: Notification) {
        // No Dock icon and no window switcher — menu bar only.
        NSApp.setActivationPolicy(.accessory)

        status = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.variableLength)
        rebuildMenu()

        server.onStateChange = { [weak self] in self?.stateChanged() }
        server.start()

        // Poll /health every 4 seconds — a cheap local call.
        poll = Timer.scheduledTimer(withTimeInterval: 4.0, repeats: true) {
            [weak self] _ in self?.server.checkHealth()
        }
    }

    func applicationWillTerminate(_ note: Notification) {
        poll?.invalidate()
        server.stop()      // do not leave the subprocess orphaned
    }

    private func stateChanged() { rebuildMenu() }

    private func rebuildMenu() {
        let menu = NSMenu()
        let st = server.state

        let openItem = NSMenuItem(title: "Open Waypost", action: #selector(openWaypost), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)

        menu.addItem(.separator())

        let modeDesc = server.lastMode == "cloud" ? "Cloud" : server.lastMode == "auto" ? "Auto" : "Local"
        let modelDesc = server.lastModel.isEmpty ? "" : " · \(server.lastModel)"
        let statusText = st == .running ? "Server: Running (\(modeDesc)\(modelDesc))"
                       : st == .starting ? "Server: Starting…"
                       : st == .stopped ? "Server: Stopped"
                       : "Server: Failed"
        let statusItem = NSMenuItem(title: statusText, action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        menu.addItem(statusItem)

        let copyItem = NSMenuItem(title: "Copy API URL", action: #selector(copyBase), keyEquivalent: "c")
        copyItem.target = self
        menu.addItem(copyItem)

        let logItem = NSMenuItem(title: "View Logs...", action: #selector(openLog), keyEquivalent: "l")
        logItem.target = self
        menu.addItem(logItem)

        menu.addItem(.separator())
        let quitItem = NSMenuItem(title: "Quit Waypost", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        status.menu = menu
        updateIcon()
    }

    private func updateIcon() {
        let st = server.state
        let mode = server.lastMode
        let symbol = st == .running ? "circle.fill"
                   : st == .starting ? "circle.dashed"
                   : "circle"

        let color: NSColor
        if st == .running {
            if mode == "local" {
                color = .systemGreen
            } else if mode == "cloud" {
                color = .systemBlue
            } else {
                color = .systemOrange // Auto / Smart router
            }
        } else if st == .starting {
            color = .systemOrange
        } else if st == .failed {
            color = .systemRed
        } else {
            color = .systemGray
        }

        let image = NSImage(
            systemSymbolName: symbol, accessibilityDescription: "waypost")
        image?.withSymbolConfiguration(
            NSImage.SymbolConfiguration(pointSize: 14, weight: .regular))
        // tint
        let tinted = image?.tinted(color)
        status.button?.image = tinted
        status.button?.image?.isTemplate = false
    }

    // --- actions ---
    @objc func openWaypost() {
        if let url = URL(string: server.chatURL) {
            AppWindowController.shared.show(url: url, title: "Waypost — Chat")
        }
    }
    @objc func openDashboard() {
        if let url = URL(string: server.dashboardURL) {
            AppWindowController.shared.show(url: url, title: "Waypost — Dashboard")
        }
    }
    @objc func openProviders() {
        if let url = URL(string: server.providersURL) {
            AppWindowController.shared.show(url: url, title: "Waypost — Providers & Keys")
        }
    }
    @objc func openSetup() {
        if let url = URL(string: server.setupURL) {
            AppWindowController.shared.show(url: url, title: "Waypost — Setup")
        }
    }
    @objc func copyBase() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(server.baseURL, forType: .string)
    }
    @objc func openLog() {
        NSWorkspace.shared.open(URL(fileURLWithPath: server.logPath))
    }
    @objc func quit() {
        NSApp.terminate(nil)
    }
}

// Light tint of an SF Symbol.
private extension NSImage {
    func tinted(_ color: NSColor) -> NSImage {
        let img = copy() as! NSImage
        img.lockFocus()
        color.set()
        NSRect(origin: .zero, size: img.size).fill(
            using: .sourceAtop)
        img.unlockFocus()
        return img
    }
}

// Server subprocess management.
final class ServerController {
    enum State { case starting, running, stopped, failed }
    var state: State = .stopped { didSet { onStateChange?() } }
    var onStateChange: (() -> Void)?

    var lastMode: String = "auto"
    var lastModel: String = "auto (Smart Router)"

    private var proc: Process?
    private let queue = DispatchQueue(label: "waypost.proc")

    let host = "127.0.0.1"
    let port = 8080
    var rootURL: String { "http://\(host):\(port)" }
    var baseURL: String { "http://\(host):\(port)/v1" }
    var chatURL: String { "http://\(host):\(port)/chat" }
    var dashboardURL: String { "http://\(host):\(port)/dashboard" }
    var providersURL: String { "http://\(host):\(port)/providers" }
    var setupURL: String { "http://\(host):\(port)/setup" }
    var logPath: String { projectDir + "/var/server.log" }

    func start() {
        queue.async { [weak self] in self?.startSync() }
    }

    private func startSync() {
        DispatchQueue.main.async { [weak self] in self?.state = .starting }
        // First make sure the port is free — maybe the server is already running.
        if isPortOpen(host, port) {
            DispatchQueue.main.async { [weak self] in self?.state = .running }
            return
        }

        let venvPython = projectDir + "/.venv/bin/python"
        let python = FileManager.default.fileExists(atPath: venvPython)
            ? venvPython : "/usr/bin/env python3"

        FileManager.default.createFile(atPath: logPath, contents: nil)
        guard let log = FileHandle(forWritingAtPath: logPath) else {
            DispatchQueue.main.async { [weak self] in self?.state = .failed }
            return
        }
        log.seekToEndOfFile()

        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.currentDirectoryURL = URL(fileURLWithPath: projectDir)
        p.arguments = ["-m", "waypost.server"]
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { [weak self] proc in
            // Code 3 — instance lock: the server is already up in another
            // process (terminal/second window). This is not a crash.
            DispatchQueue.main.async {
                guard let self else { return }
                if self.state == .stopped { return }
                if proc.terminationStatus == 3 {
                    // Another instance is serving the port — so we are running.
                    self.state = self.isPortOpen(self.host, self.port) ? .running : .failed
                } else {
                    self.state = .failed
                }
            }
        }
        do {
            try p.run()
            proc = p
        } catch {
            DispatchQueue.main.async { [weak self] in self?.state = .failed }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, let p = self.proc, p.isRunning else { return }
            p.terminate()
            // give 3 seconds for a clean exit, otherwise SIGKILL
            usleep(useconds_t(3 * 1_000_000))
            if p.isRunning { kill(p.processIdentifier, SIGKILL) }
            DispatchQueue.main.async { self.state = .stopped }
        }
    }

    func restart() {
        stop()
        queue.asyncAfter(deadline: .now() + 3.5) { [weak self] in
            self?.startSync()
        }
    }

    func checkHealth() {
        guard state != .stopped else { return }
        let url = URL(string: baseURL.replacingOccurrences(of: "/v1", with: "")
                      + "/health")!
        let task = URLSession.shared.dataTask(with: url) { [weak self] data, resp, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                let ok = (resp as? HTTPURLResponse)?.statusCode == 200
                if ok {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        let mode = json["last_mode"] as? String ?? "local"
                        let model = json["last_model"] as? String ?? ""
                        if self.lastMode != mode || self.lastModel != model {
                            self.lastMode = mode
                            self.lastModel = model
                            self.onStateChange?()
                        }
                    }
                    if self.state != .running { self.state = .running }
                }
                else if self.state == .running { self.state = .starting }
            }
        }
        task.resume()
    }

    private func isPortOpen(_ host: String, _ port: Int) -> Bool {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { close(sock) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        inet_pton(AF_INET, host, &addr.sin_addr)
        let r = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return r == 0
    }
}