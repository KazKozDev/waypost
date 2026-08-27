import Cocoa
import SwiftUI
import WebKit

// WaypostBar — a native menu-bar app (like Ollama).
// Runs the waypost local server from venv as a subprocess and keeps
// its status in the menu bar: 🟢 running / 🟠 starting / 🔴 failed.

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
final class AppWindowController: NSObject, NSWindowDelegate {
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
        let wv = WKWebView(frame: .zero, configuration: config)
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
        win.titleVisibility = .visible
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

        let statusText = st == .running ? "Server: Running"
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
        let symbol = st == .running ? "circle.fill"
                   : st == .starting ? "circle.dashed"
                   : "circle"
        let color: NSColor = st == .running ? .systemGreen
                           : st == .starting ? .systemOrange
                           : st == .failed ? .systemRed
                           : .systemGray
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
            AppWindowController.shared.show(url: url, title: "Waypost")
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

    private var proc: Process?
    private let queue = DispatchQueue(label: "waypost.proc")

    let host = "127.0.0.1"
    let port = 8080
    var rootURL: String { "http://\(host):\(port)" }
    var baseURL: String { "http://\(host):\(port)/v1" }
    var chatURL: String { "http://\(host):\(port)/chat" }
    var dashboardURL: String { "http://\(host):\(port)/dashboard" }
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
        let task = URLSession.shared.dataTask(with: url) { [weak self] _, resp, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                let ok = (resp as? HTTPURLResponse)?.statusCode == 200
                if ok { self.state = .running }
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