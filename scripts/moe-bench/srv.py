"""Start/stop llama-server on its own hidden console so it can be stopped with a real Ctrl-C
(graceful shutdown, which makes the MoE cache print its hit statistics).

usage: python -I srv.py start LOGFILE -- <llama-server.exe and args...>   -> prints PID
       python -I srv.py stop PID [timeout_s]
"""
import ctypes, subprocess, sys, time

k32 = ctypes.windll.kernel32

def start(log, cmd):
    f = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    print(p.pid)

def alive(pid):
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code)); k32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE

def stop(pid, timeout):
    if not alive(pid):
        return
    k32.FreeConsole()
    if not k32.AttachConsole(pid):
        sys.exit(f"AttachConsole failed: {ctypes.GetLastError()}")
    k32.SetConsoleCtrlHandler(None, True)      # ignore the event in this helper
    ok = k32.GenerateConsoleCtrlEvent(0, 0)    # CTRL_C_EVENT to every process on that console
    if not ok:
        print(f"GenerateConsoleCtrlEvent failed: {ctypes.GetLastError()}", file=sys.stderr)
    for _ in range(timeout * 10):
        if not alive(pid):
            k32.FreeConsole(); return
        time.sleep(0.1)
    k32.FreeConsole()
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    print("forced", file=sys.stderr)

if sys.argv[1] == "start":
    start(sys.argv[2], sys.argv[sys.argv.index("--") + 1:])
else:
    stop(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
