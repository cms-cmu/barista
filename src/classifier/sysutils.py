def recursive_interrupt():
    import signal

    import psutil

    p = psutil.Process()
    children = p.children(recursive=True)

    for child in children:
        try:
            child.send_signal(signal.SIGINT)
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(children, timeout=3)

    for p_alive in alive:
        try:
            p_alive.kill()
        except psutil.NoSuchProcess:
            pass
