def recursive_interrupt():
    import signal

    import psutil

    p = psutil.Process()
    for child in p.children(recursive=True):
        child.send_signal(signal.SIGINT)
