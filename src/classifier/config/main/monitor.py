from src.classifier.task import Main as _Main


class Main(_Main):
    def run(self, _):
        from src.classifier.monitor import Monitor
        from src.classifier.monitor.usage import Usage

        Usage.stop()
        try:
            Monitor.current()._listener[1].join()
        except KeyboardInterrupt:
            pass
