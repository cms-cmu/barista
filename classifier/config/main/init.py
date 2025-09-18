from classifier.task import ArgParser, main


class Main(main.Main):
    argparser = ArgParser(
        prog="init",
        description="Initializa all tasks for testing.",
        workflow=[
            ("main", "[blue]task()[/blue] initialize task"),
        ],
    )

    def run(self, _): ...
