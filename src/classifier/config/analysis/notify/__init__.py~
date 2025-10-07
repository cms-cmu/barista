from dataclasses import dataclass

from classifier.task import Analysis, ArgParser, parse

from ...setting import monitor as cfg


class Gmail(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--title",
        help="title of email",
        required=True,
    )
    argparser.add_argument(
        "--body",
        help="body of email",
        default=None,
    )
    argparser.add_argument(
        "--body-from-result",
        nargs="+",
        metavar=("CLASS", "KWARGS"),
        help="a class to generate the body from results",
        default=None,
    )
    argparser.add_argument(
        "--type",
        choices=["plain", "html"],
        default="plain",
        help="type of email",
    )
    argparser.add_argument(
        "--labels",
        nargs="+",
        default=[],
        action="extend",
        help="labels added to the title",
    )

    def __init__(self):
        super().__init__()

        self.enabled = (
            cfg.Gmail.address
            and cfg.Gmail.password
            and (self.opts.body is not None or self.opts.body_from_result is not None)
        )

    def analyze(self, results: list[dict]):
        if not self.enabled:
            return []
        body = self.opts.body
        if body is None:
            body = parse.instance(self.opts.body_from_result)(results)
        title = self.opts.title
        if self.opts.labels:
            title = "".join(map("[{}] ".format, self.opts.labels)) + title
        return [_send_gmail(title=title, body=body, subtype=self.opts.type)]


@dataclass
class _send_gmail:
    title: str
    body: str
    subtype: str

    def __call__(self):
        from classifier.monitor.notification.smtp_gmail import gmail_send_text

        gmail_send_text(self.title, self.body, self.subtype)
