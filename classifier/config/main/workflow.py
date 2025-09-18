import shlex
from collections import defaultdict

import fsspec
import yaml
from classifier.task import ArgParser, EntryPoint, main
from classifier.task.parse._dict import _mapping_schema, mapping
from classifier.utils import YamlIndentSequence
from yaml.representer import Representer

from .. import setting as cfg

_MAX_WIDTH = 10


def _merge_args(opts: list[str]) -> list[str]:
    if (len(opts) <= 1) or (not opts[0].startswith(main._DASH)):
        return opts
    else:
        merged = []
        current = [opts[0]]
        for opt in opts[1:]:
            if (len(opt) > _MAX_WIDTH) or (opt != shlex.quote(opt)):
                if len(current) > 0:
                    merged.append(shlex.join(current))
                    current.clear()
                merged.append(opt)
            else:
                current.append(opt)
        if len(current) > 0:
            merged.append(shlex.join(current))
        return merged


class Main(main.Main):
    _no_state = True
    _no_init = True

    argparser = ArgParser(
        prog="workflow",
        description="Generate a workflow file from command line arguments",
        workflow=[
            ("main", "generate workflow"),
        ],
    )
    argparser.add_argument(
        "path",
        help="output path to workflow file",
    )
    argparser.add_argument(
        "--embed-file",
        help="embed mappings or sequences read from file",
        action="store_true",
    )
    argparser.add_argument(
        "main",
        help="main task",
    )

    @classmethod
    def prelude(cls):
        cfg.Analysis.enable = False
        cfg.Monitor.enable = False

    def run(self, parser: EntryPoint):
        from src.storage.eos import EOS

        output = EOS(self.opts.path)
        workflow = defaultdict(list)
        args: list[str] = [*parser.args[main._MAIN][1]]
        for k in ["--embed-file", self.opts.path, self.opts.main]:
            if k in args:
                args.remove(k)
        workflow[main._MAIN] = self._parse_opts(self.opts.main, args)
        for k in parser._tasks:
            for mod, opts in parser.args[k]:
                workflow[k].append(self._parse_opts(mod, opts))

        yaml.add_representer(defaultdict, Representer.represent_dict)
        with fsspec.open(output, "wt") as f:
            yaml.dump(workflow, f, sort_keys=False, Dumper=YamlIndentSequence)

    def _parse_opts(self, mod: str, opts: list[str]):
        output = {main._MODULE: mod}
        merged = []
        group = []
        for opt in opts:
            schema = _mapping_schema(opt)[0]
            if (schema is not None) or (opt.startswith(main._DASH)):
                merged.extend(_merge_args(group))
                group.clear()
            if (
                schema is not None
                and (schema != "py")
                and (self.opts.embed_file or schema != "file")
            ):
                merged.append(mapping(opt))
            else:
                group.append(opt)
        merged.extend(_merge_args(group))
        if len(merged) > 0:
            output[main._OPTION] = merged
        return output
