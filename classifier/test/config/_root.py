import sys

from classifier.config.dataset import _root
from classifier.task import ArgParser, converter


class LoadGroupedRootForTest(_root.LoadGroupedRoot):
    argparser = ArgParser()
    argparser.add_argument(
        "--test-max-entries",
        type=converter.int_pos,
        default=sys.maxsize,
        help="max number of entries to load",
    )
    argparser.add_argument(
        "--test-max-entries-per-group",
        type=converter.int_pos,
        default=sys.maxsize,
        help="max number of entries to load per group",
    )
    argparser.add_argument(
        "--test-max-entries-per-file",
        type=converter.int_pos,
        default=sys.maxsize,
        help="max number of entries to load per file",
    )

    def _from_root(self):
        from concurrent.futures import ProcessPoolExecutor

        from classifier.monitor.progress import Progress
        from classifier.process import pool, status

        files = self.files
        with ProcessPoolExecutor(
            max_workers=self.opts.max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor:
            with Progress.new(
                total=sum(map(lambda x: len(x), self.files.values())),
                msg=("files", "Fetching metadata"),
            ) as progress:
                files = {
                    k: pool.map_async(
                        executor,
                        _root._fetch(tree=self.opts.tree),
                        v,
                        callbacks=[lambda _: _root.progress_advance(progress)],
                        preserve_order=True,
                    )
                    for k, v in files.items()
                }
                total = self.opts.test_max_entries
                max_chunk = self.opts.test_max_entries_per_file
                for k, v in files.items():
                    chunks = []
                    total_group = self.opts.test_max_entries_per_group
                    for chunk in v:
                        size = min(total, total_group, max_chunk, len(chunk))
                        chunks.append(chunk.slice(0, size))
                        total -= size
                        total_group -= size
                        if total == 0 or total_group == 0:
                            break
                    if chunks:
                        yield self.from_root(k), chunks
                    if total == 0:
                        break
