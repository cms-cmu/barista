import logging
import os

from classifier.task import ArgParser, EntryPoint, converter
from classifier.task.special import WorkInProgress

from .. import setting as cfg
from ._utils import LoadTrainingSets, SelectDevice

_PROFILE_DEFAULT = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}
_PROFILE_SCHEDULE_DEFAULT = {
    "wait": 0,
    "warmup": 0,
    "active": 1,
}


class Main(WorkInProgress, SelectDevice, LoadTrainingSets):
    argparser = ArgParser(
        prog="profiler",
        description="""Run torch profiler on models.
[1] https://pytorch.org/blog/understanding-gpu-memory-1/
[2] https://pytorch.org/blog/understanding-gpu-memory-2/""",
        workflow=[
            *LoadTrainingSets._workflow,
        ],
    )
    argparser.add_argument(
        "--dataset-size",
        type=converter.int_pos,
        default=128,
        help="size of dataset",
    )
    argparser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "CUDA"],
        help="profiling activities",
    )

    def run(self, parser: EntryPoint):
        logging.warning("[red]Profiler is currently hardcoded and for test only.[/red]")
        import numpy as np
        import torch.nn.functional as F
        import torch.optim as optim
        from classifier.nn.blocks.HCR import HCR
        from torch.profiler import ProfilerActivity, profile, schedule
        from torch.utils.data import DataLoader, Subset

        # load datasets in parallel
        dataset = self.load_training_sets(parser)
        datasets = Subset(
            dataset, np.random.choice(len(dataset), size=self.opts.dataset_size)
        )
        batch = next(iter(DataLoader(datasets, batch_size=self.opts.dataset_size)))
        input = (
            batch["CanJet"].cuda(),
            batch["NotCanJet"].cuda(),
            batch["ancillary"].cuda(),
        )
        truth = batch["label"].cuda()
        # train models in sequence
        model = HCR(8, 8, [*range(4)], "attention", "cuda", nClasses=4).to("cuda")
        opt = optim.Adam(model.parameters(), lr=0.01)

        with profile(
            activities=[
                getattr(ProfilerActivity, a) for a in self.opts.profile_activities
            ],
            schedule=schedule(**_PROFILE_SCHEDULE_DEFAULT),
            **_PROFILE_DEFAULT,
        ) as p:
            model.updateMeanStd(*input)
            model.initMeanStd()
            model.train()
            for _ in range(10):
                opt.zero_grad()
                c, _ = model(*input)
                loss = F.cross_entropy(c, truth)
                loss.backward()
                opt.step()
        logging.info("Exporting timeline for model...")
        p.export_memory_timeline(os.fspath(cfg.IO.report / "profiler.html"))
        logging.info("Exporting trace for model...")
        p.export_chrome_trace(os.fspath(cfg.IO.report / "profiler.json"))
