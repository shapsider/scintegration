import os
import pathlib
import tempfile
from abc import abstractmethod
from typing import Any, Iterable, List, Mapping, Optional

import dill
import ignite
import torch

from .config import config
from .utils import DelayedKeyboardInterrupt, logged

EPOCH_STARTED = ignite.engine.Events.EPOCH_STARTED
EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
ITERATION_COMPLETED = ignite.engine.Events.ITERATION_COMPLETED
EXCEPTION_RAISED = ignite.engine.Events.EXCEPTION_RAISED
COMPLETED = ignite.engine.Events.COMPLETED


@logged
class Trainer:

    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self.required_losses: List[str] = []

    @abstractmethod
    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:

        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:

        raise NotImplementedError  # pragma: no cover

    def report_metrics(
            self, train_state: ignite.engine.State,
            val_state: Optional[ignite.engine.State]
    ) -> None:

        if train_state.epoch % config.PRINT_LOSS_INTERVAL:
            return
        train_metrics = {
            key: float(f"{val:.3f}")
            for key, val in train_state.metrics.items()
        }
        val_metrics = {
            key: float(f"{val:.3f}")
            for key, val in val_state.metrics.items()
        } if val_state else None
        
        print("[Epoch {} has been trained]".format(train_state.epoch))

        ######################################################################
        self.logger.info(
            "[Epoch %d] train=%s, val=%s, %.1fs elapsed",
            train_state.epoch, 
            train_metrics, 
            val_metrics,
            train_state.times["EPOCH_COMPLETED"]  # Also includes validator time
        )

    def fit(
            self, train_loader: Iterable, val_loader: Optional[Iterable] = None,
            max_epochs: int = 100, random_seed: int = 0,
            directory: Optional[os.PathLike] = None,
            plugins: Optional[List["TrainingPlugin"]] = None
    ) -> None:

        interrupt_delayer = DelayedKeyboardInterrupt()
        directory = pathlib.Path(directory or tempfile.mkdtemp(prefix=config.TMP_PREFIX))
        print("Using training directory: \"%s\"", directory)

        # Construct engines
        train_engine = ignite.engine.Engine(self.train_step)
        val_engine = ignite.engine.Engine(self.val_step) if val_loader else None

        delay_interrupt = interrupt_delayer.__enter__
        train_engine.add_event_handler(EPOCH_STARTED, delay_interrupt)
        train_engine.add_event_handler(COMPLETED, delay_interrupt)

        # Exception handling
        train_engine.add_event_handler(ITERATION_COMPLETED, ignite.handlers.TerminateOnNan())

        @train_engine.on(EXCEPTION_RAISED)
        def _handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and config.ALLOW_TRAINING_INTERRUPTION:
                self.logger.info("Stopping training due to user interrupt...")
                engine.terminate()
            else:
                raise e

        # Compute metrics
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(train_engine, item)
            if val_engine:
                ignite.metrics.Average(
                    output_transform=lambda output, item=item: output[item]
                ).attach(val_engine, item)

        if val_engine:
            @train_engine.on(EPOCH_COMPLETED)
            def _validate(engine):
                val_engine.run(
                    val_loader, max_epochs=engine.state.epoch
                )  # Bumps max_epochs by 1 per training epoch, so validator resumes for 1 epoch

        @train_engine.on(EPOCH_COMPLETED)
        def _report_metrics(engine):
            self.report_metrics(engine.state, val_engine.state if val_engine else None)

        for plugin in plugins or []:
            plugin.attach(
                net=self.net, trainer=self,
                train_engine=train_engine, val_engine=val_engine,
                train_loader=train_loader, val_loader=val_loader,
                directory=directory
            )

        restore_interrupt = lambda: interrupt_delayer.__exit__(None, None, None)
        train_engine.add_event_handler(EPOCH_COMPLETED, restore_interrupt)
        train_engine.add_event_handler(COMPLETED, restore_interrupt)

        # Start engines
        torch.manual_seed(random_seed)
        train_engine.run(train_loader, max_epochs=max_epochs)

        torch.cuda.empty_cache()  # Works even if GPU is unavailable

    def get_losses(self, loader: Iterable) -> Mapping[str, float]:

        engine = ignite.engine.Engine(self.val_step)
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(engine, item)
        engine.run(loader, max_epochs=1)
        torch.cuda.empty_cache()  # Works even if GPU is unavailable
        return engine.state.metrics

    def state_dict(self) -> Mapping[str, Any]:

        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        r"""
        Load state from a state dict

        Parameters
        ----------
        state_dict
            State dict
        """


@logged
class Model:

    NET_TYPE = torch.nn.Module
    TRAINER_TYPE = Trainer

    def __init__(self, *args, **kwargs) -> None:
        self._net = self.NET_TYPE(*args, **kwargs)
        self._trainer: Optional[Trainer] = None  # Constructed upon compile

    @property
    def net(self) -> torch.nn.Module:

        return self._net

    @property
    def trainer(self) -> Trainer:

        if self._trainer is None:
            raise RuntimeError(
                "No trainer has been registered! "
                "Please call `.compile()` first."
            )
        return self._trainer

    def compile(self, *args, **kwargs) -> None:

        if self._trainer:
            self.logger.warning(
                "`compile` has already been called. "
                "Previous trainer will be overwritten!"
            )
        self._trainer = self.TRAINER_TYPE(self.net, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:

        self.trainer.fit(*args, **kwargs)

    def get_losses(self, *args, **kwargs) -> Mapping[str, float]:
        return self.trainer.get_losses(*args, **kwargs)

    def save(self, fname: os.PathLike) -> None:

        fname = pathlib.Path(fname)
        trainer_backup, self._trainer = self._trainer, None
        device_backup, self.net.device = self.net.device, torch.device("cpu")
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)
        self.net.device = device_backup
        self._trainer = trainer_backup

    def upgrade(self) -> None:
        r"""
        Upgrade the model if generated by older versions
        """

@logged
class TrainingPlugin:

    @abstractmethod
    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:

        raise NotImplementedError  # pragma: no cover
