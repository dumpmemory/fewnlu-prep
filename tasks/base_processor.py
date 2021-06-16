from abc import ABC, abstractmethod
from typing import List

from methods.utils import InputExample

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
DEV32_SET = "dev32"
UNLABELED_SET = "unlabeled"
AUGMENTED_SET = "augmented"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, DEV32_SET, UNLABELED_SET, AUGMENTED_SET]

ProcessorOutputPattern = List[InputExample]

class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development/dev32 and unlabeled examples
    for a given task.
    """

    @abstractmethod
    def get_train_examples(self, data_dir, use_cloze: bool) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_dev32_examples(self, data_dir, use_cloze: bool) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the dev32 set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_augmented_examples(self, data_dir) -> ProcessorOutputPattern:
        """Get a collection of `InputExample`s for the augmented data set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @abstractmethod
    def _create_examples(self, *args, **kwargs):
        pass